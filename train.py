"""
Train the tile classifier on synthetic + real tile data.

Combines synthetic data (data/train/) and real tile data (data/real_tiles/)
with configurable oversampling of real images. EMPTY class is excluded —
the tile detector handles empty cells, the classifier only sees A-Z.

Usage:
    uv run train.py
    uv run train.py --real-weight 3.0 --epochs 30
    uv run train.py --synthetic-only
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
from collections import defaultdict
import argparse
import time
import random

from src.classification.model import (
    TileClassifier, TILE_INPUT_SIZE, NUM_CLASSES, CLASSES, export_onnx
)


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((TILE_INPUT_SIZE, TILE_INPUT_SIZE)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((TILE_INPUT_SIZE, TILE_INPUT_SIZE)),
            transforms.ToTensor(),
        ])


class CombinedDataset(Dataset):
    """Combines synthetic and real datasets with source tracking."""

    def __init__(self, synthetic_dataset, real_dataset=None,
                 class_to_idx=None):
        self.samples = []  # (img_path, class_idx, source)
        self.class_to_idx = class_to_idx or {}
        self.transform = None

        self._add_imagefolder(synthetic_dataset, "synthetic")
        if real_dataset is not None:
            self._add_imagefolder(real_dataset, "real")

    def _add_imagefolder(self, dataset, source):
        for path, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            if class_name == "EMPTY":
                continue
            if class_name in self.class_to_idx:
                mapped_idx = self.class_to_idx[class_name]
                self.samples.append((path, mapped_idx, source))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx, source = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_idx, source


def split_real_by_board(samples, val_fraction=0.2):
    """Split real samples so each board is represented in val."""
    by_board = defaultdict(list)
    for i, (path, _, source) in enumerate(samples):
        if source != "real":
            continue
        stem = Path(path).stem
        board = "_".join(stem.split("_")[:2])  # e.g. board_001
        by_board[board].append(i)

    val_indices = set()
    for board, indices in by_board.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_indices.update(indices[:n_val])

    return val_indices


def train_model(args):
    device = torch.device(args.device)

    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    # Load datasets
    synthetic_dir = Path(args.synthetic_data)
    real_dir = Path(args.real_data)

    synthetic_ds = datasets.ImageFolder(str(synthetic_dir))
    real_ds = None
    if real_dir.exists() and not args.synthetic_only:
        # Check if real_tiles has class folders
        has_classes = any(d.is_dir() for d in real_dir.iterdir())
        if has_classes:
            real_ds = datasets.ImageFolder(str(real_dir))
            print(f"Real tiles: {len(real_ds)} images")

    combined = CombinedDataset(synthetic_ds, real_ds, class_to_idx)
    print(f"Combined dataset: {len(combined)} images ({NUM_CLASSES} classes)")

    # Count by source
    n_synthetic = sum(1 for _, _, s in combined.samples if s == "synthetic")
    n_real = sum(1 for _, _, s in combined.samples if s == "real")
    print(f"  Synthetic: {n_synthetic}, Real: {n_real}")

    # Split: all synthetic → train, real → 80/20 split by board
    real_val_indices = split_real_by_board(combined.samples)

    train_indices = []
    val_indices = []
    for i, (_, _, source) in enumerate(combined.samples):
        if i in real_val_indices:
            val_indices.append(i)
        else:
            train_indices.append(i)

    # Build sample weights for oversampling real data
    weights = []
    for i in train_indices:
        _, _, source = combined.samples[i]
        weights.append(args.real_weight if source == "real" else 1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(train_indices),
                                     replacement=True)

    # Create data loaders with transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    class SubsetWithTransform(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            path, class_idx, source = self.dataset.samples[self.indices[idx]]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, class_idx, source

    train_set = SubsetWithTransform(combined, train_indices, train_transform)
    val_set = SubsetWithTransform(combined, val_indices, val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2,
                            pin_memory=True) if val_indices else None

    n_train_real = sum(1 for i in train_indices
                       if combined.samples[i][2] == "real")
    n_train_synth = len(train_indices) - n_train_real
    n_val_real = len(val_indices)
    print(f"Train: {len(train_indices)} ({n_train_synth} synthetic + "
          f"{n_train_real} real, weight={args.real_weight}x)")
    print(f"Val: {len(val_indices)} (real tiles, split by board)")
    print(f"Device: {device}\n")

    model = TileClassifier().to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Output classes: {NUM_CLASSES} ({', '.join(CLASSES)})\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        real_correct = 0
        real_total = 0
        synth_correct = 0
        synth_total = 0
        t0 = time.time()

        for images, labels, sources in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_mask = predicted.eq(labels)
            train_correct += correct_mask.sum().item()
            train_total += labels.size(0)

            for j, src in enumerate(sources):
                if src == "real":
                    real_total += 1
                    real_correct += int(correct_mask[j])
                else:
                    synth_total += 1
                    synth_correct += int(correct_mask[j])

        # Validate
        val_acc = 0.0
        val_real_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, sources in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step()

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        synth_acc = synth_correct / synth_total if synth_total > 0 else 0.0
        real_acc_train = real_correct / real_total if real_total > 0 else 0.0
        elapsed = time.time() - t0

        parts = [
            f"Epoch {epoch + 1:3d}/{args.epochs}",
            f"Loss: {train_loss / train_total:.4f}",
            f"Train: {train_acc:.3f}",
            f"Synth: {synth_acc:.3f}",
        ]
        if real_total > 0:
            parts.append(f"Real(train): {real_acc_train:.3f}")
        if val_loader:
            parts.append(f"Val(real): {val_acc:.3f}")
        parts.append(f"{elapsed:.1f}s")
        print(" | ".join(parts))

        # Save best model based on val accuracy (or train if no val)
        check_acc = val_acc if val_loader else train_acc
        if check_acc > best_val_acc:
            best_val_acc = check_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

    metric_name = "val" if val_loader else "train"
    print(f"\nBest {metric_name} accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")

    # Export ONNX
    onnx_path = args.output.replace(".pt", ".onnx")
    best_model = TileClassifier()
    best_model.load_state_dict(torch.load(args.output, map_location="cpu",
                                          weights_only=True))
    export_onnx(best_model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scrabble tile classifier")
    parser.add_argument("--synthetic-data", default="data/train")
    parser.add_argument("--real-data", default="data/real_tiles")
    parser.add_argument("--real-weight", type=float, default=3.0,
                        help="Oversample weight for real tiles vs synthetic")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Train on synthetic data only")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/tile_classifier.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_model(args)