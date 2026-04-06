"""
Train the tile classifier on synthetic (and optionally real) data.

Run:
    python train.py
    python train.py --data data/train --epochs 30 --device cuda
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import argparse
import time

from src.classification.model import (
    TileClassifier, TILE_INPUT_SIZE, NUM_CLASSES, CLASSES, export_onnx
)


def get_transforms(train: bool = True):
    """Image transforms for training or evaluation."""
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


def train(args):
    """Train the classifier."""
    device = torch.device(args.device)

    # Load dataset
    full_dataset = datasets.ImageFolder(args.data, transform=get_transforms(train=True))
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")

    # Verify class ordering matches our expected order
    for i, cls in enumerate(full_dataset.classes):
        assert cls == CLASSES[i], (
            f"Class mismatch at index {i}: expected '{CLASSES[i]}', got '{cls}'. "
            f"Folder names must match expected class names exactly."
        )

    # Split into train/val (90/10)
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Override transform for validation split
    val_dataset = datasets.ImageFolder(args.data, transform=get_transforms(train=False))
    val_set_proper = torch.utils.data.Subset(val_dataset, val_set.indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set_proper, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Device: {device}\n")

    # Model
    model = TileClassifier().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        elapsed = time.time() - t0

        print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss / train_total:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")

    # Export ONNX
    onnx_path = args.output.replace(".pt", ".onnx")
    best_model = TileClassifier()
    best_model.load_state_dict(torch.load(args.output, map_location="cpu",
                                          weights_only=True))
    export_onnx(best_model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scrabble tile classifier")
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/tile_classifier.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(args)