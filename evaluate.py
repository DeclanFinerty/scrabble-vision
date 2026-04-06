"""
Evaluate the trained classifier on a test set.

Produces per-class accuracy and a confusion summary for the
most-confused letter pairs (e.g., I vs L, O vs Q).

Run:
    python evaluate.py --data data/test --model models/tile_classifier.pt
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import numpy as np
from collections import defaultdict

from src.classification.model import (
    TileClassifier, TILE_INPUT_SIZE, CLASSES, NUM_CLASSES
)


def evaluate(args):
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((TILE_INPUT_SIZE, TILE_INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    model = TileClassifier()
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()

    # Confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for true, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true][pred] += 1

    # Per-class accuracy
    print(f"\n{'Class':>8s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>8s}")
    print("-" * 38)

    total_correct = 0
    total_samples = 0

    for i, cls in enumerate(CLASSES):
        correct = confusion[i][i]
        total = confusion[i].sum()
        acc = correct / total if total > 0 else 0.0
        total_correct += correct
        total_samples += total
        marker = " <--" if acc < 0.9 and total > 0 else ""
        print(f"{cls:>8s}  {correct:>8d}  {total:>6d}  {acc:>7.1%}{marker}")

    overall = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n{'OVERALL':>8s}  {total_correct:>8d}  {total_samples:>6d}  {overall:>7.1%}")

    # Most confused pairs
    print(f"\nTop confused pairs:")
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and confusion[i][j] > 0:
                confusions.append((confusion[i][j], CLASSES[i], CLASSES[j]))

    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:10]:
        print(f"  {true_cls} -> {pred_cls}: {count} mistakes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tile classifier")
    parser.add_argument("--data", default="data/test")
    parser.add_argument("--model", default="models/tile_classifier.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate(args)