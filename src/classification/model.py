"""
Lightweight CNN for classifying Scrabble tile images.

26 classes: A-Z. The tile detector handles empty cells, so the
classifier only sees tiles that are already known to have a letter.

Architecture is deliberately small (~200K params) targeting <1ms per tile
on CPU. Full board (225 tiles batched) should be <50ms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

# Class labels — order MUST match ImageFolder's alphabetical folder sort.
# Only letters: tile detector handles empty cells, classifier only sees tiles.
CLASSES = sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
NUM_CLASSES = len(CLASSES)  # 26

# Input size for the classifier
TILE_INPUT_SIZE = 32  # 32x32 grayscale


class TileClassifier(nn.Module):
    """
    Small CNN for single-tile classification.

    Input:  (batch, 1, 32, 32) grayscale tile image
    Output: (batch, 26) class logits
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # -> 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # -> 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),        # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def preprocess_tiles(tile_images: list[np.ndarray]) -> torch.Tensor:
    """
    Preprocess a list of cell images into a batched tensor.

    Args:
        tile_images: list of BGR or grayscale numpy arrays (any size).

    Returns:
        Tensor of shape (B, 1, 32, 32), normalized to [0, 1].
    """
    batch = []
    for img in tile_images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (TILE_INPUT_SIZE, TILE_INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        batch.append(img)

    tensor = torch.tensor(np.array(batch), dtype=torch.float32)
    tensor = tensor.unsqueeze(1)  # (B, 1, 32, 32)
    return tensor


def predict_tiles(
    model: TileClassifier,
    tile_images: list[np.ndarray],
    device: str = "cpu",
) -> list[tuple[str, float]]:
    """
    Classify a batch of tile images.

    Returns:
        List of (predicted_class, confidence) tuples.
    """
    model.eval()
    tensor = preprocess_tiles(tile_images).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidences, indices = probs.max(dim=1)

    results = []
    for idx, conf in zip(indices.cpu().numpy(), confidences.cpu().numpy()):
        results.append((CLASSES[idx], float(conf)))

    return results


def load_model(path: str | Path, device: str = "cpu") -> TileClassifier:
    """Load a trained model from disk."""
    model = TileClassifier()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def export_onnx(model: TileClassifier, path: str | Path):
    """Export model to ONNX for Rust/mobile deployment."""
    model.eval()
    dummy = torch.randn(1, 1, TILE_INPUT_SIZE, TILE_INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["tile"],
        output_names=["logits"],
        dynamic_axes={"tile": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX model exported to {path}")