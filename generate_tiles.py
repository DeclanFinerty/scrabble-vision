"""
Generate synthetic Scrabble tile images for training.

Renders each letter on a tile-colored background with augmentations:
- Rotation, brightness/contrast jitter, Gaussian noise, blur
- Multiple font scales to simulate distance variation

Run:
    python generate_tiles.py
    python generate_tiles.py --samples 1000 --output data/train
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random

from src.classification.model import CLASSES


# Scrabble tile point values
POINT_VALUES = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4,
    "I": 1, "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3,
    "Q": 10, "R": 1, "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8,
    "Y": 4, "Z": 10,
}

# Tile background color range (beige/cream, BGR)
TILE_BG_LOW = (160, 185, 200)
TILE_BG_HIGH = (195, 220, 235)


def random_tile_bg() -> np.ndarray:
    """Random beige tile background color."""
    return np.array([
        random.randint(TILE_BG_LOW[i], TILE_BG_HIGH[i]) for i in range(3)
    ], dtype=np.uint8)


def render_tile(letter: str, size: int = 64) -> np.ndarray:
    """Render a single clean tile image."""
    img = np.full((size, size, 3), random_tile_bg(), dtype=np.uint8)

    # Draw the letter centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = size / 45.0 * random.uniform(0.85, 1.15)
    thickness = max(1, int(size / 25 * random.uniform(0.8, 1.2)))

    text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
    x = (size - text_size[0]) // 2 + random.randint(-2, 2)
    y = (size + text_size[1]) // 2 - size // 10 + random.randint(-2, 2)

    letter_color = (random.randint(15, 60),
                    random.randint(15, 50),
                    random.randint(10, 45))
    cv2.putText(img, letter, (x, y), font, font_scale, letter_color, thickness)

    # Draw point value subscript
    if letter in POINT_VALUES:
        pts = str(POINT_VALUES[letter])
        small_scale = font_scale * 0.35
        small_thick = max(1, thickness // 2)
        cv2.putText(img, pts,
                    (size - size // 4, size - size // 10),
                    font, small_scale, letter_color, small_thick)

    return img


def augment_tile(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a tile image."""
    h, w = img.shape[:2]

    # Rotation (-10 to +10 degrees)
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness/contrast
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gaussian noise
    sigma = random.uniform(3, 15)
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Random shadow (darken one side)
    if random.random() < 0.2:
        shadow = np.ones_like(img, dtype=np.float32)
        direction = random.choice(["left", "right", "top", "bottom"])
        if direction == "left":
            shadow[:, :w // 3] *= random.uniform(0.6, 0.85)
        elif direction == "right":
            shadow[:, 2 * w // 3:] *= random.uniform(0.6, 0.85)
        elif direction == "top":
            shadow[:h // 3, :] *= random.uniform(0.6, 0.85)
        else:
            shadow[2 * h // 3:, :] *= random.uniform(0.6, 0.85)
        img = (img.astype(np.float32) * shadow).clip(0, 255).astype(np.uint8)

    return img


def generate_dataset(
    output_dir: str | Path,
    samples_per_class: int = 500,
    tile_size: int = 64,
):
    """Generate full synthetic training dataset."""
    output_dir = Path(output_dir)
    total = 0

    for cls in CLASSES:
        class_dir = output_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            tile = render_tile(cls, size=tile_size)
            tile = augment_tile(tile)
            cv2.imwrite(str(class_dir / f"{cls}_{i:04d}.png"), tile)
            total += 1

        print(f"  {cls:6s}: {samples_per_class} images")

    print(f"\nTotal: {total} images across {len(CLASSES)} classes -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Scrabble tile data")
    parser.add_argument("--output", default="data/train", help="Output directory")
    parser.add_argument("--samples", type=int, default=500, help="Samples per class")
    parser.add_argument("--size", type=int, default=64, help="Tile render size (px)")
    args = parser.parse_args()

    generate_dataset(args.output, args.samples, args.size)