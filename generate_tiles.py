"""
Generate synthetic Scrabble tile images for training.

28 classes: A-Z letters on tile backgrounds, EMPTY (plain board squares),
BONUS (colored bonus squares with printed text).

Run:
    uv run generate_tiles.py
    uv run generate_tiles.py --samples 500 --output data/train
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random

from src.classification.model import CLASSES


POINT_VALUES = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4,
    "I": 1, "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3,
    "Q": 10, "R": 1, "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8,
    "Y": 4, "Z": 10,
}

TILE_BG_LOW = (160, 185, 200)
TILE_BG_HIGH = (195, 220, 235)

# Board surface colors (BGR) — varied for light and dark boards
BOARD_COLORS = [
    (170, 190, 180),    # light cream
    (140, 160, 150),    # medium
    (90, 100, 95),      # dark board
    (60, 70, 65),       # very dark
    (180, 195, 190),    # warm light
    (110, 120, 115),    # medium dark
]

# Bonus square colors (BGR) — varied saturation and brightness
BONUS_COLORS = {
    "TW": [(60, 60, 180), (40, 40, 140), (80, 80, 200), (30, 30, 100)],
    "DW": [(140, 130, 190), (120, 110, 170), (160, 150, 210), (90, 80, 140)],
    "TL": [(160, 130, 80), (130, 100, 60), (180, 150, 100), (100, 80, 50)],
    "DL": [(180, 160, 120), (150, 130, 90), (200, 180, 140), (120, 100, 70)],
}

BONUS_TEXTS = [
    "DOUBLE", "TRIPLE", "WORD", "LETTER", "SCORE",
    "DW", "TW", "DL", "TL",
]


def random_tile_bg() -> np.ndarray:
    return np.array([
        random.randint(TILE_BG_LOW[i], TILE_BG_HIGH[i]) for i in range(3)
    ], dtype=np.uint8)


def render_tile(letter: str, size: int = 64) -> np.ndarray:
    """Render a single tile image for the given class."""
    if letter == "EMPTY":
        return _render_empty(size)
    if letter == "BONUS":
        return _render_bonus(size)
    return _render_letter(letter, size)


def _render_letter(letter: str, size: int) -> np.ndarray:
    img = np.full((size, size, 3), random_tile_bg(), dtype=np.uint8)

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

    if letter in POINT_VALUES:
        pts = str(POINT_VALUES[letter])
        small_scale = font_scale * 0.35
        small_thick = max(1, thickness // 2)
        cv2.putText(img, pts,
                    (size - size // 4, size - size // 10),
                    font, small_scale, letter_color, small_thick)

    return img


def _render_empty(size: int) -> np.ndarray:
    """Plain board square — varied brightness for light/dark boards."""
    base = random.choice(BOARD_COLORS)
    color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in base)
    img = np.full((size, size, 3), color, dtype=np.uint8)

    # Slight texture variation
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def _render_bonus(size: int) -> np.ndarray:
    """Colored bonus square with simulated printed text."""
    bonus_type = random.choice(list(BONUS_COLORS.keys()))
    base = random.choice(BONUS_COLORS[bonus_type])
    color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in base)
    img = np.full((size, size, 3), color, dtype=np.uint8)

    # Draw small text to simulate "DOUBLE WORD SCORE" etc.
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_lines = random.randint(1, 3)
    for i in range(n_lines):
        text = random.choice(BONUS_TEXTS)
        scale = random.uniform(0.2, 0.35)
        thickness = 1
        text_color = tuple(max(0, min(255, c + random.randint(-40, 40)))
                           for c in color)
        y = int(size * (0.3 + 0.2 * i)) + random.randint(-3, 3)
        x = random.randint(2, max(3, size // 6))
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

    return img


def augment_tile(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    sigma = random.uniform(3, 15)
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

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


def generate_dataset(output_dir: str | Path, samples_per_class: int = 500,
                     tile_size: int = 64):
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
