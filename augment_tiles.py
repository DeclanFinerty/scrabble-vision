"""
Augment real tile images to balance classes and increase training data.

Reads originals from data/real_tiles/, generates augmented variants
to equalize class counts, and saves them alongside the originals.

The class with the most images sets the target count. Classes with
fewer images get more augmentations per image, giving rare letters
more diversity.

Usage:
    uv run augment_tiles.py
    uv run augment_tiles.py --input data/real_tiles --target-multiplier 1.0
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from collections import defaultdict


def augment(img: np.ndarray) -> np.ndarray:
    """Apply random augmentation to a tile image."""
    h, w = img.shape[:2]

    # Random crop (5-15% from random edges)
    crop_pct = random.uniform(0.05, 0.15)
    top = int(h * random.uniform(0, crop_pct))
    bottom = int(h * random.uniform(0, crop_pct))
    left = int(w * random.uniform(0, crop_pct))
    right = int(w * random.uniform(0, crop_pct))
    img = img[top:h - bottom, left:w - right]
    img = cv2.resize(img, (w, h))

    # Rotation (-12 to +12 degrees)
    angle = random.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Slight perspective skew
    if random.random() < 0.5:
        skew = random.uniform(0.02, 0.08)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = w * skew * random.choice([-1, 1])
        dy = h * skew * random.choice([-1, 1])
        corner = random.randint(0, 3)
        pts2 = pts1.copy()
        pts2[corner] += np.float32([dx, dy])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (w, h),
                                   borderMode=cv2.BORDER_REFLECT)

    # Brightness shift (0.6-1.4)
    alpha = random.uniform(0.6, 1.4)
    beta = random.uniform(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Saturation shift
    if len(img.shape) == 3 and random.random() < 0.4:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gaussian noise
    sigma = random.uniform(3, 15)
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def main():
    parser = argparse.ArgumentParser(description="Augment real tile images")
    parser.add_argument("--input", default="data/real_tiles")
    parser.add_argument("--target-multiplier", type=float, default=1.0,
                        help="Multiply the target count (max class size) by this")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        print("Run extract_tiles.py first.")
        return

    # Count originals per class (exclude previously augmented)
    class_originals = {}
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        originals = [f for f in class_dir.iterdir()
                     if f.suffix == ".png" and "_aug" not in f.stem]
        class_originals[class_dir.name] = originals

    if not class_originals:
        print("No classes found.")
        return

    # Remove old augmented images
    removed = 0
    for class_name, _ in class_originals.items():
        class_dir = input_dir / class_name
        for f in class_dir.glob("*_aug*.png"):
            f.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} old augmented images")

    max_count = max(len(imgs) for imgs in class_originals.values())
    target = int(max_count * args.target_multiplier)
    print(f"Target per class: {target} (max original: {max_count})")

    total_generated = 0

    for class_name in sorted(class_originals):
        originals = class_originals[class_name]
        n_orig = len(originals)

        if n_orig == 0:
            print(f"  {class_name}: 0 originals, skipping")
            continue

        n_needed = max(0, target - n_orig)
        augs_per_image = n_needed // n_orig if n_orig > 0 else 0
        remainder = n_needed % n_orig if n_orig > 0 else 0

        generated = 0
        for i, img_path in enumerate(originals):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            n_augs = augs_per_image + (1 if i < remainder else 0)
            for j in range(n_augs):
                aug_img = augment(img)
                aug_name = f"{img_path.stem}_aug{j:03d}.png"
                cv2.imwrite(str(img_path.parent / aug_name), aug_img)
                generated += 1

        total_generated += generated
        total = n_orig + generated
        print(f"  {class_name}: {n_orig} original + {generated} augmented = {total}")

    print(f"\nTotal augmented images generated: {total_generated}")

    # Final counts
    print(f"\nFinal counts per class:")
    for class_name in sorted(class_originals):
        class_dir = input_dir / class_name
        count = len(list(class_dir.glob("*.png")))
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
