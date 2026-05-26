"""
Visual + numeric sanity check on the augmented training data.

Picks N random rows from a corner_manifest.csv (default
data/augmented/corner_manifest.csv; pass --manifest to point at a
per-run augmented set under training/corner/runs/run_NNN/augmented/),
draws the manifest's normalized corner coords on the 224x224 image,
and saves to training/corner/sanity_check/. Also reports min/max of
all coordinates so we can confirm the distribution is sensible
(mostly [0,1] with a tail of negatives / >1 from crops that
excluded a corner).

If the colored dots land on the actual board corners in every
saved image, the training labels are correct end-to-end. If they
don't, the augmentation math is producing bad labels and the model
can't possibly learn from them.

Corner colors: TL=red, TR=green, BR=blue, BL=yellow (matches
preview_augmentations.py).

Usage:
    uv run scripts/sanity_check_corners.py
    uv run scripts/sanity_check_corners.py --count 10 --seed 0
    uv run scripts/sanity_check_corners.py --manifest training/corner/runs/run_024/augmented/corner_manifest.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MANIFEST = Path("data/augmented/corner_manifest.csv")
OUT_DIR = Path("training/corner/sanity_check")

# Same palette as preview_augmentations.py.
COLOR_TL = (0, 0, 255)      # red
COLOR_TR = (0, 255, 0)      # green
COLOR_BR = (255, 0, 0)      # blue
COLOR_BL = (0, 255, 255)    # yellow
LINE_COLOR = (255, 255, 255)
CORNER_COLORS = [COLOR_TL, COLOR_TR, COLOR_BR, COLOR_BL]
COORD_KEYS = ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"]


def draw(image: np.ndarray, corners_px: np.ndarray) -> np.ndarray:
    out = image.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in corners_px]
    for i in range(4):
        cv2.line(out, pts[i], pts[(i + 1) % 4], LINE_COLOR, 1,
                 lineType=cv2.LINE_AA)
    for p, color in zip(pts, CORNER_COLORS):
        cv2.circle(out, p, 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p, 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--count", type=int, default=5,
                        help="Number of random samples (default 5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: non-deterministic).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"Path to a corner_manifest.csv "
                             f"(default {DEFAULT_MANIFEST}).")
    args = parser.parse_args()

    manifest = args.manifest
    if not manifest.is_file():
        print(f"Manifest not found: {manifest}")
        print("Run scripts/augment_corners.py first, or pass --manifest "
              "pointing at a per-run augmented set.")
        sys.exit(1)

    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Empty manifest: {manifest}")
        sys.exit(1)

    xs, ys = [], []
    for r in rows:
        for k in COORD_KEYS:
            v = float(r[k])
            (xs if k.endswith("_x") else ys).append(v)
    xs_a = np.array(xs); ys_a = np.array(ys)
    n_x_in = int(((xs_a >= 0) & (xs_a <= 1)).sum())
    n_y_in = int(((ys_a >= 0) & (ys_a <= 1)).sum())

    print(f"Manifest:  {manifest}  ({len(rows)} rows)")
    print(f"  X range:        [{xs_a.min():+.4f}, {xs_a.max():+.4f}]   "
          f"mean {xs_a.mean():+.4f}")
    print(f"  Y range:        [{ys_a.min():+.4f}, {ys_a.max():+.4f}]   "
          f"mean {ys_a.mean():+.4f}")
    print(f"  X in [0,1]:     {n_x_in}/{len(xs)}  "
          f"({100 * n_x_in / len(xs):.1f}%)")
    print(f"  Y in [0,1]:     {n_y_in}/{len(ys)}  "
          f"({100 * n_y_in / len(ys):.1f}%)")
    print()

    rng = random.Random(args.seed)
    picks = rng.sample(rows, min(args.count, len(rows)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(picks)} annotated samples to {OUT_DIR}/")
    for i, r in enumerate(picks, start=1):
        path = Path(r["image_path"])
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [{i}/{len(picks)}] could not read {path}")
            continue
        h, w = img.shape[:2]
        coords_px = np.array([
            [float(r["tl_x"]) * w, float(r["tl_y"]) * h],
            [float(r["tr_x"]) * w, float(r["tr_y"]) * h],
            [float(r["br_x"]) * w, float(r["br_y"]) * h],
            [float(r["bl_x"]) * w, float(r["bl_y"]) * h],
        ], dtype=np.float32)
        out_path = OUT_DIR / path.name
        cv2.imwrite(str(out_path), draw(img, coords_px))
        norms = ", ".join(f"{k}={float(r[k]):+.3f}" for k in COORD_KEYS)
        print(f"  [{i}/{len(picks)}] {path.name}  ({w}x{h})")
        print(f"      {norms}")
        print(f"      source: {r['source_image']}")


if __name__ == "__main__":
    main()
