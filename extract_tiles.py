"""
Extract real tile images from board photos using saved corners and ground truth.

Saves each cell to data/real_tiles/{CLASS}/:
  - Letters (A-Z): cells with letters in ground truth
  - BONUS: empty cells at bonus square positions (TW, DW, TL, DL)
  - EMPTY: empty cells at normal positions

Usage:
    uv run extract_tiles.py
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

from src.detection.grid_detect import (
    detect_grid, extract_cell_images, load_corners, GRID_SIZE
)


BOARDS_DIR = Path("data/raw_boards")
CORNERS_DIR = BOARDS_DIR / "corners"
GT_DIR = BOARDS_DIR / "ground_truth"

# Bonus square positions (same as every standard Scrabble board)
BONUS_SQUARES = set()
for r, c in [(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)]:
    BONUS_SQUARES.add((r,c))
for r, c in [(1,1),(2,2),(3,3),(4,4),(10,10),(11,11),(12,12),(13,13),
             (1,13),(2,12),(3,11),(4,10),(10,4),(11,3),(12,2),(13,1),(7,7)]:
    BONUS_SQUARES.add((r,c))
for r, c in [(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),
             (9,1),(9,5),(9,9),(9,13),(13,5),(13,9)]:
    BONUS_SQUARES.add((r,c))
for r, c in [(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),
             (6,2),(6,6),(6,8),(6,12),(7,3),(7,11),
             (8,2),(8,6),(8,8),(8,12),
             (11,0),(11,7),(11,14),(12,6),(12,8),(14,3),(14,11)]:
    BONUS_SQUARES.add((r,c))


def load_ground_truth(path: Path) -> list[list[str]] | None:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if len(lines) != GRID_SIZE:
        return None
    board = []
    for line in lines:
        if len(line) != GRID_SIZE:
            return None
        board.append(list(line))
    return board


def main():
    parser = argparse.ArgumentParser(description="Extract real tiles from board images")
    parser.add_argument("--output", default="data/real_tiles")
    args = parser.parse_args()

    output_dir = Path(args.output)

    images = sorted(
        p for p in BOARDS_DIR.iterdir()
        if p.suffix.lower() in (".jpeg", ".jpg", ".png")
    )

    if not images:
        print(f"No images found in {BOARDS_DIR}")
        return

    total_extracted = 0
    per_class = defaultdict(int)
    per_board = defaultdict(int)

    for img_path in images:
        stem = img_path.stem
        corners_path = CORNERS_DIR / f"{stem}.npy"
        gt_path = GT_DIR / f"{stem}.txt"

        if not corners_path.exists():
            print(f"  {stem}: no saved corners, skipping")
            continue
        if not gt_path.exists():
            print(f"  {stem}: no ground truth, skipping")
            continue

        truth = load_ground_truth(gt_path)
        if truth is None:
            print(f"  {stem}: invalid ground truth, skipping")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  {stem}: could not read image, skipping")
            continue

        corners = load_corners(str(corners_path))
        grid = detect_grid(image, corners=corners)
        cell_images = extract_cell_images(grid)

        board_count = 0
        for row, col, cell_img in cell_images:
            letter = truth[row][col]

            if letter == "_":
                continue

            if letter == ".":
                cls = "BONUS" if (row, col) in BONUS_SQUARES else "EMPTY"
            else:
                cls = letter

            cls_dir = output_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{stem}_{row:02d}_{col:02d}.png"
            cv2.imwrite(str(cls_dir / filename), cell_img)

            per_class[cls] += 1
            board_count += 1

        per_board[stem] = board_count
        total_extracted += board_count
        print(f"  {stem}: {board_count} cells extracted")

    print(f"\nTotal: {total_extracted} cells from {len(per_board)} boards")
    print(f"\nPer class:")
    for cls in sorted(per_class):
        print(f"  {cls}: {per_class[cls]}")

    print(f"\nPer board:")
    for board in sorted(per_board):
        print(f"  {board}: {per_board[board]}")


if __name__ == "__main__":
    main()
