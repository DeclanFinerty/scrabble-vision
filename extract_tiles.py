"""
Extract real tile images from board photos using saved corners and ground truth.

For each board with both corners and ground truth, warps the board,
segments into 15x15 grid, and saves each lettered cell to
data/real_tiles/{LETTER}/{board}_{row}_{col}.png

Usage:
    uv run extract_tiles.py
    uv run extract_tiles.py --output data/real_tiles
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
    per_letter = defaultdict(int)
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
            if letter in (".", "_"):
                continue

            letter_dir = output_dir / letter
            letter_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{stem}_{row:02d}_{col:02d}.png"
            cv2.imwrite(str(letter_dir / filename), cell_img)

            per_letter[letter] += 1
            board_count += 1

        per_board[stem] = board_count
        total_extracted += board_count
        print(f"  {stem}: {board_count} tiles extracted")

    print(f"\nTotal: {total_extracted} tiles from {len(per_board)} boards")
    print(f"\nPer letter:")
    for letter in sorted(per_letter):
        print(f"  {letter}: {per_letter[letter]}")

    print(f"\nPer board:")
    for board in sorted(per_board):
        print(f"  {board}: {per_board[board]}")


if __name__ == "__main__":
    main()
