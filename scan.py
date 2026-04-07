"""
Full pipeline: board photo -> 15x15 letter grid.

Usage:
    python scan.py --image board.jpg
    python scan.py --image board.jpg --interactive
    python scan.py --image board.jpg --interactive --debug
    python scan.py --image board.jpg --corners corners.npy
    python scan.py --image board.jpg --interactive --debug --words
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path

from src.detection.grid_detect import (
    detect_grid, extract_cell_images, debug_grid,
    save_corners, load_corners, GRID_SIZE
)
from src.classification.model import (
    load_model, predict_tiles, classify_result
)


def scan_board(
    image_path: str,
    model_path: str = "models/tile_classifier.pt",
    confidence_threshold: float = 0.7,
    device: str = "cpu",
    interactive: bool = False,
    corners: np.ndarray = None,
    debug: bool = False,
) -> dict:
    timings = {}

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ── Grid detection ──
    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=interactive)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    if grid.method == "manual" and debug:
        save_corners(grid.corners, "corners.npy")

    if debug:
        debug_grid(image, grid, "debug_grid.jpg")

    cell_images = extract_cell_images(grid)

    # ── Classification (all 225 cells in one pass) ──
    t0 = time.perf_counter()

    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    uncertain = []
    tile_count = 0

    model = load_model(model_path, device=device)
    all_imgs = [img for _, _, img in cell_images]
    predictions = predict_tiles(model, all_imgs, device=device)

    for (row, col, _), (label, conf) in zip(cell_images, predictions):
        mapped = classify_result(label)
        board[row][col] = mapped
        confidence[row][col] = conf
        if mapped != ".":
            tile_count += 1
            if conf < confidence_threshold:
                uncertain.append((row, col))

    timings["classify_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "board": board,
        "confidence": confidence,
        "uncertain": uncertain,
        "tile_count": tile_count,
        "empty_count": GRID_SIZE * GRID_SIZE - tile_count,
        "grid_method": grid.method,
        "timings": timings,
    }


def print_board(result: dict):
    board = result["board"]
    uncertain_set = set(map(tuple, result["uncertain"]))

    print(f"\n{'=' * 50}")
    print(f"Grid: {result['grid_method']} | "
          f"Tiles: {result['tile_count']} | "
          f"Empty: {result['empty_count']}")
    print(f"Time: {result['timings']['total_ms']:.0f}ms "
          f"(grid {result['timings']['grid_ms']:.0f} + "
          f"classify {result['timings']['classify_ms']:.0f})")
    print(f"{'=' * 50}")

    print("    " + " ".join(f"{i:2d}" for i in range(GRID_SIZE)))
    print("    " + "-" * (GRID_SIZE * 3))

    for r in range(GRID_SIZE):
        row_str = f"{r:2d} |"
        for c in range(GRID_SIZE):
            cell = board[r][c]
            if (r, c) in uncertain_set:
                row_str += f" {cell}?"
            else:
                row_str += f" {cell} "
        print(row_str)

    if result["uncertain"]:
        print(f"\n{len(result['uncertain'])} uncertain:")
        for r, c in result["uncertain"]:
            conf = result["confidence"][r][c]
            print(f"  ({r},{c}): '{board[r][c]}' @ {conf:.1%}")


def board_to_words(board: list[list[str]]) -> list[tuple]:
    words = []

    for r in range(GRID_SIZE):
        word, start_c = "", 0
        for c in range(GRID_SIZE):
            ch = board[r][c]
            if ch not in (".", "?"):
                if not word:
                    start_c = c
                word += ch
            else:
                if len(word) >= 2:
                    words.append((word, "across", r, start_c))
                word = ""
        if len(word) >= 2:
            words.append((word, "across", r, start_c))

    for c in range(GRID_SIZE):
        word, start_r = "", 0
        for r in range(GRID_SIZE):
            ch = board[r][c]
            if ch not in (".", "?"):
                if not word:
                    start_r = r
                word += ch
            else:
                if len(word) >= 2:
                    words.append((word, "down", start_r, c))
                word = ""
        if len(word) >= 2:
            words.append((word, "down", start_r, c))

    return words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan a Scrabble board")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/tile_classifier.pt")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--corners", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--words", action="store_true")
    args = parser.parse_args()

    corners = None
    if args.corners:
        corners = load_corners(args.corners)

    result = scan_board(
        args.image, args.model,
        confidence_threshold=args.threshold,
        device=args.device,
        interactive=args.interactive,
        corners=corners,
        debug=args.debug,
    )
    print_board(result)

    if args.words:
        words = board_to_words(result["board"])
        print(f"\nWords found ({len(words)}):")
        for word, direction, r, c in sorted(words):
            print(f"  {word:15s} {direction:6s} @ ({r},{c})")
