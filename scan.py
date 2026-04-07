"""
Full pipeline: board photo -> 15x15 letter grid.

Usage:
    # Auto-detect board corners
    python scan.py --image board.jpg

    # Interactively click 4 corners (recommended for best results)
    python scan.py --image board.jpg --interactive

    # Save debug images to see what the pipeline is doing
    python scan.py --image board.jpg --interactive --debug

    # Reuse previously saved corners
    python scan.py --image board.jpg --corners corners.npy

    # Also extract detected words
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
from src.detection.tile_detect import (
    detect_tile_presence, debug_tile_detection
)
from src.classification.model import load_model, predict_tiles


def scan_board(
    image_path: str,
    model_path: str = "models/tile_classifier.pt",
    confidence_threshold: float = 0.7,
    device: str = "cpu",
    interactive: bool = False,
    corners: np.ndarray = None,
    debug: bool = False,
) -> dict:
    """
    Full scan pipeline.

    Args:
        image_path: Path to board photo.
        model_path: Path to trained classifier weights.
        confidence_threshold: Flag cells below this as uncertain.
        device: 'cpu' or 'cuda'.
        interactive: Open corner selector GUI.
        corners: Pre-specified corners (4x2 array).
        debug: Save debug images.

    Returns:
        Dict with board, confidence, uncertain, timings, etc.
    """
    timings = {}

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ── Stage 1: Grid detection ──
    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=interactive)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    # Save corners for reuse
    if grid.method == "manual" and debug:
        save_corners(grid.corners, "corners.npy")

    if debug:
        debug_grid(image, grid, "debug_grid.jpg")

    cell_images = extract_cell_images(grid)

    # ── Stage 2: Tile presence detection ──
    t0 = time.perf_counter()
    tile_cells = []
    empty_cells = []

    for row, col, cell_img in cell_images:
        has_tile, _ = detect_tile_presence(cell_img, row, col)
        if has_tile:
            tile_cells.append((row, col, cell_img))
        else:
            empty_cells.append((row, col))

    timings["tiles_ms"] = (time.perf_counter() - t0) * 1000

    if debug:
        debug_tile_detection(cell_images, grid.board_image,
                              grid.cells, "debug_tiles.jpg")

    # ── Stage 3: Letter classification ──
    t0 = time.perf_counter()

    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    uncertain = []

    if tile_cells:
        model = load_model(model_path, device=device)
        tile_imgs = [img for _, _, img in tile_cells]
        predictions = predict_tiles(model, tile_imgs, device=device)

        for (row, col, _), (label, conf) in zip(tile_cells, predictions):
            board[row][col] = label
            confidence[row][col] = conf
            if conf < confidence_threshold:
                uncertain.append((row, col))

    timings["classify_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "board": board,
        "confidence": confidence,
        "uncertain": uncertain,
        "tile_count": len(tile_cells),
        "empty_count": len(empty_cells),
        "grid_method": grid.method,
        "timings": timings,
    }


def print_board(result: dict):
    """Pretty-print the scanned board."""
    board = result["board"]
    uncertain_set = set(map(tuple, result["uncertain"]))

    print(f"\n{'=' * 50}")
    print(f"Grid: {result['grid_method']} | "
          f"Tiles: {result['tile_count']} | "
          f"Empty: {result['empty_count']}")
    print(f"Time: {result['timings']['total_ms']:.0f}ms "
          f"(grid {result['timings']['grid_ms']:.0f} + "
          f"detect {result['timings']['tiles_ms']:.0f} + "
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
    """Extract words from the board (across and down)."""
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
    parser.add_argument("--interactive", action="store_true",
                        help="Click the 4 board corners manually")
    parser.add_argument("--corners", default=None,
                        help="Path to saved corners.npy file")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug_grid.jpg and debug_tiles.jpg")
    parser.add_argument("--words", action="store_true",
                        help="Extract and print detected words")
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