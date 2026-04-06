"""
Batch evaluation of the scan pipeline against ground truth boards.

Scans every image in data/raw_boards/, compares against ground truth
files, and reports per-cell accuracy with detailed error breakdown.

Usage:
    uv run evaluate.py                    # all boards, interactive if no saved corners
    uv run evaluate.py --image board_001  # single board
    uv run evaluate.py --no-interactive   # skip boards without saved corners
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from collections import defaultdict

from src.detection.grid_detect import (
    detect_grid, extract_cell_images, debug_grid,
    save_corners, load_corners, GRID_SIZE
)
from src.detection.tile_detect import detect_tile_presence, debug_tile_detection
from src.classification.model import load_model, predict_tiles

BOARDS_DIR = Path("data/raw_boards")
CORNERS_DIR = BOARDS_DIR / "corners"
DEBUG_DIR = BOARDS_DIR / "debug"
GT_DIR = BOARDS_DIR / "ground_truth"


def load_ground_truth(path: Path) -> list[list[str]] | None:
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if len(lines) != GRID_SIZE:
        print(f"  WARNING: {path} has {len(lines)} lines, expected {GRID_SIZE}")
        return None
    board = []
    for line in lines:
        if len(line) != GRID_SIZE:
            print(f"  WARNING: line '{line}' has {len(line)} chars, expected {GRID_SIZE}")
            return None
        board.append(list(line))
    return board


def scan_board_image(image: np.ndarray, corners: np.ndarray = None,
                     interactive: bool = False, model=None) -> dict:
    """Run the full pipeline on an image, returning scan result."""
    timings = {}

    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=interactive)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    cell_images = extract_cell_images(grid)

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

    t0 = time.perf_counter()
    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    if tile_cells and model is not None:
        tile_imgs = [img for _, _, img in tile_cells]
        predictions = predict_tiles(model, tile_imgs, device="cpu")
        for (row, col, _), (label, conf) in zip(tile_cells, predictions):
            if label == "EMPTY":
                board[row][col] = "?"
            elif label == "BLANK":
                board[row][col] = "_"
            else:
                board[row][col] = label
            confidence[row][col] = conf
    timings["classify_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "board": board,
        "confidence": confidence,
        "grid": grid,
        "cell_images": cell_images,
        "tile_count": len(tile_cells),
        "empty_count": len(empty_cells),
        "timings": timings,
    }


def compare_boards(predicted: list[list[str]], truth: list[list[str]]) -> dict:
    """Compare predicted board against ground truth."""
    correct = 0
    wrong = []
    tile_expected = 0
    tile_detected = 0
    tile_missed = 0
    empty_as_tile = 0
    confusion = defaultdict(int)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            pred = predicted[r][c]
            true = truth[r][c]
            is_true_tile = true not in (".", "_")
            is_pred_tile = pred not in (".", "?")

            if true == ".":
                tile_expected += 0
                if is_pred_tile:
                    empty_as_tile += 1
            else:
                tile_expected += 1

            if is_pred_tile:
                tile_detected += 1

            if true == "." and pred == ".":
                correct += 1
            elif true == "_" and pred == "_":
                correct += 1
            elif true == pred:
                correct += 1
            elif true == "." and pred in (".", "?"):
                correct += 1
            else:
                wrong.append((r, c, true, pred))
                if is_true_tile and is_pred_tile:
                    confusion[(true, pred)] += 1
                elif is_true_tile and not is_pred_tile:
                    tile_missed += 1
                    confusion[(true, "MISSED")] += 1

    return {
        "correct": correct,
        "total": GRID_SIZE * GRID_SIZE,
        "wrong": wrong,
        "tile_expected": tile_expected,
        "tile_detected": tile_detected,
        "tile_missed": tile_missed,
        "empty_as_tile": empty_as_tile,
        "confusion": dict(confusion),
    }


def print_comparison(result: dict, predicted: list[list[str]],
                     truth: list[list[str]], confidence: list[list[float]]):
    """Print detailed comparison between predicted and ground truth."""
    acc = result["correct"] / result["total"]
    print(f"  Accuracy: {result['correct']}/{result['total']} ({acc:.1%})")
    print(f"  Tiles: {result['tile_detected']} detected / {result['tile_expected']} expected")
    if result["tile_missed"]:
        print(f"  Missed tiles: {result['tile_missed']}")
    if result["empty_as_tile"]:
        print(f"  False positives (empty→tile): {result['empty_as_tile']}")

    if result["wrong"]:
        print(f"\n  Wrong cells ({len(result['wrong'])}):")
        for r, c, true, pred in result["wrong"]:
            conf = confidence[r][c]
            print(f"    ({r:2d},{c:2d}): expected '{true}' got '{pred}' ({conf:.0%})")


def print_side_by_side(predicted: list[list[str]], truth: list[list[str]]):
    """Print predicted vs truth boards side by side."""
    print("\n  Expected:                          Got:")
    header = "    " + " ".join(f"{i:2d}" for i in range(GRID_SIZE))
    print(header + "    " + header)
    for r in range(GRID_SIZE):
        true_row = f" {r:2d} " + "  ".join(truth[r])
        pred_row = f" {r:2d} " + "  ".join(predicted[r])
        markers = ""
        for c in range(GRID_SIZE):
            if predicted[r][c] != truth[r][c]:
                if not (truth[r][c] == "." and predicted[r][c] in (".", "?")):
                    markers += f" ({c})"
        suffix = f"  ← err{markers}" if markers else ""
        print(f"  {true_row}    {pred_row}{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scan pipeline")
    parser.add_argument("--image", default=None,
                        help="Single image stem (e.g. board_001)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip boards without saved corners")
    parser.add_argument("--model", default="models/tile_classifier.pt")
    args = parser.parse_args()

    CORNERS_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    # Find board images
    if args.image:
        patterns = [BOARDS_DIR / f"{args.image}.jpeg",
                    BOARDS_DIR / f"{args.image}.jpg",
                    BOARDS_DIR / f"{args.image}.png"]
        images = [p for p in patterns if p.exists()]
        if not images:
            print(f"Image not found: {args.image}")
            return
    else:
        images = sorted(
            p for p in BOARDS_DIR.iterdir()
            if p.suffix.lower() in (".jpeg", ".jpg", ".png")
        )

    if not images:
        print("No images found in", BOARDS_DIR)
        return

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device="cpu")

    all_results = []

    for img_path in images:
        stem = img_path.stem
        print(f"\n{'=' * 60}")
        print(f"Board: {stem}")
        print(f"{'=' * 60}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not read {img_path}")
            continue

        # Load or select corners
        corners_path = CORNERS_DIR / f"{stem}.npy"
        corners = None
        interactive = False

        if corners_path.exists():
            corners = load_corners(str(corners_path))
            print(f"  Using saved corners: {corners_path}")
        elif args.no_interactive:
            print(f"  No saved corners, skipping (--no-interactive)")
            continue
        else:
            interactive = True
            print(f"  No saved corners — opening interactive selector")

        # Run pipeline
        result = scan_board_image(image, corners=corners,
                                  interactive=interactive, model=model)

        # Save corners if we just selected them interactively
        if interactive and result["grid"].corners is not None:
            save_corners(result["grid"].corners, str(corners_path))

        # Save debug images
        debug_grid(image, result["grid"],
                   str(DEBUG_DIR / f"{stem}_grid.jpg"))
        debug_tile_detection(result["cell_images"], result["grid"].board_image,
                             result["grid"].cells,
                             str(DEBUG_DIR / f"{stem}_tiles.jpg"))

        t = result["timings"]
        print(f"  Tiles: {result['tile_count']} detected, "
              f"{result['empty_count']} empty")
        print(f"  Time: grid {t['grid_ms']:.0f}ms + "
              f"detect {t['tiles_ms']:.0f}ms + "
              f"classify {t['classify_ms']:.0f}ms")

        # Compare against ground truth
        gt_path = GT_DIR / f"{stem}.txt"
        truth = load_ground_truth(gt_path)
        if truth is not None:
            comparison = compare_boards(result["board"], truth)
            print_comparison(comparison, result["board"], truth,
                             result["confidence"])
            print_side_by_side(result["board"], truth)
            all_results.append((stem, comparison))
        else:
            print(f"  No ground truth at {gt_path}")
            # Print what we got
            print("\n  Predicted board:")
            for r in range(GRID_SIZE):
                print(f"  {r:2d} " + "  ".join(result["board"][r]))

    # Summary across all boards
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        total_correct = sum(r["correct"] for _, r in all_results)
        total_cells = sum(r["total"] for _, r in all_results)
        total_wrong = sum(len(r["wrong"]) for _, r in all_results)

        print(f"Boards evaluated: {len(all_results)}")
        print(f"Overall accuracy: {total_correct}/{total_cells} "
              f"({total_correct / total_cells:.1%})")
        print(f"Total errors: {total_wrong}")

        # Aggregate confusion pairs
        all_confusion = defaultdict(int)
        for _, r in all_results:
            for pair, count in r["confusion"].items():
                all_confusion[pair] += count

        if all_confusion:
            sorted_conf = sorted(all_confusion.items(),
                                 key=lambda x: x[1], reverse=True)
            print(f"\nMost confused pairs:")
            for (true, pred), count in sorted_conf[:15]:
                print(f"  {true} → {pred}: {count}")

    elif len(all_results) == 1:
        stem, r = all_results[0]
        if r["confusion"]:
            sorted_conf = sorted(r["confusion"].items(),
                                 key=lambda x: x[1], reverse=True)
            print(f"\nMost confused pairs:")
            for (true, pred), count in sorted_conf[:15]:
                print(f"  {true} → {pred}: {count}")


if __name__ == "__main__":
    main()
