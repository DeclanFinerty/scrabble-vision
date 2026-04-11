"""
Batch evaluation of the scan pipeline against ground truth boards.

Usage:
    uv run evaluate.py
    uv run evaluate.py --image board_001
    uv run evaluate.py --no-interactive
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
from src.classification.model import (
    load_model, predict_tiles, classify_result
)

EVAL_DIR = Path("data/eval")
CORNERS_DIR = EVAL_DIR / "corners"
DEBUG_DIR = EVAL_DIR / "debug"
GT_DIR = EVAL_DIR / "ground_truth"


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
    timings = {}

    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=interactive)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    cell_images = extract_cell_images(grid)

    t0 = time.perf_counter()
    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    tile_count = 0

    if model is not None:
        all_imgs = [img for _, _, img in cell_images]
        predictions = predict_tiles(model, all_imgs, device="cpu")
        for (row, col, _), (label, conf) in zip(cell_images, predictions):
            mapped = classify_result(label)
            board[row][col] = mapped
            confidence[row][col] = conf
            if mapped != ".":
                tile_count += 1

    timings["classify_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "board": board,
        "confidence": confidence,
        "grid": grid,
        "cell_images": cell_images,
        "tile_count": tile_count,
        "empty_count": GRID_SIZE * GRID_SIZE - tile_count,
        "timings": timings,
    }


def compare_boards(predicted: list[list[str]], truth: list[list[str]]) -> dict:
    correct = 0
    wrong = []
    letters_correct = 0
    letters_total = 0
    empty_correct = 0
    empty_total = 0
    confusion = defaultdict(int)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            pred = predicted[r][c]
            true = truth[r][c]

            is_true_letter = true not in (".", "_")
            is_pred_letter = pred != "."

            if true in (".", "_"):
                empty_total += 1
                if pred == ".":
                    empty_correct += 1
                    correct += 1
                else:
                    wrong.append((r, c, true, pred))
                    confusion[("EMPTY", pred)] += 1
            else:
                letters_total += 1
                if true == pred:
                    letters_correct += 1
                    correct += 1
                elif pred == ".":
                    wrong.append((r, c, true, pred))
                    confusion[(true, "MISSED")] += 1
                else:
                    wrong.append((r, c, true, pred))
                    confusion[(true, pred)] += 1

    return {
        "correct": correct,
        "total": GRID_SIZE * GRID_SIZE,
        "wrong": wrong,
        "letters_correct": letters_correct,
        "letters_total": letters_total,
        "empty_correct": empty_correct,
        "empty_total": empty_total,
        "confusion": dict(confusion),
    }


def print_comparison(result: dict, predicted: list[list[str]],
                     truth: list[list[str]], confidence: list[list[float]]):
    acc = result["correct"] / result["total"]
    print(f"  Cell accuracy: {result['correct']}/{result['total']} ({acc:.1%})")

    lc, lt = result["letters_correct"], result["letters_total"]
    if lt > 0:
        print(f"  Letter accuracy: {lc}/{lt} ({lc / lt:.1%})")

    ec, et = result["empty_correct"], result["empty_total"]
    if et > 0:
        print(f"  Empty accuracy:  {ec}/{et} ({ec / et:.1%})")

    if result["wrong"]:
        print(f"\n  Wrong cells ({len(result['wrong'])}):")
        for row, col, true, pred in result["wrong"]:
            conf = confidence[row][col]
            print(f"    ({row:2d},{col:2d}): expected '{true}' got '{pred}' ({conf:.0%})")


def print_side_by_side(predicted: list[list[str]], truth: list[list[str]]):
    print("\n  Expected:                          Got:")
    header = "    " + " ".join(f"{i:2d}" for i in range(GRID_SIZE))
    print(header + "    " + header)
    for r in range(GRID_SIZE):
        true_row = f" {r:2d} " + "  ".join(truth[r])
        pred_row = f" {r:2d} " + "  ".join(predicted[r])
        markers = ""
        for c in range(GRID_SIZE):
            if predicted[r][c] != truth[r][c]:
                if not (truth[r][c] == "." and predicted[r][c] == "."):
                    markers += f" ({c})"
        suffix = f"  <- err{markers}" if markers else ""
        print(f"  {true_row}    {pred_row}{suffix}")


def label_corners_mode():
    """Interactive corner labeling for saved boards."""
    import json
    from datetime import datetime, timezone

    saved_dir = Path("saved_boards")
    if not saved_dir.is_dir():
        print("No saved_boards/ directory found.")
        return

    boards = []
    for d in sorted(saved_dir.iterdir()):
        meta_path = d / "board.json"
        img_path = d / "image.jpg"
        if meta_path.is_file() and img_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            boards.append((d, meta, img_path))

    total = len(boards)
    unlabeled = [(d, m, p) for d, m, p in boards if m.get("ground_truth_corners") is None]
    already_labeled = total - len(unlabeled)
    labeled_this_session = 0

    print(f"Found {total} saved boards, {len(unlabeled)} unlabeled, {already_labeled} already labeled.")
    if not unlabeled:
        print("Nothing to label.")
        return

    for idx, (board_dir, meta, img_path) in enumerate(unlabeled):
        board_id = meta.get("id", board_dir.name)
        print(f"\n[{idx + 1}/{len(unlabeled)}] Board: {meta.get('name', board_id)}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not read {img_path}, skipping.")
            continue

        clicks = []
        labels = ["TL", "TR", "BR", "BL"]

        def draw_overlay():
            vis = image.copy()
            h, w = vis.shape[:2]
            # Draw auto-detected corners (gray)
            auto = meta.get("auto_detected_corners")
            if auto:
                pts = np.array(auto, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (150, 150, 150), 2)
                for pt in pts:
                    cv2.circle(vis, tuple(pt), 6, (150, 150, 150), -1)
            # Draw final corners (blue)
            final = meta.get("final_corners")
            if final:
                pts = np.array(final, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 150, 0), 2)
                for pt in pts:
                    cv2.circle(vis, tuple(pt), 6, (255, 150, 0), -1)
            # Draw clicked ground truth points (green)
            for i, (x, y) in enumerate(clicks):
                cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(vis, labels[i], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(vis, clicks[i - 1], (x, y), (0, 255, 0), 2)
            if len(clicks) == 4:
                cv2.line(vis, clicks[3], clicks[0], (0, 255, 0), 2)
            # Instructions
            if len(clicks) < 4:
                msg = f"Click {labels[len(clicks)]} corner ({len(clicks)}/4)"
            else:
                msg = "Press Enter/y=save, r=reset, s=skip, q=quit"
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "gray=auto  blue=final  green=ground truth",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return vis

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y))
                cv2.imshow("Label Corners", draw_overlay())

        win_name = "Label Corners"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, on_click)
        cv2.imshow(win_name, draw_overlay())

        quit_all = False
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                quit_all = True
                break
            elif key == ord('s'):
                print("  Skipped.")
                break
            elif key == ord('r'):
                clicks.clear()
                cv2.imshow(win_name, draw_overlay())
            elif (key == 13 or key == ord('y')) and len(clicks) == 4:
                # Save ground truth corners
                meta["ground_truth_corners"] = clicks
                meta["ground_truth_set_at"] = datetime.now(timezone.utc).isoformat()
                with open(board_dir / "board.json", "w") as f:
                    json.dump(meta, f)
                labeled_this_session += 1
                print(f"  Saved ground truth corners: {clicks}")
                break

        cv2.destroyWindow(win_name)
        if quit_all:
            print("  Quitting.")
            break

    still_unlabeled = len(unlabeled) - labeled_this_session
    print(f"\nDone. Labeled {labeled_this_session} this session, "
          f"{still_unlabeled} still unlabeled, {total} total.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scan pipeline")
    parser.add_argument("--image", default=None,
                        help="Single image stem (e.g. board_001)")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--model", default="models/tile_classifier.pt")
    parser.add_argument("--label-corners", action="store_true",
                        help="Interactive corner labeling for saved boards")
    args = parser.parse_args()

    if args.label_corners:
        label_corners_mode()
        return

    CORNERS_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    if args.image:
        patterns = [EVAL_DIR / f"{args.image}.jpeg",
                    EVAL_DIR / f"{args.image}.jpg",
                    EVAL_DIR / f"{args.image}.png"]
        images = [p for p in patterns if p.exists()]
        if not images:
            print(f"Image not found: {args.image}")
            return
    else:
        images = sorted(
            p for p in EVAL_DIR.iterdir()
            if p.suffix.lower() in (".jpeg", ".jpg", ".png")
        )

    if not images:
        print("No images found in", EVAL_DIR)
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

        result = scan_board_image(image, corners=corners,
                                  interactive=interactive, model=model)

        if interactive and result["grid"].corners is not None:
            save_corners(result["grid"].corners, str(corners_path))

        debug_grid(image, result["grid"],
                   str(DEBUG_DIR / f"{stem}_grid.jpg"))

        t = result["timings"]
        total_ms = t["grid_ms"] + t["classify_ms"]
        print(f"  Tiles: {result['tile_count']} detected, "
              f"{result['empty_count']} empty")
        print(f"  Time: {total_ms:.0f}ms total "
              f"(grid {t['grid_ms']:.0f} + classify {t['classify_ms']:.0f}ms)")

        gt_path = GT_DIR / f"{stem}.txt"
        truth = load_ground_truth(gt_path)
        if truth is not None:
            comparison = compare_boards(result["board"], truth)
            comparison["timings"] = result["timings"]
            comparison["tile_count"] = result["tile_count"]
            print_comparison(comparison, result["board"], truth,
                             result["confidence"])
            print_side_by_side(result["board"], truth)
            all_results.append((stem, comparison))
        else:
            print(f"  No ground truth at {gt_path}")
            print("\n  Predicted board:")
            for r in range(GRID_SIZE):
                print(f"  {r:2d} " + "  ".join(result["board"][r]))

    # Summary
    if not all_results:
        return

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total_correct = sum(r["correct"] for _, r in all_results)
    total_cells = sum(r["total"] for _, r in all_results)
    total_wrong = sum(len(r["wrong"]) for _, r in all_results)

    print(f"Boards evaluated: {len(all_results)}")
    print(f"Cell accuracy: {total_correct}/{total_cells} "
          f"({total_correct / total_cells:.1%})")
    print(f"Total errors: {total_wrong}")

    sum_lc = sum(r["letters_correct"] for _, r in all_results)
    sum_lt = sum(r["letters_total"] for _, r in all_results)
    if sum_lt > 0:
        print(f"Letter accuracy: {sum_lc}/{sum_lt} ({sum_lc / sum_lt:.1%})")

    sum_ec = sum(r["empty_correct"] for _, r in all_results)
    sum_et = sum(r["empty_total"] for _, r in all_results)
    if sum_et > 0:
        print(f"Empty accuracy:  {sum_ec}/{sum_et} ({sum_ec / sum_et:.1%})")

    # Timing
    n = len(all_results)
    total_tiles = sum(r.get("tile_count", 0) for _, r in all_results)
    grid_ms = sum(r["timings"]["grid_ms"] for _, r in all_results)
    classify_ms = sum(r["timings"]["classify_ms"] for _, r in all_results)
    total_ms = grid_ms + classify_ms

    print(f"\nTiming ({n} boards):")
    print(f"  Total:    {total_ms:7.0f}ms  ({total_ms / n:5.0f}ms/board)")
    print(f"  Grid:     {grid_ms:7.0f}ms  ({grid_ms / n:5.0f}ms/board)")
    print(f"  Classify: {classify_ms:7.0f}ms  ({classify_ms / n:5.0f}ms/board)")

    # Confused pairs
    all_confusion = defaultdict(int)
    for _, r in all_results:
        for pair, count in r["confusion"].items():
            all_confusion[pair] += count

    if all_confusion:
        sorted_conf = sorted(all_confusion.items(),
                             key=lambda x: x[1], reverse=True)
        print(f"\nMost confused pairs:")
        for (true, pred), count in sorted_conf[:15]:
            print(f"  {true} -> {pred}: {count}")


if __name__ == "__main__":
    main()
