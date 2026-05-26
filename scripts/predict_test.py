"""
Run a trained corner model on the held-out test set and save the
annotated predictions to a fresh, auto-numbered folder inside the
run dir.

Reads:
  <run_dir>/config.yaml       -- backbone + dropout used at train time
  <run_dir>/best_model.pt     -- model state_dict
  data/splits/corner_test.txt -- test filenames
  data/annotations/corners.json -- ground-truth corners

Writes (auto-incrementing, never overwrites):
  <run_dir>/predictions_test_NN/
    NNN_<stem>.jpg   each test image with predicted (red) +
                     ground-truth (green) corners drawn on top.
                     NNN starts at 001 in test-split order.
    summary.jpg      grid of the same.
    metrics.json     per-image errors + dataset-level aggregates.

Usage:
    uv run scripts/predict_test.py training/corner/runs/run_034
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import corners_store  # noqa: E402
from src.image_io import load_board_image  # noqa: E402

from train_corners import (  # noqa: E402
    CORNER_NAMES, TARGET_SIZE,
    find_image, image_to_tensor,
    letterbox, unletterbox_norm_to_orig_px,
)
from visualize_predictions import (  # noqa: E402
    SUMMARY_CELL, annotate, fit_with_corners, load_model, make_summary,
)


TEST_SPLIT = Path("data/splits/corner_test.txt")


def next_predictions_dir(run_dir: Path) -> Path:
    """Return runs/run_NNN/predictions_test_MM where MM is the next
    free 2-digit slot. Created on the way out so callers can rely on
    it existing."""
    existing = []
    for p in run_dir.iterdir() if run_dir.is_dir() else []:
        if not p.is_dir():
            continue
        if not p.name.startswith("predictions_test_"):
            continue
        try:
            existing.append(int(p.name.split("_")[-1]))
        except ValueError:
            continue
    n = max(existing, default=0) + 1
    out = run_dir / f"predictions_test_{n:02d}"
    out.mkdir()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path,
                        help="Training run directory containing "
                             "best_model.pt + config.yaml.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not (run_dir / "config.yaml").is_file():
        print(f"Missing config.yaml in {run_dir}")
        sys.exit(1)
    if not (run_dir / "best_model.pt").is_file():
        print(f"Missing best_model.pt in {run_dir}")
        sys.exit(1)
    if not TEST_SPLIT.is_file():
        print(f"Missing test split: {TEST_SPLIT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(run_dir, device)
    print(f"Loaded model from {run_dir}")

    out_dir = next_predictions_dir(run_dir)
    print(f"Output dir: {out_dir}\n")

    test_names = [ln.strip() for ln in TEST_SPLIT.read_text().splitlines()
                  if ln.strip()]
    corner_entries = corners_store.load()

    header = (f"{'#':>3}  {'image':<32} {'size':<12} {'mean':>8} {'%diag':>7}  "
              f"{'TL':>7} {'TR':>7} {'BR':>7} {'BL':>7}")
    print(header)
    print("-" * len(header))

    cells: list[np.ndarray] = []
    per_image: list[dict] = []
    mean_errs: list[float] = []
    mean_errs_pct: list[float] = []
    per_corner_acc = np.zeros(4, dtype=np.float64)
    per_corner_acc_pct = np.zeros(4, dtype=np.float64)

    idx = 0
    for fname in test_names:
        path = find_image(fname)
        entry = corner_entries.get(fname)
        if path is None or entry is None or entry["corners"] is None:
            print(f"  skipping {fname}: not found / no corners")
            continue
        image = load_board_image(path)
        if image is None:
            print(f"  skipping {fname}: could not load")
            continue
        h, w = image.shape[:2]

        padded, scale, pad_x, pad_y = letterbox(image, TARGET_SIZE)
        x = image_to_tensor(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x).cpu().numpy().reshape(4, 2)
        pred_px = unletterbox_norm_to_orig_px(pred_norm, scale, pad_x, pad_y)
        gt_px = np.array(entry["corners"], dtype=np.float32)

        per_corner = np.linalg.norm(pred_px - gt_px, axis=1)
        mean = float(per_corner.mean())
        diag = float(np.sqrt(w * w + h * h))
        per_corner_pct = per_corner / diag
        mean_pct = float(per_corner_pct.mean())
        mean_errs.append(mean)
        mean_errs_pct.append(mean_pct)
        per_corner_acc += per_corner
        per_corner_acc_pct += per_corner_pct

        idx += 1
        size_str = f"{w}x{h}"
        print(f"{idx:3d}  {fname:<32} {size_str:<12} "
              f"{mean:7.2f} {mean_pct * 100:6.2f}%  "
              f"{per_corner[0]:6.2f} {per_corner[1]:6.2f} "
              f"{per_corner[2]:6.2f} {per_corner[3]:6.2f}")

        annotated = annotate(image, gt_px, pred_px)
        out_name = f"{idx:03d}_{Path(fname).stem}.jpg"
        cv2.imwrite(str(out_dir / out_name),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        cell_img, gt_cell, pred_cell = fit_with_corners(
            image, gt_px, pred_px, SUMMARY_CELL)
        cells.append(annotate(cell_img, gt_cell, pred_cell))

        per_image.append({
            "index": idx,
            "filename": fname,
            "saved_as": out_name,
            "width": w,
            "height": h,
            "mean_error_px": mean,
            "mean_error_pct": mean_pct,
            "per_corner_error_px": {n: float(per_corner[i])
                                    for i, n in enumerate(CORNER_NAMES)},
            "per_corner_error_pct": {n: float(per_corner_pct[i])
                                     for i, n in enumerate(CORNER_NAMES)},
        })

    if not cells:
        print("\nNo test images processed.")
        sys.exit(1)

    summary = make_summary(cells)
    cv2.imwrite(str(out_dir / "summary.jpg"),
                summary, [cv2.IMWRITE_JPEG_QUALITY, 92])

    per_corner_mean = per_corner_acc / len(cells)
    per_corner_mean_pct = per_corner_acc_pct / len(cells)
    print("-" * len(header))
    print(f"     {'mean over ' + str(len(cells)) + ' images':<32} "
          f"{'':<12} {np.mean(mean_errs):7.2f} "
          f"{np.mean(mean_errs_pct) * 100:6.2f}%  "
          f"{per_corner_mean[0]:6.2f} {per_corner_mean[1]:6.2f} "
          f"{per_corner_mean[2]:6.2f} {per_corner_mean[3]:6.2f}")
    print(f"     {'per-corner (% of diagonal)':<32} "
          f"{'':<12} {'':>8} {'':>7}  "
          f"{per_corner_mean_pct[0] * 100:6.2f} "
          f"{per_corner_mean_pct[1] * 100:6.2f} "
          f"{per_corner_mean_pct[2] * 100:6.2f} "
          f"{per_corner_mean_pct[3] * 100:6.2f}")

    metrics = {
        "run_dir": str(run_dir),
        "test_split": str(TEST_SPLIT),
        "image_count": len(cells),
        "mean_corner_error_px": float(np.mean(mean_errs)),
        "mean_corner_error_pct": float(np.mean(mean_errs_pct)),
        "per_corner_error_px": {n: float(per_corner_mean[i])
                                for i, n in enumerate(CORNER_NAMES)},
        "per_corner_error_pct": {n: float(per_corner_mean_pct[i])
                                 for i, n in enumerate(CORNER_NAMES)},
        "per_image": per_image,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {len(cells)} annotated images + summary + metrics.json "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
