"""
Run a trained dense corner model on the held-out test set and save
the annotated 256-point predictions to a fresh, auto-numbered folder
inside the run dir.

Reads:
  <run_dir>/config.yaml         -- backbone + unfreeze used at train time
  <run_dir>/best_model.pt       -- model state_dict
  data/splits/corner_test.txt   -- test filenames
  data/annotations/corners.json -- 4 corners per board (dense GT is
                                    derived analytically via the same
                                    compute_dense_targets used at train)

Writes (auto-incrementing, never overwrites):
  <run_dir>/predictions_test_NN/
    NNN_<stem>.jpg   each test image with predicted (red) +
                     ground-truth (green) 256-point lattices drawn on
                     top; the 4 board corners (indices 0, 15, 240, 255)
                     are drawn larger for easy comparison with the
                     4-corner regressor.
    summary.jpg      grid of letterboxed thumbnails, one per test image.
    metrics.json     per-image errors + dataset-level aggregates,
                     reporting both corner-only and all-256 stats.

Usage:
    uv run scripts/predict_test_dense.py training/corner_dense/runs/run_001
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
    letterbox, letterbox_corners_to_norm,
    unletterbox_norm_to_orig_px,
)
from train_corner_dense import (  # noqa: E402
    CORNER_INDICES, NUM_POINTS, compute_dense_targets,
)
from visualize_predictions_dense import (  # noqa: E402
    SUMMARY_CELL, annotate, fit_with_points, load_model, make_summary,
)


TEST_SPLIT = Path("data/splits/corner_test.txt")


def next_predictions_dir(run_dir: Path) -> Path:
    """Return <run_dir>/predictions_test_MM where MM is the next free
    2-digit slot. Created on the way out so callers can rely on it
    existing."""
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

    header = (f"{'#':>3}  {'image':<32} {'size':<12} "
              f"{'corner_px':>10} {'corner_%':>9}  "
              f"{'all_px':>8} {'all_%':>7}  "
              f"{'TL':>7} {'TR':>7} {'BR':>7} {'BL':>7}")
    print(header)
    print("-" * len(header))

    cells: list[np.ndarray] = []
    per_image: list[dict] = []
    corner_errs_px: list[float] = []
    corner_errs_pct: list[float] = []
    all_errs_px: list[float] = []
    all_errs_pct: list[float] = []
    per_corner_acc = np.zeros(4, dtype=np.float64)
    per_corner_acc_pct = np.zeros(4, dtype=np.float64)
    corner_idx = np.array(CORNER_INDICES, dtype=np.int64)

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
            pred_norm = model(x).cpu().numpy().reshape(NUM_POINTS, 2)

        gt_px_corners = np.array(entry["corners"], dtype=np.float32)
        gt_norm_corners = letterbox_corners_to_norm(
            gt_px_corners.copy(), scale, pad_x, pad_y, TARGET_SIZE)
        gt_norm = compute_dense_targets(gt_norm_corners)            # (256, 2)

        pred_px = unletterbox_norm_to_orig_px(
            pred_norm.copy(), scale, pad_x, pad_y)                  # (256, 2)
        gt_px = unletterbox_norm_to_orig_px(
            gt_norm.copy(), scale, pad_x, pad_y)                    # (256, 2)

        err = np.linalg.norm(pred_px - gt_px, axis=1)               # (256,)
        diag = float(np.sqrt(w * w + h * h))
        per_corner = err[corner_idx]
        per_corner_pct = per_corner / diag
        corner_mean = float(per_corner.mean())
        corner_mean_pct = float(per_corner_pct.mean())
        all_mean = float(err.mean())
        all_mean_pct = all_mean / diag

        corner_errs_px.append(corner_mean)
        corner_errs_pct.append(corner_mean_pct)
        all_errs_px.append(all_mean)
        all_errs_pct.append(all_mean_pct)
        per_corner_acc += per_corner
        per_corner_acc_pct += per_corner_pct

        idx += 1
        size_str = f"{w}x{h}"
        print(f"{idx:3d}  {fname:<32} {size_str:<12} "
              f"{corner_mean:9.2f} {corner_mean_pct * 100:8.2f}%  "
              f"{all_mean:7.2f} {all_mean_pct * 100:6.2f}%  "
              f"{per_corner[0]:6.2f} {per_corner[1]:6.2f} "
              f"{per_corner[2]:6.2f} {per_corner[3]:6.2f}")

        annotated = annotate(image, gt_px, pred_px)
        out_name = f"{idx:03d}_{Path(fname).stem}.jpg"
        cv2.imwrite(str(out_dir / out_name),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        cell_img, gt_cell, pred_cell = fit_with_points(
            image, gt_px, pred_px, SUMMARY_CELL)
        cells.append(annotate(cell_img, gt_cell, pred_cell))

        per_image.append({
            "index": idx,
            "filename": fname,
            "saved_as": out_name,
            "width": w,
            "height": h,
            "mean_corner_error_px": corner_mean,
            "mean_corner_error_pct": corner_mean_pct,
            "mean_all_points_error_px": all_mean,
            "mean_all_points_error_pct": all_mean_pct,
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
          f"{'':<12} "
          f"{np.mean(corner_errs_px):9.2f} {np.mean(corner_errs_pct) * 100:8.2f}%  "
          f"{np.mean(all_errs_px):7.2f} {np.mean(all_errs_pct) * 100:6.2f}%  "
          f"{per_corner_mean[0]:6.2f} {per_corner_mean[1]:6.2f} "
          f"{per_corner_mean[2]:6.2f} {per_corner_mean[3]:6.2f}")
    print(f"     {'per-corner (% of diagonal)':<32} "
          f"{'':<12} {'':>10} {'':>9}  {'':>8} {'':>7}  "
          f"{per_corner_mean_pct[0] * 100:6.2f} "
          f"{per_corner_mean_pct[1] * 100:6.2f} "
          f"{per_corner_mean_pct[2] * 100:6.2f} "
          f"{per_corner_mean_pct[3] * 100:6.2f}")

    metrics = {
        "run_dir": str(run_dir),
        "test_split": str(TEST_SPLIT),
        "image_count": len(cells),
        "mean_corner_error_px": float(np.mean(corner_errs_px)),
        "mean_corner_error_pct": float(np.mean(corner_errs_pct)),
        "mean_all_points_error_px": float(np.mean(all_errs_px)),
        "mean_all_points_error_pct": float(np.mean(all_errs_pct)),
        "per_corner_error_px": {n: float(per_corner_mean[i])
                                for i, n in enumerate(CORNER_NAMES)},
        "per_corner_error_pct": {n: float(per_corner_mean_pct[i])
                                 for i, n in enumerate(CORNER_NAMES)},
        "corner_indices": list(CORNER_INDICES),
        "per_image": per_image,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFor reference: {', '.join(CORNER_NAMES)} are at "
          f"indices {', '.join(str(i) for i in CORNER_INDICES)} in the "
          f"256-point lattice.")
    print(f"Wrote {len(cells)} annotated images + summary + metrics.json "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
