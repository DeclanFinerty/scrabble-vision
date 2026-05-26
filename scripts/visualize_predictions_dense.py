"""
Visualize a trained dense corner model's predictions on the validation
set: draw all 256 predicted grid intersections as small dots, with the
4 board corners (indices 0, 15, 240, 255) drawn larger, alongside
ground-truth points in green.

Reads:
  <run_dir>/config.yaml         -- backbone + unfreeze used at train time
  <run_dir>/best_model.pt       -- model state_dict
  data/splits/corner_val.txt    -- val filenames
  data/annotations/corners.json -- ground-truth corners (dense GT is
                                    derived analytically via the same
                                    compute_dense_targets used at train)

Writes (to <run_dir>/predictions/):
  <stem>.jpg     full-resolution image with all 256 predicted (red)
                 and ground-truth (green) points drawn; the 4 board
                 corners are drawn larger
  summary.jpg    grid of letterboxed thumbnails, one per val image

Per-image error is printed for both the 4 board corners (for direct
comparison with the 4-corner regressor) and the full 256-point set.

Usage:
    uv run scripts/visualize_predictions_dense.py training/corner_dense/runs/run_001
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

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
    CORNER_INDICES, CornerDenseRegressor, NUM_POINTS,
    compute_dense_targets, resolve_config,
)


VAL_SPLIT = Path("data/splits/corner_val.txt")

GT_COLOR = (0, 255, 0)        # green (BGR)
PRED_COLOR = (0, 0, 255)      # red

# Faint connecting polylines along each row / column so the predicted
# lattice is readable even when individual dots are tiny.
GT_LINE_COLOR = (0, 160, 0)
PRED_LINE_COLOR = (0, 0, 160)

SUMMARY_COLS = 4
SUMMARY_CELL = 400
SUMMARY_GAP = 6
SUMMARY_BG = (40, 40, 40)


def dot_radius(image_dim: int, is_corner: bool) -> int:
    """Inner-point dots stay small; the 4 board corners are larger so
    they're identifiable in the dense lattice."""
    if is_corner:
        return max(7, image_dim // 220)
    return max(2, image_dim // 800)


def line_thickness(image_dim: int) -> int:
    return max(1, image_dim // 1200)


def draw_dense(image: np.ndarray, pts: np.ndarray,
               color: tuple[int, int, int],
               line_color: tuple[int, int, int]) -> None:
    """Draw the full 256-point lattice in place.

    Faint polylines along each of the 16 rows + 16 columns make the
    grid structure visible; dots mark every intersection, with the 4
    board corners drawn larger.
    """
    s = max(image.shape[:2])
    thick = line_thickness(s)
    r_inner = dot_radius(s, is_corner=False)
    r_corner = dot_radius(s, is_corner=True)
    int_pts = np.round(pts).astype(np.int32).reshape(16, 16, 2)

    # cv2.polylines needs C-contiguous int32 arrays; column slices of
    # a (16, 16, 2) array are strided views, so copy before passing.
    for r in range(16):
        cv2.polylines(image, [np.ascontiguousarray(int_pts[r, :, :])],
                      False, line_color, thick, lineType=cv2.LINE_AA)
    for c in range(16):
        cv2.polylines(image, [np.ascontiguousarray(int_pts[:, c, :])],
                      False, line_color, thick, lineType=cv2.LINE_AA)

    corner_idx_set = set(CORNER_INDICES)
    for i, (x, y) in enumerate(pts):
        center = (int(round(x)), int(round(y)))
        is_corner = i in corner_idx_set
        cv2.circle(image, center,
                   r_corner if is_corner else r_inner,
                   color, -1, lineType=cv2.LINE_AA)
        if is_corner:
            cv2.circle(image, center, r_corner, (0, 0, 0),
                       max(1, thick), lineType=cv2.LINE_AA)


def annotate(image: np.ndarray, gt_pts: np.ndarray,
             pred_pts: np.ndarray) -> np.ndarray:
    """Return a copy with GT drawn first, predicted on top."""
    out = image.copy()
    draw_dense(out, gt_pts, GT_COLOR, GT_LINE_COLOR)
    draw_dense(out, pred_pts, PRED_COLOR, PRED_LINE_COLOR)
    return out


def fit_with_points(image: np.ndarray, gt_pts: np.ndarray, pred_pts: np.ndarray,
                    cell: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letterbox `image` into a `cell`x`cell` square and apply the same
    affine to the points so annotations stay aligned."""
    h, w = image.shape[:2]
    scale = cell / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell, cell, 3), SUMMARY_BG, dtype=np.uint8)
    y0 = (cell - new_h) // 2
    x0 = (cell - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    offset = np.array([x0, y0], dtype=np.float32)
    return canvas, gt_pts * scale + offset, pred_pts * scale + offset


def make_summary(cells: list[np.ndarray]) -> np.ndarray:
    n = len(cells)
    cols = SUMMARY_COLS
    rows = math.ceil(n / cols)
    grid_w = cols * SUMMARY_CELL + (cols + 1) * SUMMARY_GAP
    grid_h = rows * SUMMARY_CELL + (rows + 1) * SUMMARY_GAP
    canvas = np.full((grid_h, grid_w, 3), SUMMARY_BG, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r = idx // cols
        c = idx % cols
        y = SUMMARY_GAP + r * (SUMMARY_CELL + SUMMARY_GAP)
        x = SUMMARY_GAP + c * (SUMMARY_CELL + SUMMARY_GAP)
        canvas[y:y + SUMMARY_CELL, x:x + SUMMARY_CELL] = cell
    return canvas


def load_model(run_dir: Path, device: torch.device) -> CornerDenseRegressor:
    config = resolve_config(yaml.safe_load((run_dir / "config.yaml").read_text()))
    model = CornerDenseRegressor(
        backbone=config["backbone"],
        unfreeze_layers=int(config["unfreeze_layers"]),
        pool_size=int(config["pool_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path,
                        help="Training run directory "
                             "(e.g. training/corner_dense/runs/run_001).")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not (run_dir / "config.yaml").is_file():
        print(f"Missing config.yaml in {run_dir}")
        sys.exit(1)
    if not (run_dir / "best_model.pt").is_file():
        print(f"Missing best_model.pt in {run_dir}")
        sys.exit(1)
    if not VAL_SPLIT.is_file():
        print(f"Missing val split: {VAL_SPLIT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(run_dir, device)
    print(f"Loaded model from {run_dir}\n")

    val_names = [ln.strip() for ln in VAL_SPLIT.read_text().splitlines()
                 if ln.strip()]
    corner_entries = corners_store.load()

    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)

    header = (f"{'image':<32} {'size':<12} "
              f"{'corner_px':>10} {'corner_%':>9}  "
              f"{'all_px':>8} {'all_%':>7}  "
              f"{'TL':>7} {'TR':>7} {'BR':>7} {'BL':>7}")
    print(header)
    print("-" * len(header))

    cells: list[np.ndarray] = []
    corner_errs_px: list[float] = []
    corner_errs_pct: list[float] = []
    all_errs_px: list[float] = []
    all_errs_pct: list[float] = []
    per_corner_acc = np.zeros(4, dtype=np.float64)
    per_corner_acc_pct = np.zeros(4, dtype=np.float64)
    corner_idx = np.array(CORNER_INDICES, dtype=np.int64)

    for fname in val_names:
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
        gt_norm = compute_dense_targets(gt_norm_corners)         # (256, 2)

        pred_px = unletterbox_norm_to_orig_px(
            pred_norm.copy(), scale, pad_x, pad_y)               # (256, 2)
        gt_px = unletterbox_norm_to_orig_px(
            gt_norm.copy(), scale, pad_x, pad_y)                 # (256, 2)

        err = np.linalg.norm(pred_px - gt_px, axis=1)            # (256,)
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

        size_str = f"{w}x{h}"
        print(f"{fname:<32} {size_str:<12} "
              f"{corner_mean:9.2f} {corner_mean_pct * 100:8.2f}%  "
              f"{all_mean:7.2f} {all_mean_pct * 100:6.2f}%  "
              f"{per_corner[0]:6.2f} {per_corner[1]:6.2f} "
              f"{per_corner[2]:6.2f} {per_corner[3]:6.2f}")

        annotated = annotate(image, gt_px, pred_px)
        cv2.imwrite(str(out_dir / f"{Path(fname).stem}.jpg"),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        cell_img, gt_cell, pred_cell = fit_with_points(
            image, gt_px, pred_px, SUMMARY_CELL)
        cells.append(annotate(cell_img, gt_cell, pred_cell))

    if not cells:
        print("\nNo val images processed.")
        return

    summary = make_summary(cells)
    cv2.imwrite(str(out_dir / "summary.jpg"),
                summary, [cv2.IMWRITE_JPEG_QUALITY, 92])

    per_corner_mean = per_corner_acc / len(cells)
    per_corner_mean_pct = per_corner_acc_pct / len(cells)
    print("-" * len(header))
    print(f"{'mean over ' + str(len(cells)) + ' images':<32} "
          f"{'':<12} "
          f"{np.mean(corner_errs_px):9.2f} {np.mean(corner_errs_pct) * 100:8.2f}%  "
          f"{np.mean(all_errs_px):7.2f} {np.mean(all_errs_pct) * 100:6.2f}%  "
          f"{per_corner_mean[0]:6.2f} {per_corner_mean[1]:6.2f} "
          f"{per_corner_mean[2]:6.2f} {per_corner_mean[3]:6.2f}")
    print(f"{'per-corner (% of diagonal)':<32} "
          f"{'':<12} {'':>10} {'':>9}  {'':>8} {'':>7}  "
          f"{per_corner_mean_pct[0] * 100:6.2f} "
          f"{per_corner_mean_pct[1] * 100:6.2f} "
          f"{per_corner_mean_pct[2] * 100:6.2f} "
          f"{per_corner_mean_pct[3] * 100:6.2f}")
    print(f"\nFor reference: {', '.join(CORNER_NAMES)} are at "
          f"indices {', '.join(str(i) for i in CORNER_INDICES)} in the "
          f"256-point lattice.")
    print(f"Wrote {len(cells)} annotated images + summary to {out_dir}")


if __name__ == "__main__":
    main()
