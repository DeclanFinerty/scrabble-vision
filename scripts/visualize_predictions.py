"""
Visualize a trained corner model's predictions on the validation set.

Reads:
  <run_dir>/config.yaml      -- backbone + dropout used at train time
  <run_dir>/best_model.pt    -- model state_dict
  data/splits/corner_val.txt -- val filenames
  data/annotations/corners.json -- ground-truth corners

Writes (to <run_dir>/predictions/):
  <stem>.jpg     full-resolution image with predicted (red) and
                 ground-truth (green) corners drawn on top
  summary.jpg    grid of the same, one cell per val image

Per-image pixel error is printed to stdout. Use this after every
training run to eyeball which val images the model still misses.

Usage:
    uv run scripts/visualize_predictions.py training/corner/runs/run_001
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
    CORNER_NAMES, CornerRegressor, TARGET_SIZE,
    find_image, image_to_tensor,
    letterbox, letterbox_corners_to_norm,
    resolve_config, unletterbox_norm_to_orig_px,
)


VAL_SPLIT = Path("data/splits/corner_val.txt")

GT_COLOR = (0, 255, 0)      # green (BGR)
PRED_COLOR = (0, 0, 255)    # red

SUMMARY_COLS = 4
SUMMARY_CELL = 400
SUMMARY_GAP = 6
SUMMARY_BG = (40, 40, 40)


def line_thickness(image_dim: int) -> int:
    return max(2, image_dim // 600)


def dot_radius(image_dim: int) -> int:
    return max(8, image_dim // 200)


def draw_quad(image: np.ndarray, corners_px: np.ndarray,
              color: tuple[int, int, int]) -> None:
    """Draw 4 dots + the connecting quadrilateral in place on `image`."""
    s = max(image.shape[:2])
    thick = line_thickness(s)
    r = dot_radius(s)
    pts = [(int(round(x)), int(round(y))) for x, y in corners_px]
    for i in range(4):
        cv2.line(image, pts[i], pts[(i + 1) % 4], color, thick,
                 lineType=cv2.LINE_AA)
    for p in pts:
        cv2.circle(image, p, r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, p, r, (0, 0, 0), max(1, thick // 2),
                   lineType=cv2.LINE_AA)


def annotate(image: np.ndarray, gt_px: np.ndarray,
             pred_px: np.ndarray) -> np.ndarray:
    """Return a copy with GT drawn first, predicted on top."""
    out = image.copy()
    draw_quad(out, gt_px, GT_COLOR)
    draw_quad(out, pred_px, PRED_COLOR)
    return out


def fit_with_corners(image: np.ndarray, gt_px: np.ndarray, pred_px: np.ndarray,
                     cell: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letterbox `image` into a `cell`x`cell` square and apply the same
    transform to the corner points so annotations stay aligned."""
    h, w = image.shape[:2]
    scale = cell / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell, cell, 3), SUMMARY_BG, dtype=np.uint8)
    y0 = (cell - new_h) // 2
    x0 = (cell - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    offset = np.array([x0, y0], dtype=np.float32)
    return canvas, gt_px * scale + offset, pred_px * scale + offset


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


def print_trace(fname: str, w: int, h: int, scale: float,
                pad_x: int, pad_y: int, pred_norm: np.ndarray,
                gt_px: np.ndarray) -> None:
    """Dump every intermediate value in the inference pipeline so
    you can verify the letterbox + denormalize math by hand."""
    print(f"\n=== TRACE: {fname} ===")
    print(f"  original image:     {w} x {h}")
    print(f"  letterbox params:   scale={scale:.6f}  pad_x={pad_x}  pad_y={pad_y}")
    print(f"                      (resized = {int(round(w * scale))} x "
          f"{int(round(h * scale))})")

    gt_norm = letterbox_corners_to_norm(gt_px.copy(), scale, pad_x, pad_y)
    rt_px = unletterbox_norm_to_orig_px(gt_norm.copy(), scale, pad_x, pad_y)
    print(f"  GT round-trip max err: "
          f"{float(np.max(np.abs(rt_px - gt_px))):.6f} px (must be ~0)")

    print(f"\n  Model output (normalized in letterbox space):")
    for i, name in enumerate(CORNER_NAMES):
        d = pred_norm[i] - gt_norm[i]
        print(f"    {name}: ({pred_norm[i,0]:+.4f}, {pred_norm[i,1]:+.4f})   "
              f"GT: ({gt_norm[i,0]:+.4f}, {gt_norm[i,1]:+.4f})   "
              f"diff: ({d[0]:+.4f}, {d[1]:+.4f})")

    print(f"\n  Step 1: norm * {TARGET_SIZE} -> letterbox px")
    lb_px = pred_norm * TARGET_SIZE
    for i, name in enumerate(CORNER_NAMES):
        print(f"    {name}: ({lb_px[i,0]:8.2f}, {lb_px[i,1]:8.2f})")

    print(f"\n  Step 2: - pad ({pad_x},{pad_y}) -> resized px")
    rs_px = lb_px.copy()
    rs_px[:, 0] -= pad_x
    rs_px[:, 1] -= pad_y
    for i, name in enumerate(CORNER_NAMES):
        print(f"    {name}: ({rs_px[i,0]:8.2f}, {rs_px[i,1]:8.2f})")

    print(f"\n  Step 3: / scale ({scale:.6f}) -> original px")
    orig_px = rs_px / scale
    for i, name in enumerate(CORNER_NAMES):
        e = float(np.linalg.norm(orig_px[i] - gt_px[i]))
        print(f"    {name}: ({orig_px[i,0]:8.2f}, {orig_px[i,1]:8.2f})   "
              f"GT: ({gt_px[i,0]:8.2f}, {gt_px[i,1]:8.2f})   err: {e:7.2f} px")

    mean_norm = float(np.mean(np.abs(pred_norm.flatten() - gt_norm.flatten())))
    print(f"\n  mean |norm error| per coord: {mean_norm:.4f}")
    print(f"  -> in 224-px space:           {mean_norm * TARGET_SIZE:.2f} px")
    print(f"  -> in original-px space:      {mean_norm * TARGET_SIZE / scale:.2f} px")
    print(f"=== end trace ===\n")


def load_model(run_dir: Path, device: torch.device) -> CornerRegressor:
    config = resolve_config(yaml.safe_load((run_dir / "config.yaml").read_text()))
    model = CornerRegressor(
        backbone=config["backbone"],
        dropout=float(config["dropout"]),
        unfreeze_layers=int(config["unfreeze_layers"]),
        pool_size=int(config["pool_size"]),
    ).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir", type=Path,
                        help="Training run directory "
                             "(e.g. training/corner/runs/run_001).")
    parser.add_argument("--trace", action="store_true",
                        help="Print the full letterbox/inference/denormalize "
                             "math for the first val image, to verify nothing "
                             "is wrong end-to-end.")
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

    header = (f"{'image':<32} {'size':<12} {'mean':>8} {'%diag':>7}  "
              f"{'TL':>7} {'TR':>7} {'BR':>7} {'BL':>7}")
    print(header)
    print("-" * len(header))

    cells: list[np.ndarray] = []
    mean_errs: list[float] = []
    mean_errs_pct: list[float] = []
    per_corner_acc = np.zeros(4, dtype=np.float64)
    per_corner_acc_pct = np.zeros(4, dtype=np.float64)
    trace_emitted = False

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
            pred_norm = model(x).cpu().numpy().reshape(4, 2)

        gt_px = np.array(entry["corners"], dtype=np.float32)

        if args.trace and not trace_emitted:
            trace_emitted = True
            print_trace(fname, w, h, scale, pad_x, pad_y, pred_norm, gt_px)

        pred_px = unletterbox_norm_to_orig_px(pred_norm, scale, pad_x, pad_y)

        per_corner = np.linalg.norm(pred_px - gt_px, axis=1)
        mean = float(per_corner.mean())
        # Diagonal-normalised error: a 50 px miss on a 4032 px image
        # and a 50 px miss on a 1300 px image are dramatically
        # different errors -- this metric makes them comparable.
        diag = float(np.sqrt(w * w + h * h))
        per_corner_pct = per_corner / diag
        mean_pct = float(per_corner_pct.mean())
        mean_errs.append(mean)
        mean_errs_pct.append(mean_pct)
        per_corner_acc += per_corner
        per_corner_acc_pct += per_corner_pct

        size_str = f"{w}x{h}"
        print(f"{fname:<32} {size_str:<12} {mean:7.2f} {mean_pct * 100:6.2f}%  "
              f"{per_corner[0]:6.2f} {per_corner[1]:6.2f} "
              f"{per_corner[2]:6.2f} {per_corner[3]:6.2f}")

        annotated = annotate(image, gt_px, pred_px)
        cv2.imwrite(str(out_dir / f"{Path(fname).stem}.jpg"),
                    annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        cell_img, gt_cell, pred_cell = fit_with_corners(
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
          f"{'':<12} {np.mean(mean_errs):7.2f} "
          f"{np.mean(mean_errs_pct) * 100:6.2f}%  "
          f"{per_corner_mean[0]:6.2f} {per_corner_mean[1]:6.2f} "
          f"{per_corner_mean[2]:6.2f} {per_corner_mean[3]:6.2f}")
    print(f"{'per-corner (% of diagonal)':<32} "
          f"{'':<12} {'':>8} {'':>7}  "
          f"{per_corner_mean_pct[0] * 100:6.2f} "
          f"{per_corner_mean_pct[1] * 100:6.2f} "
          f"{per_corner_mean_pct[2] * 100:6.2f} "
          f"{per_corner_mean_pct[3] * 100:6.2f}")
    print(f"\nWrote {len(cells)} annotated images + summary to {out_dir}")


if __name__ == "__main__":
    main()
