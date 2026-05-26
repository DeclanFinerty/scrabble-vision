"""
Train the dense corner regressor: predicts all 256 intersections of
the 15x15 Scrabble grid (a 16x16 lattice) from a 224x224 letterboxed
image. MobileNetV2 backbone + fully-convolutional regression head.

Dense targets are computed *analytically* from the existing 4 corner
labels via a perspective transform from the unit grid to image space
-- no additional labeling is required.

Reads:
  training/corner_dense/dense_config.yaml -- config (overridable with --config)
  data/splits/corner_val.txt              -- val filenames (no augmentation)
  data/splits/corner_train.txt            -- train filenames (used by augment)
  data/annotations/corners.json           -- 4 corners per board

Writes (auto-incrementing) training/corner_dense/runs/run_NNN/:
  config.yaml                 copy of the config that produced this run
  best_model.pt               state_dict at the lowest val-loss epoch
  metrics.json                best-epoch metrics + full per-epoch history
  loss_curve.png              total + base + grid-reg loss per epoch
  log.txt                     stdout duplicate
  augmented/corner_train/     per-run augmented JPEGs (gitignored;
                              deleted on success unless --keep-augmented)
  augmented/corner_manifest.csv  manifest for this run's training set

The augmentation step is shared with the 4-corner pipeline: it writes
a 4-corner manifest; this script re-derives the 256 dense targets per
sample at load time, so the augmenter does not need a dense-aware
variant.

The 256 points are ordered row-major (index = row * 16 + col), with
row 0 at the top of the board. The four board corners therefore land
at indices 0 (TL), 15 (TR), 255 (BR), 240 (BL); per-corner error at
these indices is reported every epoch for direct comparison against
`training/corner/runs/`.

Loss: Smooth L1 on all 512 normalized outputs +
  `grid_reg_weight` * (mean variance of consecutive-point distances
                       along each row + col)
+ `collinearity_weight` * (mean squared perpendicular distance from
                           each intermediate point to the line
                           through that row/col's two endpoints).
The spacing term encourages evenly-spaced points; the collinearity
term encourages each row/col to lie on a straight line. Together
they regularize the lattice where per-point supervision is weak.

Usage:
    uv run scripts/train_corner_dense.py
    uv run scripts/train_corner_dense.py --config path/to/exp.yaml
    uv run scripts/train_corner_dense.py --reuse-augmented training/corner/runs/run_NNN/augmented/corner_manifest.csv
    uv run scripts/train_corner_dense.py --keep-augmented
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision.models as tvm  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import corners_store  # noqa: E402
from src.image_io import load_board_image  # noqa: E402

from augment_corners import (  # noqa: E402
    DEFAULT_COUNT, DEFAULT_SEED, SPLIT_FILE as TRAIN_SPLIT,
    image_dir_for, load_aug_params, run_augment,
)
from train_corners import (  # noqa: E402
    BOARDS_DIR, CORNER_NAMES, IMAGENET_MEAN, IMAGENET_STD, TARGET_SIZE,
    TeeStdout, find_image, image_to_tensor, letterbox,
    letterbox_corners_to_norm, next_run_dir,
    unletterbox_norm_to_orig_px,
)


DEFAULT_CONFIG = Path("training/corner_dense/dense_config.yaml")
RUNS_DIR = Path("training/corner_dense/runs")
VAL_SPLIT = Path("data/splits/corner_val.txt")

GRID_SIDE = 16            # 15 board rows/cols => 16 intersection lines per axis
NUM_POINTS = GRID_SIDE * GRID_SIDE      # 256
NUM_OUTPUTS = NUM_POINTS * 2            # 512

# Row-major indices of the 4 board corners in the 256-point lattice.
# (TL, TR, BR, BL) matches CORNER_NAMES from train_corners.
CORNER_INDICES = (
    0,                                  # TL: row 0, col 0
    GRID_SIDE - 1,                      # TR: row 0, col 15  -> 15
    NUM_POINTS - 1,                     # BR: row 15, col 15 -> 255
    NUM_POINTS - GRID_SIDE,             # BL: row 15, col 0  -> 240
)


# --- Dense-target geometry ---------------------------------------

def compute_dense_targets(corners_4: np.ndarray) -> np.ndarray:
    """Compute the 256 grid intersection points from 4 corner points.

    `corners_4`: (4, 2) array of (x, y) for TL, TR, BR, BL, in any
    consistent coordinate frame (normalized [0, 1] in letterboxed
    space here, but the math is frame-agnostic).

    Returns (256, 2) in the same frame, row-major (index = row*16 +
    col, row 0 at the top, col 0 at the left). The board corners land
    at the indices in `CORNER_INDICES`.

    The 4 corners define a perspective transform from a unit square
    to image space; we apply it to every (col/15, row/15) lattice
    point.
    """
    src = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    dst = np.asarray(corners_4, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    # Row-major lattice: outer loop over row, inner over col.
    # np.meshgrid(x, y) with default indexing='xy' yields arrays
    # whose .ravel() walks col-fastest, which is exactly what we want.
    axis = np.linspace(0.0, 1.0, GRID_SIDE, dtype=np.float64)
    cols, rows = np.meshgrid(axis, axis)            # both shape (16, 16)
    unit = np.stack([cols.ravel(), rows.ravel()], axis=1)  # (256, 2)

    homo = np.hstack([unit, np.ones((NUM_POINTS, 1), dtype=np.float64)])
    out = (M @ homo.T).T
    out = out[:, :2] / out[:, 2:3]
    return out.astype(np.float32)


# --- Model -------------------------------------------------------

class CornerDenseRegressor(nn.Module):
    """MobileNetV2 backbone + fully-convolutional regression head that
    predicts 512 normalized coordinates (256 grid intersections).

    Head structure:

        features (B, 1280, 7, 7)
          -> Conv2d(1280, 512, 1x1) -> ReLU
          -> Conv2d(512,  256, 1x1) -> ReLU
          -> AdaptiveAvgPool2d(pool_size)
          -> Flatten
          -> Dropout(dropout)
          -> Linear(256 * pool_size * pool_size, 512)

    The 1x1 convs do per-spatial-cell channel mixing on the native
    7x7 grid. `pool_size` controls how much spatial information the
    final FC layer sees:
      pool_size = 1 : pure GAP -> 256-dim summary (smallest head).
      pool_size = 7 : no spatial pooling -- the FC sees the full
                      7x7 grid (12,544 features).
    Dropout sits between Flatten and the final Linear, matching the
    4-corner head; it's a no-op at inference (model.eval()) so a
    weights file trained at one dropout rate loads cleanly at any
    other rate.
    """

    def __init__(self, backbone: str, unfreeze_layers: int,
                 pool_size: int, dropout: float):
        super().__init__()
        if backbone != "mobilenetv2":
            raise ValueError(
                f"Dense head is wired for mobilenetv2's 1280-channel "
                f"feature map; got {backbone!r}."
            )
        if pool_size < 1:
            raise ValueError(
                f"pool_size must be >= 1; got {pool_size}."
            )
        base = tvm.mobilenet_v2(
            weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        feat_dim = 1280

        # unfreeze_layers: -1 = all trainable, 0 = fully frozen,
        # N > 0 = unfreeze the last N children of self.features.
        n_blocks = len(self.features)
        if unfreeze_layers == -1:
            self._frozen_count = 0
        elif unfreeze_layers == 0:
            self._frozen_count = n_blocks
        else:
            self._frozen_count = max(0, n_blocks - unfreeze_layers)
        for i, block in enumerate(self.features):
            requires_grad = i >= self._frozen_count
            for p in block.parameters():
                p.requires_grad = requires_grad

        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * pool_size * pool_size, NUM_OUTPUTS),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def train(self, mode: bool = True):
        # Keep BN in eval mode on frozen blocks so their running
        # stats don't drift while only later layers are training.
        super().train(mode)
        if mode and self._frozen_count > 0:
            for i in range(self._frozen_count):
                self.features[i].eval()
        return self


def resolve_config(raw: dict) -> dict:
    """Normalise a raw config dict to current keys.

    Translates legacy `freeze_backbone` to `unfreeze_layers` and
    fills in `grid_reg_weight` defaulted to 0.1 so older configs
    saved before the dense pipeline still load."""
    config = dict(raw)
    if "unfreeze_layers" not in config:
        config["unfreeze_layers"] = (
            0 if config.get("freeze_backbone", False) else -1
        )
    config.setdefault("pool_size", 1)
    config.setdefault("dropout", 0.0)
    config.setdefault("grid_reg_weight", 0.1)
    config.setdefault("collinearity_weight", 0.1)
    return config


# --- Loss --------------------------------------------------------

def grid_regularity_penalty(pred_pts: torch.Tensor) -> torch.Tensor:
    """For each row (16 consecutive points) compute the variance of
    the 15 consecutive-point distances; same for each column. Return
    the mean of all 32 variances.

    A perfectly affine grid has zero penalty (all row distances are
    equal within a row, similarly per column). The penalty rises
    smoothly as the grid warps -- complementing the per-point Smooth
    L1, which by itself does not enforce inter-point consistency.

    `pred_pts`: (B, 256, 2), row-major (index = row*16 + col).
    """
    B = pred_pts.shape[0]
    pts = pred_pts.view(B, GRID_SIDE, GRID_SIDE, 2)

    # Row distances: 15 segments per row, 16 rows.
    row_diffs = pts[:, :, 1:, :] - pts[:, :, :-1, :]
    row_dists = torch.sqrt(
        (row_diffs * row_diffs).sum(dim=-1).clamp_min(1e-12)
    )  # (B, 16, 15)
    row_var = row_dists.var(dim=-1, unbiased=False)  # (B, 16)

    # Column distances: 15 segments per column, 16 columns.
    col_diffs = pts[:, 1:, :, :] - pts[:, :-1, :, :]
    col_dists = torch.sqrt(
        (col_diffs * col_diffs).sum(dim=-1).clamp_min(1e-12)
    )  # (B, 15, 16)
    col_var = col_dists.var(dim=-2, unbiased=False)  # (B, 16)

    return 0.5 * (row_var.mean() + col_var.mean())


def grid_collinearity_penalty(pred_pts: torch.Tensor) -> torch.Tensor:
    """For each row, fit the line through point[0] -> point[15], then
    take the squared perpendicular distance from each of the 14
    intermediate points to that line. Same per column. Return the
    mean of all such squared distances.

    A perfectly affine projection of the unit lattice puts every row
    on a single line (and same for every column), so the penalty is
    zero. Curvature in the predicted lattice (lens distortion left in
    by the head, or per-point noise) shows up here as nonzero terms,
    even when consecutive spacing happens to be uniform.

    `pred_pts`: (B, 256, 2), row-major (index = row*16 + col).
    """
    B = pred_pts.shape[0]
    pts = pred_pts.view(B, GRID_SIDE, GRID_SIDE, 2)
    last = GRID_SIDE - 1

    # --- Rows ---
    # Endpoints + intermediates per row. d is the row direction;
    # v_i = p_i - p_0 is the offset of each intermediate from p_0.
    # 2D cross |d x v_i| equals the parallelogram area = |d| * h,
    # so h^2 = (d x v_i)^2 / |d|^2.
    row_p0 = pts[:, :, 0, :]                            # (B, 16, 2)
    row_p1 = pts[:, :, last, :]                         # (B, 16, 2)
    row_d = row_p1 - row_p0                             # (B, 16, 2)
    row_d_len2 = (row_d * row_d).sum(dim=-1).clamp_min(1e-12)   # (B, 16)
    row_inter = pts[:, :, 1:last, :]                    # (B, 16, 14, 2)
    row_v = row_inter - row_p0.unsqueeze(2)             # (B, 16, 14, 2)
    row_cross = (row_d[..., 0:1] * row_v[..., 1]
                 - row_d[..., 1:2] * row_v[..., 0])     # (B, 16, 14)
    row_perp_sq = (row_cross * row_cross) / row_d_len2.unsqueeze(-1)

    # --- Columns ---
    col_p0 = pts[:, 0, :, :]                            # (B, 16, 2)
    col_p1 = pts[:, last, :, :]                         # (B, 16, 2)
    col_d = col_p1 - col_p0                             # (B, 16, 2)
    col_d_len2 = (col_d * col_d).sum(dim=-1).clamp_min(1e-12)   # (B, 16)
    col_inter = pts[:, 1:last, :, :]                    # (B, 14, 16, 2)
    col_v = col_inter - col_p0.unsqueeze(1)             # (B, 14, 16, 2)
    col_d_uns = col_d.unsqueeze(1)                      # (B, 1, 16, 2)
    col_cross = (col_d_uns[..., 0] * col_v[..., 1]
                 - col_d_uns[..., 1] * col_v[..., 0])   # (B, 14, 16)
    col_perp_sq = (col_cross * col_cross) / col_d_len2.unsqueeze(1)

    return 0.5 * (row_perp_sq.mean() + col_perp_sq.mean())


def dense_loss(pred: torch.Tensor, target: torch.Tensor,
               grid_reg_weight: float,
               collinearity_weight: float,
               base_criterion: nn.Module
               ) -> tuple[torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor]:
    """Combined Smooth L1 + grid-regularity + collinearity loss.

    `pred`, `target`: (B, 512) normalized in letterboxed space.
    Returns (total, base, reg, coll).

    Setting either weight to 0 disables that term's contribution to
    the gradient but leaves the unweighted penalty in the returned
    tuple, so the training loop can still log it as a diagnostic. Use
    weight=0 to keep a past experiment's config reproducible while
    turning off a term whose assumption no longer holds (e.g. drop
    grid_reg_weight to 0 once augmentation includes perspective
    warps, since a warped board's lattice has unequal row/col
    spacing by construction)."""
    base = base_criterion(pred, target)
    pred_pts = pred.view(pred.shape[0], NUM_POINTS, 2)
    reg = grid_regularity_penalty(pred_pts)
    coll = grid_collinearity_penalty(pred_pts)
    total = (base
             + float(grid_reg_weight) * reg
             + float(collinearity_weight) * coll)
    return total, base, reg, coll


# --- Data --------------------------------------------------------

class DenseCornerManifestDataset(torch.utils.data.Dataset):
    """Pre-augmented training rows from corner_manifest.csv.

    The manifest stores only the 4 augmented corner points (the
    augmenter is corner-only); we expand to 256 dense targets per
    item at __getitem__ time so the augmentation pipeline does not
    have to know about the dense head."""

    def __init__(self, manifest_path: Path):
        with manifest_path.open() as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = cv2.imread(r["image_path"])
        if img is None:
            raise RuntimeError(f"Could not read {r['image_path']}")
        corners_4 = np.array(
            [[r["tl_x"], r["tl_y"]],
             [r["tr_x"], r["tr_y"]],
             [r["br_x"], r["br_y"]],
             [r["bl_x"], r["bl_y"]]],
            dtype=np.float32,
        )
        dense = compute_dense_targets(corners_4)        # (256, 2)
        return (
            image_to_tensor(img),
            torch.from_numpy(dense.reshape(-1).astype(np.float32)),
        )


def load_val(val_split: Path, corner_entries: dict
             ) -> tuple[torch.Tensor, torch.Tensor,
                        list[tuple[int, int, float, int, int]], list[str]]:
    """Returns (imgs, dense_coords, meta, kept).

    `dense_coords`: (N, 512) normalized in letterboxed space.
    `meta[i]`: (orig_w, orig_h, scale, pad_x, pad_y) so predictions
    can be un-letterboxed back to original-image pixels.
    """
    names = [ln.strip() for ln in val_split.read_text().splitlines() if ln.strip()]
    imgs, coords, meta, kept = [], [], [], []
    for fname in names:
        path = find_image(fname)
        entry = corner_entries.get(fname)
        if path is None or entry is None or entry["corners"] is None:
            print(f"  skipping val {fname}: not found / no corners")
            continue
        img = load_board_image(path)
        if img is None:
            print(f"  skipping val {fname}: could not load")
            continue
        h, w = img.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(img, TARGET_SIZE)
        imgs.append(image_to_tensor(padded))

        pts = np.array(entry["corners"], dtype=np.float32)
        norm = letterbox_corners_to_norm(pts, scale, pad_x, pad_y, TARGET_SIZE)
        dense = compute_dense_targets(norm)             # (256, 2)
        coords.append(torch.from_numpy(
            dense.reshape(-1).astype(np.float32)))
        meta.append((w, h, scale, pad_x, pad_y))
        kept.append(fname)
    if not imgs:
        raise RuntimeError(f"No usable val images in {val_split}")
    return torch.stack(imgs), torch.stack(coords), meta, kept


# --- Evaluation --------------------------------------------------

def evaluate(model: nn.Module, val_imgs: torch.Tensor,
             val_coords: torch.Tensor,
             val_meta: list[tuple[int, int, float, int, int]],
             device: torch.device,
             criterion: nn.Module,
             grid_reg_weight: float,
             collinearity_weight: float) -> tuple[float, dict]:
    """Returns (val_loss_total, metrics).

    Metrics report:
      - per-corner pixel + %-of-diagonal error at the 4 board-corner
        indices (TL/TR/BR/BL) for direct comparison with the 4-corner
        regressor runs;
      - mean error across all 256 lattice points;
      - val base (Smooth L1), grid-spacing, and collinearity loss
        components, each reported unweighted.
    """
    model.eval()
    with torch.no_grad():
        preds = model(val_imgs.to(device))
        targets = val_coords.to(device)
        total, base, reg, coll = dense_loss(
            preds, targets, grid_reg_weight, collinearity_weight,
            criterion)
    preds_cpu = preds.cpu().numpy().reshape(-1, NUM_POINTS, 2)
    gt_cpu = val_coords.numpy().reshape(-1, NUM_POINTS, 2)

    per_image_all_px, per_image_all_pct = [], []
    per_image_corner_px, per_image_corner_pct = [], []
    corner_idx = np.array(CORNER_INDICES, dtype=np.int64)
    for p, g, (w, h, scale, pad_x, pad_y) in zip(preds_cpu, gt_cpu, val_meta):
        p_px = unletterbox_norm_to_orig_px(p.copy(), scale, pad_x, pad_y)
        g_px = unletterbox_norm_to_orig_px(g.copy(), scale, pad_x, pad_y)
        err = np.linalg.norm(p_px - g_px, axis=1)       # (256,)
        diag = float(np.sqrt(w * w + h * h))
        per_image_all_px.append(err)
        per_image_all_pct.append(err / diag)
        per_image_corner_px.append(err[corner_idx])
        per_image_corner_pct.append(err[corner_idx] / diag)

    mean_all_px = float(np.stack(per_image_all_px).mean())
    mean_all_pct = float(np.stack(per_image_all_pct).mean())
    per_corner_px = np.mean(np.stack(per_image_corner_px), axis=0)
    per_corner_pct = np.mean(np.stack(per_image_corner_pct), axis=0)

    return float(total.item()), {
        "mean_corner_error_px": float(per_corner_px.mean()),
        "mean_corner_error_pct": float(per_corner_pct.mean()),
        "mean_all_points_error_px": mean_all_px,
        "mean_all_points_error_pct": mean_all_pct,
        "per_corner_error_px": {n: float(per_corner_px[i])
                                for i, n in enumerate(CORNER_NAMES)},
        "per_corner_error_pct": {n: float(per_corner_pct[i])
                                 for i, n in enumerate(CORNER_NAMES)},
        "val_base_loss": float(base.item()),
        "val_reg_loss": float(reg.item()),
        "val_coll_loss": float(coll.item()),
    }


# --- Plot --------------------------------------------------------

def save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    epochs = list(range(1, len(history["train_loss"]) + 1))

    axes[0].plot(epochs, history["train_loss"], label="train (total)")
    axes[0].plot(epochs, history["val_loss"], label="val (total)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Total loss (Smooth L1 + grid_reg + collinearity)")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, history["val_base_loss"], label="val base (Smooth L1)")
    axes[1].plot(epochs, history["val_reg_loss"], label="val grid_reg")
    axes[1].plot(epochs, history["val_coll_loss"], label="val collinearity")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss (unweighted)")
    axes[1].set_title("Val loss components")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Training loop -----------------------------------------------

def train_one_run(config: dict, run_dir: Path,
                  train_manifest: Path) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}\n")

    grid_reg_weight = float(config["grid_reg_weight"])
    collinearity_weight = float(config["collinearity_weight"])

    print(f"Loading train manifest: {train_manifest}")
    train_ds = DenseCornerManifestDataset(train_manifest)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  train rows: {len(train_ds)}")

    print("Loading val set...")
    corner_entries = corners_store.load()
    val_imgs, val_coords, val_meta, val_names = load_val(VAL_SPLIT, corner_entries)
    print(f"  val images: {len(val_names)}")

    model = CornerDenseRegressor(
        backbone=config["backbone"],
        unfreeze_layers=int(config["unfreeze_layers"]),
        pool_size=int(config["pool_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['backbone']} dense head  "
          f"pool_size={config['pool_size']}  "
          f"dropout={config['dropout']}  "
          f"unfreeze_layers={config['unfreeze_layers']}  "
          f"grid_reg_weight={grid_reg_weight}  "
          f"collinearity_weight={collinearity_weight}  "
          f"(trainable {n_trainable:,} / total {n_total:,})\n")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=int(config["lr_patience"]),
    )
    criterion = nn.SmoothL1Loss()

    history = {
        "train_loss": [], "train_base_loss": [],
        "train_reg_loss": [], "train_coll_loss": [],
        "val_loss": [], "val_base_loss": [],
        "val_reg_loss": [], "val_coll_loss": [],
        "mean_corner_err_px": [], "mean_corner_err_pct": [],
        "mean_all_err_px": [], "mean_all_err_pct": [],
        "lr": [],
    }
    best_val = float("inf")
    best_epoch = 0
    best_metrics: dict = {}
    since_improve = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        t0 = time.time()
        running_total = running_base = running_reg = running_coll = 0.0
        n_batches = 0
        for imgs, targets in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            total, base, reg, coll = dense_loss(
                model(imgs), targets, grid_reg_weight,
                collinearity_weight, criterion)
            total.backward()
            optimizer.step()
            running_total += float(total.item())
            running_base += float(base.item())
            running_reg += float(reg.item())
            running_coll += float(coll.item())
            n_batches += 1
        train_loss = running_total / max(1, n_batches)
        train_base = running_base / max(1, n_batches)
        train_reg = running_reg / max(1, n_batches)
        train_coll = running_coll / max(1, n_batches)

        val_loss, val_metrics = evaluate(
            model, val_imgs, val_coords, val_meta,
            device, criterion, grid_reg_weight, collinearity_weight,
        )
        scheduler.step(val_loss)

        mean_err_px = val_metrics["mean_corner_error_px"]
        mean_err_pct = val_metrics["mean_corner_error_pct"]
        mean_all_px = val_metrics["mean_all_points_error_px"]
        mean_all_pct = val_metrics["mean_all_points_error_pct"]
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_base_loss"].append(train_base)
        history["train_reg_loss"].append(train_reg)
        history["train_coll_loss"].append(train_coll)
        history["val_loss"].append(val_loss)
        history["val_base_loss"].append(val_metrics["val_base_loss"])
        history["val_reg_loss"].append(val_metrics["val_reg_loss"])
        history["val_coll_loss"].append(val_metrics["val_coll_loss"])
        history["mean_corner_err_px"].append(mean_err_px)
        history["mean_corner_err_pct"].append(mean_err_pct)
        history["mean_all_err_px"].append(mean_all_px)
        history["mean_all_err_pct"].append(mean_all_pct)
        history["lr"].append(lr_now)

        print(f"epoch {epoch:3d}  "
              f"train {train_loss:.5f} "
              f"(base {train_base:.5f} reg {train_reg:.5f} coll {train_coll:.5f})  "
              f"val {val_loss:.5f}  "
              f"corners {mean_err_px:6.2f}px ({mean_err_pct * 100:5.2f}%)  "
              f"all {mean_all_px:6.2f}px ({mean_all_pct * 100:5.2f}%)  "
              f"lr {lr_now:.2e}  ({time.time() - t0:.1f}s)")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_loss,
                "train_base_loss": train_base,
                "train_reg_loss": train_reg,
                "train_coll_loss": train_coll,
                "val_loss": val_loss,
                **val_metrics,
            }
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= int(config["early_stop_patience"]):
                print(f"\nEarly stop: no val improvement for "
                      f"{config['early_stop_patience']} epochs.")
                break

    save_loss_curve(history, run_dir / "loss_curve.png")

    metrics = {
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "val_image_count": len(val_names),
        "val_images": val_names,
        "grid_reg_weight": grid_reg_weight,
        "collinearity_weight": collinearity_weight,
        **best_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n--- Summary ---")
    print(f"Best epoch:           {best_epoch} / {len(history['train_loss'])}")
    print(f"Best val loss:        {best_val:.5f}")
    print(f"  base (SmoothL1):    {best_metrics['val_base_loss']:.5f}")
    print(f"  grid_reg:           {best_metrics['val_reg_loss']:.5f}  "
          f"(weighted x{grid_reg_weight})")
    print(f"  collinearity:       {best_metrics['val_coll_loss']:.5f}  "
          f"(weighted x{collinearity_weight})")
    print(f"Train loss at best:   {best_metrics['train_loss']:.5f}")
    print(f"Mean corner error:    {best_metrics['mean_corner_error_px']:.2f} px  "
          f"({best_metrics['mean_corner_error_pct'] * 100:.2f}% of image diagonal)")
    for name in CORNER_NAMES:
        px = best_metrics["per_corner_error_px"][name]
        pct = best_metrics["per_corner_error_pct"][name]
        print(f"  {name}: {px:7.2f} px  ({pct * 100:5.2f}%)")
    print(f"Mean all-256 error:   "
          f"{best_metrics['mean_all_points_error_px']:.2f} px  "
          f"({best_metrics['mean_all_points_error_pct'] * 100:.2f}%)")
    print(f"Run dir:              {run_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"Path to training config (default: {DEFAULT_CONFIG}).")
    parser.add_argument("--reuse-augmented", type=Path, default=None,
                        metavar="MANIFEST",
                        help="Path to an existing corner_manifest.csv "
                             "to reuse (e.g. from a sibling 4-corner "
                             "run), instead of generating a fresh "
                             "augmented set inside the new run dir.")
    parser.add_argument("--keep-augmented", action="store_true",
                        help="Don't delete the augmented images on "
                             "successful training. Default behaviour is "
                             "to drop the per-run corner_train/ JPEGs "
                             "(keeping the manifest CSV) to save disk. "
                             "Ignored when --reuse-augmented is set.")
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    if not VAL_SPLIT.is_file():
        print(f"Val split not found: {VAL_SPLIT}")
        print("Run scripts/make_splits.py first.")
        sys.exit(1)
    if args.reuse_augmented is not None and not args.reuse_augmented.is_file():
        print(f"--reuse-augmented manifest not found: {args.reuse_augmented}")
        sys.exit(1)

    raw_config = yaml.safe_load(args.config.read_text()) or {}
    config = resolve_config(raw_config)

    run_dir = next_run_dir(RUNS_DIR)
    shutil.copy(args.config, run_dir / "config.yaml")

    tee = TeeStdout(run_dir / "log.txt")
    sys.stdout = tee
    try:
        print(f"Run dir: {run_dir}")
        print(f"Config source: {args.config}\n")

        owns_augmented = args.reuse_augmented is None
        if args.reuse_augmented is not None:
            train_manifest = args.reuse_augmented
            print(f"Reusing augmented set: {train_manifest}\n")
        else:
            aug_out = run_dir / "augmented"
            print(f"Augmenting into {aug_out}...")
            aug_params = load_aug_params(args.config)
            count = int(raw_config.get("aug_count", DEFAULT_COUNT))
            seed = int(raw_config.get("aug_seed", DEFAULT_SEED))
            for k, v in aug_params.items():
                print(f"  {k}: {v}")
            print(f"  aug_count: {count}")
            print(f"  aug_seed: {seed}\n")
            train_manifest = run_augment(aug_out, aug_params, count, seed,
                                         TRAIN_SPLIT)
            print()

        train_one_run(config, run_dir, train_manifest)

        if owns_augmented and not args.keep_augmented:
            img_dir = image_dir_for(run_dir / "augmented")
            if img_dir.is_dir():
                print(f"\nCleaning augmented images: {img_dir}")
                shutil.rmtree(img_dir)
                print("  (kept manifest CSV; pass --keep-augmented to "
                      "preserve images on the next run)")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
