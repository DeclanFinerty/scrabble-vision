"""
Train the corner regressor (MobileNetV2 or ResNet18 backbone + 8-value
regression head) on the augmented training set.

Reads:
  training/corner/corner_config.yaml   -- config (overridable with --config)
  data/augmented/corner_manifest.csv   -- pre-augmented training set
  data/splits/corner_val.txt           -- val filenames (no augmentation)
  data/annotations/corners.json        -- corners for val images

Writes (auto-incrementing) training/corner/runs/run_NNN/:
  config.yaml       copy of the config that produced this run
  best_model.pt     state_dict at the lowest val-loss epoch
  metrics.json      best-epoch metrics + full per-epoch history
  loss_curve.png    train + val Smooth L1 loss per epoch
  log.txt           stdout duplicate

Targets are 8 normalised coordinates in TL, TR, BR, BL order.
Val pixel error is reported at each image's original resolution.

Usage:
    uv run scripts/train_corners.py
    uv run scripts/train_corners.py --config training/corner/configs/exp_resnet18.yaml
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

from src import corners_store  # noqa: E402
from src.image_io import load_board_image  # noqa: E402


DEFAULT_CONFIG = Path("training/corner/corner_config.yaml")
RUNS_DIR = Path("training/corner/runs")
TRAIN_MANIFEST = Path("data/augmented/corner_manifest.csv")
VAL_SPLIT = Path("data/splits/corner_val.txt")
BOARDS_DIR = Path("data/raw/boards")

TARGET_SIZE = 224
NUM_OUTPUTS = 8
CORNER_NAMES = ("TL", "TR", "BR", "BL")

# Pretrained backbones expect ImageNet-normalised RGB input.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --- Model -------------------------------------------------------

class CornerRegressor(nn.Module):
    def __init__(self, backbone: str, dropout: float,
                 unfreeze_layers: int, pool_size: int):
        super().__init__()
        if backbone == "mobilenetv2":
            base = tvm.mobilenet_v2(
                weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = base.features
            feat_dim = 1280
        elif backbone == "resnet18":
            base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            # Drop avgpool + fc; keep the conv stack through layer4.
            self.features = nn.Sequential(*list(base.children())[:-2])
            feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

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

        # AdaptiveAvgPool((P,P)) keeps a coarse PxP spatial grid of
        # features so the FC head can place corners. GAP(1) (P=1)
        # collapses position entirely and makes corner regression
        # effectively impossible.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(),
            nn.Linear(feat_dim * pool_size * pool_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_OUTPUTS),
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
    defaults `pool_size` to 1 (the original GAP(1) head) so configs
    saved by older runs still load correctly."""
    config = dict(raw)
    if "unfreeze_layers" not in config:
        config["unfreeze_layers"] = (
            0 if config.get("freeze_backbone", False) else -1
        )
    config.setdefault("pool_size", 1)
    return config


# --- Data --------------------------------------------------------

def image_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def letterbox(image: np.ndarray,
              target_size: int = TARGET_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, then pad to a target_size square.
    Returns (padded_image, scale, pad_x, pad_y) so callers can map
    coordinates between original-image and letterboxed pixel space."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def letterbox_corners_to_norm(corners_orig_px: np.ndarray, scale: float,
                              pad_x: int, pad_y: int,
                              target_size: int = TARGET_SIZE) -> np.ndarray:
    """Original-image pixel coords -> [0,1] normalised in letterboxed space."""
    out = corners_orig_px * scale
    out[:, 0] += pad_x
    out[:, 1] += pad_y
    return out / target_size


def unletterbox_norm_to_orig_px(pred_norm: np.ndarray, scale: float,
                                pad_x: int, pad_y: int,
                                target_size: int = TARGET_SIZE) -> np.ndarray:
    """[0,1] normalised letterboxed prediction -> original-image pixel coords."""
    out = pred_norm * target_size
    out[:, 0] -= pad_x
    out[:, 1] -= pad_y
    return out / scale


class CornerManifestDataset(torch.utils.data.Dataset):
    """Pre-augmented training rows from corner_manifest.csv."""

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
        coords = np.array(
            [r["tl_x"], r["tl_y"], r["tr_x"], r["tr_y"],
             r["br_x"], r["br_y"], r["bl_x"], r["bl_y"]],
            dtype=np.float32,
        )
        return image_to_tensor(img), torch.from_numpy(coords)


def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    matches = [p for p in BOARDS_DIR.rglob(name) if p.is_file()]
    return matches[0] if matches else None


def load_val(val_split: Path, corner_entries: dict
             ) -> tuple[torch.Tensor, torch.Tensor,
                        list[tuple[int, int, float, int, int]], list[str]]:
    """Returns (imgs, coords, meta, kept) where meta[i] is
    (orig_w, orig_h, scale, pad_x, pad_y) for un-letterboxing
    predictions back to original-image pixels."""
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
        coords.append(torch.from_numpy(norm.flatten().astype(np.float32)))
        meta.append((w, h, scale, pad_x, pad_y))
        kept.append(fname)
    if not imgs:
        raise RuntimeError(f"No usable val images in {val_split}")
    return torch.stack(imgs), torch.stack(coords), meta, kept


# --- Run dir + logging -------------------------------------------

def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        try:
            existing.append(int(p.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    n = max(existing, default=0) + 1
    run_dir = base / f"run_{n:03d}"
    run_dir.mkdir()
    return run_dir


class TeeStdout:
    def __init__(self, log_path: Path):
        self.file = log_path.open("w", buffering=1)
        self.stdout = sys.stdout

    def write(self, s: str) -> int:
        self.stdout.write(s)
        return self.file.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return self.stdout.isatty()

    def close(self) -> None:
        self.file.close()


# --- Evaluation --------------------------------------------------

def evaluate(model: nn.Module, val_imgs: torch.Tensor,
             val_coords: torch.Tensor,
             val_meta: list[tuple[int, int, float, int, int]],
             device: torch.device, criterion: nn.Module
             ) -> tuple[float, float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        preds = model(val_imgs.to(device))
        loss = float(criterion(preds, val_coords.to(device)).item())
    preds_cpu = preds.cpu().numpy().reshape(-1, 4, 2)
    gt_cpu = val_coords.numpy().reshape(-1, 4, 2)
    per_corner_errs = []
    for p, g, (_w, _h, scale, pad_x, pad_y) in zip(preds_cpu, gt_cpu, val_meta):
        p_px = unletterbox_norm_to_orig_px(p.copy(), scale, pad_x, pad_y)
        g_px = unletterbox_norm_to_orig_px(g.copy(), scale, pad_x, pad_y)
        per_corner_errs.append(np.linalg.norm(p_px - g_px, axis=1))
    per_corner = np.mean(np.stack(per_corner_errs), axis=0)
    mean_err = float(per_corner.mean())
    by_name = {name: float(per_corner[i]) for i, name in enumerate(CORNER_NAMES)}
    return loss, mean_err, by_name


# --- Plot --------------------------------------------------------

def save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Smooth L1 loss")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Training loop -----------------------------------------------

def train_one_run(config: dict, run_dir: Path) -> dict:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}\n")

    print("Loading train manifest...")
    train_ds = CornerManifestDataset(TRAIN_MANIFEST)
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

    model = CornerRegressor(
        backbone=config["backbone"],
        dropout=float(config["dropout"]),
        unfreeze_layers=int(config["unfreeze_layers"]),
        pool_size=int(config["pool_size"]),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['backbone']}  "
          f"pool_size={config['pool_size']}  "
          f"unfreeze_layers={config['unfreeze_layers']}  "
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

    history = {"train_loss": [], "val_loss": [],
               "mean_corner_err_px": [], "lr": []}
    best_val = float("inf")
    best_epoch = 0
    best_metrics: dict = {}
    since_improve = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        for imgs, targets in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(1, n_batches)

        val_loss, mean_err, per_corner = evaluate(
            model, val_imgs, val_coords, val_meta, device, criterion
        )
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_corner_err_px"].append(mean_err)
        history["lr"].append(lr_now)

        print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}  "
              f"mean_err {mean_err:6.2f}px  lr {lr_now:.2e}  "
              f"({time.time() - t0:.1f}s)")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mean_corner_error_px": mean_err,
                "per_corner_error_px": per_corner,
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
        **best_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n--- Summary ---")
    print(f"Best epoch:           {best_epoch} / {len(history['train_loss'])}")
    print(f"Best val loss:        {best_val:.5f}")
    print(f"Train loss at best:   {best_metrics['train_loss']:.5f}")
    print(f"Mean corner error:    {best_metrics['mean_corner_error_px']:.2f} px "
          f"(at each image's original resolution)")
    for name, err in best_metrics["per_corner_error_px"].items():
        print(f"  {name}: {err:6.2f} px")
    print(f"Run dir:              {run_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"Path to training config (default: {DEFAULT_CONFIG}).")
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    if not TRAIN_MANIFEST.is_file():
        print(f"Train manifest not found: {TRAIN_MANIFEST}")
        print("Run scripts/augment_corners.py first.")
        sys.exit(1)
    if not VAL_SPLIT.is_file():
        print(f"Val split not found: {VAL_SPLIT}")
        print("Run scripts/make_splits.py first.")
        sys.exit(1)

    config = resolve_config(yaml.safe_load(args.config.read_text()))

    run_dir = next_run_dir(RUNS_DIR)
    shutil.copy(args.config, run_dir / "config.yaml")

    tee = TeeStdout(run_dir / "log.txt")
    sys.stdout = tee
    try:
        print(f"Run dir: {run_dir}")
        print(f"Config source: {args.config}\n")
        train_one_run(config, run_dir)
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
