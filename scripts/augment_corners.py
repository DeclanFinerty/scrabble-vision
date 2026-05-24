"""
Generate augmented training data for the corner regressor.

Reads:
  data/splits/corner_train.txt  -- training filenames (one per line)
  data/annotations/corners.json -- ground-truth corner coordinates

For each training image, writes:
  - The original resized to 224x224 (no spatial augmentation).
  - N augmented 224x224 JPEGs (configurable via --count, default 30).

Each augmented sample composes all four transforms in order, each
with random parameters:
  1. Rotation:    uniform 0-360 degrees, canvas expands to fit.
  2. Perspective: each rect corner perturbed up to 5% of image dim.
  3. Random crop: square crop containing >=60% of the board area
                  and >=2 corners in-frame, resized to 224x224.
  4. Color jitter: brightness/contrast/saturation/hue (image only).

The spatial transforms (1-3) are composed as a single 3x3 homography
applied once to both the image (via cv2.warpPerspective) and the 4
corner points. Each composed transform is round-trip verified --
T_inverse(T(corners)) must reproduce the source corners to floating-
point tolerance, or augmentation aborts.

Normalized corner coords in the manifest may fall outside [0, 1]
when the crop excludes one or two corners -- this is intentional.

Outputs:
  data/augmented/corner_train/<stem>_orig.jpg
  data/augmented/corner_train/<stem>_<NNN>.jpg
  data/augmented/corner_manifest.csv  (image_path + 8 coords + source)

Usage:
    uv run scripts/augment_corners.py
    uv run scripts/augment_corners.py --count 50 --seed 42
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.image_io import load_board_image  # noqa: E402


SPLIT_FILE = Path("data/splits/corner_train.txt")
OUT_IMG_DIR = Path("data/augmented/corner_train")
OUT_MANIFEST = Path("data/augmented/corner_manifest.csv")
BOARDS_DIR = Path("data/raw/boards")

TARGET_SIZE = 224
MAX_SOURCE_DIM = 1024
PERSPECTIVE_FRAC = 0.05
CROP_MIN_BOARD_FRAC = 0.60
CROP_MIN_CORNERS = 2
CROP_MAX_ATTEMPTS = 50
# Tolerance for the round-trip check, in pixels of the 224-pixel
# output. 1e-3 px is ~4e-6 normalized -- well under any meaningful
# label noise, while still catching genuine math bugs.
ROUND_TRIP_TOL = 1e-3
JPEG_QUALITY = 95

MANIFEST_HEADER = [
    "image_path",
    "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y",
    "source_image",
]


# --- File discovery -----------------------------------------------

def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    matches = [p for p in BOARDS_DIR.rglob(name) if p.is_file()]
    return matches[0] if matches else None


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        print(f"Split file not found: {path}")
        sys.exit(1)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# --- Geometry primitives ------------------------------------------

def apply_h(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography M to an (N, 2) array of points."""
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])
    out_h = (M @ pts_h.T).T
    return out_h[:, :2] / out_h[:, 2:3]


def polygon_area(pts: np.ndarray) -> float:
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def round_trip_check(M: np.ndarray, src_pts: np.ndarray,
                     transformed_pts: np.ndarray, tol: float) -> None:
    M_inv = np.linalg.inv(M)
    recovered = apply_h(M_inv, transformed_pts)
    err = float(np.max(np.abs(recovered - src_pts)))
    if err > tol:
        raise RuntimeError(
            f"Round-trip check failed: max error {err:.6g} px > {tol} px.\n"
            f"src:\n{src_pts}\nT(src):\n{transformed_pts}\n"
            f"T_inv(T(src)):\n{recovered}"
        )


# --- Transform matrices -------------------------------------------

def downscale_matrix(orig_w: int, orig_h: int,
                     max_dim: int) -> tuple[np.ndarray, tuple[int, int]]:
    scale = min(max_dim / max(orig_w, orig_h), 1.0)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    M = np.diag([scale, scale, 1.0])
    return M, (new_w, new_h)


def rotation_matrix(image_size: tuple[int, int],
                    angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Rotate around the image center and expand the canvas to fit."""
    w, h = image_size
    M2 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos_a = abs(M2[0, 0])
    sin_a = abs(M2[0, 1])
    new_w = int(np.ceil(w * cos_a + h * sin_a))
    new_h = int(np.ceil(w * sin_a + h * cos_a))
    M2[0, 2] += (new_w - w) / 2.0
    M2[1, 2] += (new_h - h) / 2.0
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = M2
    return M, (new_w, new_h)


def perspective_matrix(image_size: tuple[int, int], frac: float,
                       rng: np.random.Generator) -> np.ndarray:
    w, h = image_size
    pert = frac * min(w, h)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    dst = src + rng.uniform(-pert, pert, src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def crop_scale_matrix(crop_xywh: tuple[float, float, float, float],
                      out_size: int) -> np.ndarray:
    x0, y0, w, h = crop_xywh
    return np.array([
        [out_size / w, 0.0,          -x0 * out_size / w],
        [0.0,          out_size / h, -y0 * out_size / h],
        [0.0,          0.0,           1.0],
    ], dtype=np.float64)


# --- Random crop sampling -----------------------------------------

def board_area_in_crop(corners_local: np.ndarray,
                       crop_w: int, crop_h: int) -> int:
    """Area (px) of the board polygon inside the crop. fillPoly clips
    automatically to the mask, giving us the intersection."""
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    poly = np.round(corners_local).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    return int(np.count_nonzero(mask))


def count_corners_in_crop(corners: np.ndarray,
                          crop_xywh: tuple[float, float, float, float]) -> int:
    x0, y0, w, h = crop_xywh
    inside = ((corners[:, 0] >= x0) & (corners[:, 0] < x0 + w) &
              (corners[:, 1] >= y0) & (corners[:, 1] < y0 + h))
    return int(np.sum(inside))


def pick_crop(corners: np.ndarray, image_size: tuple[int, int],
              rng: np.random.Generator) -> tuple[float, float, float, float]:
    iw, ih = image_size
    short = float(min(iw, ih))
    board_area = polygon_area(corners)
    cx_b, cy_b = corners.mean(axis=0)

    extent = float(max(corners[:, 0].max() - corners[:, 0].min(),
                       corners[:, 1].max() - corners[:, 1].min()))
    side_min = float(min(short, max(extent * 0.7, short * 0.5)))
    side_max = short
    if side_min > side_max:
        side_min = side_max

    for _ in range(CROP_MAX_ATTEMPTS):
        side = float(rng.uniform(side_min, side_max))
        cx = cx_b + rng.uniform(-side * 0.25, side * 0.25)
        cy = cy_b + rng.uniform(-side * 0.25, side * 0.25)
        x0 = float(np.clip(cx - side / 2.0, 0.0, iw - side))
        y0 = float(np.clip(cy - side / 2.0, 0.0, ih - side))
        crop = (x0, y0, side, side)

        corners_local = corners - np.array([x0, y0])
        inside_px = board_area_in_crop(corners_local,
                                       int(round(side)), int(round(side)))
        frac = inside_px / board_area if board_area > 0 else 0.0
        if (frac >= CROP_MIN_BOARD_FRAC
                and count_corners_in_crop(corners, crop) >= CROP_MIN_CORNERS):
            return crop

    # Fallback: bounding box of the board, padded, clipped to image.
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    pad = 0.10 * max(x_max - x_min, y_max - y_min)
    side = float(min(max(x_max - x_min, y_max - y_min) + 2 * pad, short))
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x0 = float(np.clip(cx - side / 2.0, 0.0, iw - side))
    y0 = float(np.clip(cy - side / 2.0, 0.0, ih - side))
    return (x0, y0, side, side)


# --- Color jitter -------------------------------------------------

def color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bright = float(rng.uniform(0.7, 1.3))
    contrast = float(rng.uniform(0.7, 1.3))
    saturation = float(rng.uniform(0.7, 1.3))
    hue_deg = float(rng.uniform(-20.0, 20.0))

    img = image.astype(np.float32)
    img = img * bright
    img = (img - 128.0) * contrast + 128.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV hue is [0, 179]; one cv2-unit equals two real degrees.
    hsv[..., 0] = (hsv[..., 0] + hue_deg / 2.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# --- Pipeline -----------------------------------------------------

def augment_one(image: np.ndarray, corners_src: np.ndarray,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    angle = float(rng.uniform(0.0, 360.0))
    M_rot, (rw, rh) = rotation_matrix((w, h), angle)
    M_persp = perspective_matrix((rw, rh), PERSPECTIVE_FRAC, rng)

    M_pre = M_persp @ M_rot
    corners_pre = apply_h(M_pre, corners_src)

    crop = pick_crop(corners_pre, (rw, rh), rng)
    M_cs = crop_scale_matrix(crop, TARGET_SIZE)
    M_total = M_cs @ M_pre

    corners_final = apply_h(M_total, corners_src)
    round_trip_check(M_total, corners_src, corners_final, ROUND_TRIP_TOL)

    warped = cv2.warpPerspective(
        image, M_total, (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped = color_jitter(warped, rng)
    return warped, corners_final


def resize_original(image: np.ndarray,
                    corners_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    M = np.array([
        [TARGET_SIZE / w, 0.0,             0.0],
        [0.0,             TARGET_SIZE / h, 0.0],
        [0.0,             0.0,             1.0],
    ], dtype=np.float64)
    resized = cv2.warpPerspective(image, M, (TARGET_SIZE, TARGET_SIZE),
                                  flags=cv2.INTER_AREA)
    corners_final = apply_h(M, corners_src)
    round_trip_check(M, corners_src, corners_final, ROUND_TRIP_TOL)
    return resized, corners_final


def letterbox_matrix(orig_size: tuple[int, int],
                     out_size: int) -> tuple[np.ndarray, int, int]:
    """Aspect-preserving resize to (out_size, out_size) with black
    padding. Returns (3x3 homography, pad_x, pad_y) so callers can
    feed both image and corner points through the same matrix."""
    w, h = orig_size
    scale = out_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x = (out_size - new_w) // 2
    pad_y = (out_size - new_h) // 2
    M = np.array([
        [scale, 0.0,   float(pad_x)],
        [0.0,   scale, float(pad_y)],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    return M, pad_x, pad_y


def letterbox_original(image: np.ndarray, corners_src: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving 224x224 version of the source image. Used
    for color_only entries so train and val share the same letterbox
    pipeline and the model never sees a stretched board."""
    h, w = image.shape[:2]
    M, _, _ = letterbox_matrix((w, h), TARGET_SIZE)
    resized = cv2.warpPerspective(
        image, M, (TARGET_SIZE, TARGET_SIZE),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    corners_final = apply_h(M, corners_src)
    round_trip_check(M, corners_src, corners_final, ROUND_TRIP_TOL)
    return resized, corners_final


def color_only_one(image: np.ndarray, corners_src: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Color jitter only -- no rotation, perspective, or crop.
    Image is letterboxed to TARGET_SIZE so the corners stay anchored
    to the same physical board features and the aspect ratio matches
    the val/inference pipeline."""
    letterboxed, corners_final = letterbox_original(image, corners_src)
    return color_jitter(letterboxed, rng), corners_final


def manifest_row(path: Path, corners_224: np.ndarray, source: str) -> list:
    norm = corners_224 / TARGET_SIZE
    return [
        str(path),
        f"{norm[0, 0]:.6f}", f"{norm[0, 1]:.6f}",
        f"{norm[1, 0]:.6f}", f"{norm[1, 1]:.6f}",
        f"{norm[2, 0]:.6f}", f"{norm[2, 1]:.6f}",
        f"{norm[3, 0]:.6f}", f"{norm[3, 1]:.6f}",
        source,
    ]


# --- Main ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--count", type=int, default=30,
                        help="Augmented samples per training image "
                             "(default %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default %(default)s).")
    parser.add_argument("--split", type=Path, default=SPLIT_FILE,
                        help="Split manifest (default %(default)s).")
    args = parser.parse_args()

    train_names = read_split(args.split)
    corner_entries = corners_store.load()

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    rows: list[list] = []
    n_orig = n_aug = 0
    n_full_sources = n_color_sources = 0
    skipped: list[str] = []

    for i, fname in enumerate(train_names, start=1):
        path = find_image(fname)
        if path is None:
            skipped.append(f"{fname} (file not found)")
            continue
        entry = corner_entries.get(fname)
        if entry is None or entry["corners"] is None or not entry["corners_verified"]:
            skipped.append(f"{fname} (no verified corners)")
            continue
        image = load_board_image(path)
        if image is None:
            skipped.append(f"{fname} (could not load)")
            continue

        # Downscale early so intermediate buffers (esp. after rotation)
        # stay reasonable. Corner coords are pre-scaled by the same
        # matrix, so the round-trip check covers the full chain.
        h0, w0 = image.shape[:2]
        M_down, (w, h) = downscale_matrix(w0, h0, MAX_SOURCE_DIM)
        if (w, h) != (w0, h0):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        corners_src = apply_h(M_down,
                              np.array(entry["corners"], dtype=np.float64))

        stem = Path(fname).stem
        level = entry["augment_level"]

        if level == "color_only":
            orig_img, orig_corners = letterbox_original(image, corners_src)
            aug_fn = color_only_one
            n_color_sources += 1
        else:
            orig_img, orig_corners = resize_original(image, corners_src)
            aug_fn = augment_one
            n_full_sources += 1

        orig_path = OUT_IMG_DIR / f"{stem}_orig.jpg"
        cv2.imwrite(str(orig_path), orig_img,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        rows.append(manifest_row(orig_path, orig_corners, fname))
        n_orig += 1

        for k in range(args.count):
            aug_img, aug_corners = aug_fn(image, corners_src, rng)
            out_path = OUT_IMG_DIR / f"{stem}_{k:03d}.jpg"
            cv2.imwrite(str(out_path), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            rows.append(manifest_row(out_path, aug_corners, fname))
            n_aug += 1

        print(f"[{i}/{len(train_names)}] {fname}: 1 orig + {args.count} aug "
              f"({level})")

    with OUT_MANIFEST.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)} rows to {OUT_MANIFEST}")
    print(f"  Source images:    {n_full_sources + n_color_sources}  "
          f"(full={n_full_sources}, color_only={n_color_sources})")
    print(f"  Originals:        {n_orig}")
    print(f"  Augmented:        {n_aug}")
    if skipped:
        print(f"  Skipped:          {len(skipped)}")
        for s in skipped:
            print(f"    {s}")


if __name__ == "__main__":
    main()
