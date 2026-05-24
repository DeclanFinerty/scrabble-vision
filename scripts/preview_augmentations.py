"""
Show 20 random augmentations of a board image with corner overlays.

Reads:
  data/raw/boards/<filename>      -- source image (resolved by name)
  data/annotations/corners.json   -- ground-truth corners

For each augmentation, draws the 4 corner points (TL=red, TR=green,
BR=blue, BL=yellow) connected by a white quadrilateral, on top of
the 224x224 augmented image. Arranges all 20 into a 4-col x 5-row
grid in a single OpenCV window (or saves to disk with --save).

Use this to eyeball the augmentation math: if every dot lands on the
matching physical board corner across all 20 samples, the coord
transforms are correct. Dots can sit outside the 224 cell when a
crop excluded that corner -- those won't be drawn but the connecting
line will still extend toward the edge.

Usage:
    uv run scripts/preview_augmentations.py IMG_5082.HEIC
    uv run scripts/preview_augmentations.py IMG_5082 --seed 42
    uv run scripts/preview_augmentations.py IMG_5082.HEIC --save preview.png
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import corners_store  # noqa: E402
from src.image_io import BOARD_IMAGE_EXTS, load_board_image  # noqa: E402

from augment_corners import (  # noqa: E402
    MAX_SOURCE_DIM, TARGET_SIZE,
    apply_h, augment_one, downscale_matrix,
)


BOARDS_DIR = Path("data/raw/boards")
WINDOW_NAME = "Augmentations preview"
GRID_COLS = 4
GRID_ROWS = 5
N_SAMPLES = GRID_COLS * GRID_ROWS
GRID_GAP = 4
GRID_BG = (40, 40, 40)

# BGR triples (cv2 colour order)
COLOR_TL = (0, 0, 255)      # red
COLOR_TR = (0, 255, 0)      # green
COLOR_BR = (255, 0, 0)      # blue
COLOR_BL = (0, 255, 255)    # yellow
LINE_COLOR = (255, 255, 255)
CORNER_COLORS = [COLOR_TL, COLOR_TR, COLOR_BR, COLOR_BL]
DOT_RADIUS = 6


def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    matches = [p for p in BOARDS_DIR.rglob(name) if p.is_file()]
    if matches:
        return matches[0]
    for ext in BOARD_IMAGE_EXTS:
        matches = [p for p in BOARDS_DIR.rglob(f"{name}{ext}") if p.is_file()]
        if matches:
            return matches[0]
    return None


def draw_corners(image: np.ndarray, corners_px: np.ndarray) -> np.ndarray:
    out = image.copy()
    # Lines first so the dots sit on top of any edge intersections.
    for i in range(4):
        p1 = (int(round(corners_px[i, 0])), int(round(corners_px[i, 1])))
        p2 = (int(round(corners_px[(i + 1) % 4, 0])),
              int(round(corners_px[(i + 1) % 4, 1])))
        cv2.line(out, p1, p2, LINE_COLOR, 1, lineType=cv2.LINE_AA)
    for pt, color in zip(corners_px, CORNER_COLORS):
        center = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out, center, DOT_RADIUS, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, center, DOT_RADIUS, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def make_grid(cells: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = cells[0].shape[:2]
    grid_w = GRID_COLS * cell_w + (GRID_COLS + 1) * GRID_GAP
    grid_h = GRID_ROWS * cell_h + (GRID_ROWS + 1) * GRID_GAP
    canvas = np.full((grid_h, grid_w, 3), GRID_BG, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        y = GRID_GAP + r * (cell_h + GRID_GAP)
        x = GRID_GAP + c * (cell_w + GRID_GAP)
        canvas[y:y + cell_h, x:x + cell_w] = cell
    return canvas


def wait_for_close(window_name: str) -> None:
    """Block until the user presses a key or closes the window."""
    while True:
        try:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return
        if visible < 1:
            return
        if cv2.waitKey(50) != -1:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("image",
                        help="Filename in data/raw/boards/ "
                             "(with or without extension).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: non-deterministic).")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save grid to this path instead of displaying.")
    args = parser.parse_args()

    path = find_image(args.image)
    if path is None:
        print(f"Image not found in {BOARDS_DIR}: {args.image}")
        sys.exit(1)

    entry = corners_store.load().get(path.name)
    if entry is None or entry["corners"] is None:
        print(f"No corners in corners.json for {path.name}")
        sys.exit(1)
    if not entry["corners_verified"]:
        print(f"Warning: {path.name} has corners but is not verified.")

    image = load_board_image(path)
    if image is None:
        print(f"Could not load image: {path}")
        sys.exit(1)

    # Apply the same downscale step augment_corners uses, so the
    # preview reflects what training actually sees.
    h0, w0 = image.shape[:2]
    M_down, (w, h) = downscale_matrix(w0, h0, MAX_SOURCE_DIM)
    if (w, h) != (w0, h0):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    corners_src = apply_h(M_down, np.array(entry["corners"], dtype=np.float64))

    rng = np.random.default_rng(args.seed)
    cells = []
    for _ in range(N_SAMPLES):
        aug_img, aug_corners = augment_one(image, corners_src, rng)
        cells.append(draw_corners(aug_img, aug_corners))

    grid = make_grid(cells)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(args.save), grid)
        if not ok:
            print(f"Failed to write {args.save}")
            sys.exit(1)
        print(f"Wrote {args.save}  ({grid.shape[1]}x{grid.shape[0]})")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, grid.shape[1], grid.shape[0])
    cv2.imshow(WINDOW_NAME, grid)
    print(f"Showing {N_SAMPLES} augmentations for {path.name}. "
          f"Press any key or close the window to exit.")
    wait_for_close(WINDOW_NAME)
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
