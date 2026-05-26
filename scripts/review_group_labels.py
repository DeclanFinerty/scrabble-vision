"""
Render every image in one board group with its saved corner labels
drawn on top, arranged into a single contact-sheet image. Use this
to spot-check that the corners propagated correctly during
group-mode verification -- if one image's labels are rotated or
mirrored relative to the others, it'll jump out visually.

Reads:
  data/annotations/board_groups.json  -- group assignments
  data/annotations/corners.json       -- saved corner coords
  data/raw/boards/<filename>          -- source images (prefers the
                                         live file over original/
                                         backups, same as verify_corners)

Corner colors match preview_augmentations.py:
  TL=red, TR=green, BR=blue, BL=yellow, white connecting outline.

Cells without verified corners are still rendered, but tagged so
they don't get confused with a wrong rotation.

Usage:
    uv run scripts/review_group_labels.py --list
    uv run scripts/review_group_labels.py AG
    uv run scripts/review_group_labels.py AG --save out.png
    uv run scripts/review_group_labels.py AG --cols 5 --cell 320
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.image_io import BOARD_IMAGE_EXTS, load_board_image  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")
BOARD_GROUPS_FILE = Path("data/annotations/board_groups.json")

WINDOW_NAME = "Group label review"
DEFAULT_CELL = 320
DEFAULT_GAP = 6
LABEL_BAR = 24
BG_COLOR = (40, 40, 40)
MISSING_COLOR = (60, 60, 60)
TEXT_COLOR = (220, 220, 220)
TAG_COLOR_REJECTED = (40, 40, 180)
TAG_COLOR_NO_LABEL = (40, 90, 160)

# BGR triples
COLOR_TL = (0, 0, 255)      # red
COLOR_TR = (0, 255, 0)      # green
COLOR_BR = (255, 0, 0)      # blue
COLOR_BL = (0, 255, 255)    # yellow
CORNER_COLORS = [COLOR_TL, COLOR_TR, COLOR_BR, COLOR_BL]
LINE_COLOR = (255, 255, 255)
SLOT_NAMES = ("TL", "TR", "BR", "BL")


def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    matches = sorted(p for p in BOARDS_DIR.rglob(name) if p.is_file())
    if matches:
        return matches[0]
    for ext in BOARD_IMAGE_EXTS:
        matches = sorted(p for p in BOARDS_DIR.rglob(f"{name}{ext}")
                         if p.is_file())
        if matches:
            return matches[0]
    return None


def load_group_map() -> dict[str, list[str]]:
    if not BOARD_GROUPS_FILE.is_file():
        print(f"No group map: {BOARD_GROUPS_FILE}")
        sys.exit(1)
    data = json.loads(BOARD_GROUPS_FILE.read_text())
    out: dict[str, list[str]] = {}
    for fname, key in data.get("assignments", {}).items():
        out.setdefault(key, []).append(fname)
    for key in out:
        out[key].sort()
    return out


def letterbox(image: np.ndarray, size: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving resize + black-pad to (size, size). Returns
    the resized image and the 3x3 matrix mapping source-pixel coords
    into cell-pixel coords (so saved corners can be transformed)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    M = np.array([
        [scale, 0.0,   float(pad_x)],
        [0.0,   scale, float(pad_y)],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    return canvas, M


def apply_h(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])
    out_h = (M @ pts_h.T).T
    return out_h[:, :2] / out_h[:, 2:3]


def draw_corners(cell: np.ndarray, pts: np.ndarray) -> None:
    h, w = cell.shape[:2]
    for i in range(4):
        p1 = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        p2 = (int(round(pts[(i + 1) % 4, 0])),
              int(round(pts[(i + 1) % 4, 1])))
        cv2.line(cell, p1, p2, LINE_COLOR, 1, lineType=cv2.LINE_AA)
    for pt, color, name in zip(pts, CORNER_COLORS, SLOT_NAMES):
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        # Dot
        cv2.circle(cell, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(cell, (cx, cy), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        # Slot tag (TL/TR/BR/BL) so a wrong rotation is obvious even
        # to a colorblind reviewer. Place inward from the corner.
        tag_dx = -16 if cx > w / 2 else 6
        tag_dy = -6 if cy > h / 2 else 16
        cv2.putText(cell, name, (cx + tag_dx, cy + tag_dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(cell, name, (cx + tag_dx, cy + tag_dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    cv2.LINE_AA)


def truncate_label(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def label_strip(width: int, text: str, bg: tuple[int, int, int]
                ) -> np.ndarray:
    strip = np.full((LABEL_BAR, width, 3), bg, dtype=np.uint8)
    chars = max(1, width // 8)
    cv2.putText(strip, truncate_label(text, chars), (4, LABEL_BAR - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    return strip


def render_cell(fname: str, cell_size: int,
                store: dict, group_size: int) -> np.ndarray:
    """Returns a (cell_size + LABEL_BAR, cell_size, 3) BGR cell."""
    cell = np.full((cell_size, cell_size, 3), BG_COLOR, dtype=np.uint8)
    entry = store.get(fname)
    rejected = bool(entry and entry["corners_rejected"])

    path = find_image(fname)
    label_bg = BG_COLOR
    tag = None
    if path is None:
        cell[:] = MISSING_COLOR
        cv2.putText(cell, "MISSING", (8, cell_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
    else:
        image = load_board_image(path)
        if image is None:
            cell[:] = MISSING_COLOR
            cv2.putText(cell, "LOAD FAIL", (8, cell_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1,
                        cv2.LINE_AA)
        else:
            cell, M = letterbox(image, cell_size)
            if entry and entry["corners"] is not None and not rejected:
                pts_src = np.array(entry["corners"], dtype=np.float64)
                if pts_src.shape == (4, 2):
                    pts_cell = apply_h(M, pts_src)
                    draw_corners(cell, pts_cell)
                    if not entry["corners_verified"]:
                        tag = "UNVERIFIED"
                        label_bg = TAG_COLOR_NO_LABEL
                else:
                    tag = "BAD SHAPE"
                    label_bg = TAG_COLOR_NO_LABEL
            elif rejected:
                tag = "REJECTED"
                label_bg = TAG_COLOR_REJECTED
            else:
                tag = "NO LABEL"
                label_bg = TAG_COLOR_NO_LABEL

    if tag:
        cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, 18),
                      label_bg, -1)
        cv2.putText(cell, tag, (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)

    strip = label_strip(cell_size, fname, label_bg)
    return np.vstack([cell, strip])


def make_grid(cells: list[np.ndarray], cols: int, gap: int
              ) -> np.ndarray:
    rows = (len(cells) + cols - 1) // cols
    cell_h, cell_w = cells[0].shape[:2]
    grid_w = cols * cell_w + (cols + 1) * gap
    grid_h = rows * cell_h + (rows + 1) * gap
    canvas = np.full((grid_h, grid_w, 3), BG_COLOR, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r = idx // cols
        c = idx % cols
        y = gap + r * (cell_h + gap)
        x = gap + c * (cell_w + gap)
        canvas[y:y + cell_h, x:x + cell_w] = cell
    return canvas


def fit_window(image: np.ndarray, max_w: int, max_h: int
               ) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def wait_for_close(window_name: str) -> None:
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
    parser.add_argument("group", nargs="?", default=None,
                        help="Group key from board_groups.json "
                             "(e.g. 'AG' or 'U41').")
    parser.add_argument("--list", action="store_true",
                        help="List all groups + sizes and exit.")
    parser.add_argument("--cols", type=int, default=None,
                        help="Grid columns (default: ceil(sqrt(N))).")
    parser.add_argument("--cell", type=int, default=DEFAULT_CELL,
                        help=f"Cell size in pixels "
                             f"(default {DEFAULT_CELL}).")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save grid to this path instead of showing "
                             "a window.")
    parser.add_argument("--max-window", type=int, default=1600,
                        help="Cap window dimensions (default 1600x1000); "
                             "saved files keep the full resolution.")
    args = parser.parse_args()

    groups = load_group_map()

    if args.list:
        counts = Counter({k: len(v) for k, v in groups.items()})
        print(f"{len(groups)} groups, {sum(counts.values())} images")
        for key, n in sorted(counts.items(),
                             key=lambda kv: (-kv[1], kv[0])):
            print(f"  {key:>5}  {n:>3}")
        return

    if args.group is None:
        parser.error("provide a group key or pass --list")
    if args.group not in groups:
        print(f"Group {args.group!r} not in {BOARD_GROUPS_FILE}.")
        print(f"Run with --list to see all keys.")
        sys.exit(1)

    members = groups[args.group]
    store = corners_store.load()
    cols = args.cols or max(1, int(np.ceil(np.sqrt(len(members)))))

    print(f"Group {args.group!r}: {len(members)} image(s), "
          f"rendering at {args.cell}px/cell in {cols} column(s).")

    cells = [render_cell(fname, args.cell, store, len(members))
             for fname in members]
    grid = make_grid(cells, cols, DEFAULT_GAP)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(args.save), grid)
        if not ok:
            print(f"Failed to write {args.save}")
            sys.exit(1)
        print(f"Wrote {args.save}  ({grid.shape[1]}x{grid.shape[0]})")
        return

    shown = fit_window(grid, args.max_window, args.max_window * 5 // 8)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, shown.shape[1], shown.shape[0])
    cv2.imshow(WINDOW_NAME, shown)
    print("Press any key or close the window to exit.")
    wait_for_close(WINDOW_NAME)
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
