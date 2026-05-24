"""
Set the per-image `augment_level` field on each verified board.

For each image under data/raw/boards/ that has verified corners,
show the photo in a cv2 window and prompt for which augmentation
profile the board should receive when scripts/augment_corners.py
regenerates the training set:

  f   full        rotation + perspective warp + random crop + color
  c   color_only  brightness/contrast/saturation/hue only
  s   skip        leave the entry alone (ask again next run)
  q   quit        already-saved choices are kept

"full" is the default for any entry that doesn't have augment_level
set yet. By default, images that already have an explicit level
are skipped. Pass --force to revisit every verified image.

Each choice is saved to data/annotations/corners.json immediately,
so quitting (or Ctrl+C) never loses work.

Usage:
    uv run scripts/set_augment_level.py
    uv run scripts/set_augment_level.py --force
    uv run scripts/set_augment_level.py --display-max 700
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.corners_store import AUGMENT_LEVELS, CornerEntry  # noqa: E402
from src.image_io import is_board_image, load_board_image  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")
WINDOW_NAME = "Augment level - f=full, c=color_only"
DEFAULT_MAX_DISPLAY = 800


def fit_to_screen(image, max_dim: int):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))


def show(image, max_display: int) -> None:
    cv2.imshow(WINDOW_NAME, fit_to_screen(image, max_display))
    # Two short waitKeys give the Qt backend time to actually flush
    # the frame before input() blocks the main thread.
    cv2.waitKey(1)
    cv2.waitKey(30)


def find_pending(store: dict[str, CornerEntry], force: bool,
                 prior_levels: dict[str, str | None]) -> list[Path]:
    if not BOARDS_DIR.is_dir():
        return []
    targets = []
    for p in sorted(BOARDS_DIR.rglob("*")):
        if not p.is_file() or not is_board_image(p):
            continue
        entry = store.get(p.name)
        if entry is None or not entry["corners_verified"]:
            continue
        if not force and prior_levels.get(p.name) is not None:
            continue
        targets.append(p)
    return targets


def counts(store: dict[str, CornerEntry]) -> tuple[int, int]:
    """(full_count, color_only_count) across verified entries."""
    n_full = n_color = 0
    for entry in store.values():
        if not entry["corners_verified"]:
            continue
        if entry["augment_level"] == "color_only":
            n_color += 1
        else:
            n_full += 1
    return n_full, n_color


def prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return "q"


def assign_one(img_path: Path, idx: int, total: int,
               store: dict[str, CornerEntry], max_display: int) -> str | None:
    """Returns the chosen level, "__skip__", or "__quit__"."""
    image = load_board_image(img_path)
    if image is None:
        print(f"  could not read {img_path}")
        return "__skip__"
    show(image, max_display)

    current = store[img_path.name]["augment_level"]
    n_full, n_color = counts(store)
    print(f"\n=== [{idx}/{total}] {img_path.name} ===")
    print(f"  current: {current}    "
          f"so far: full={n_full}  color_only={n_color}")
    print("  Type 'f' (full), 'c' (color_only), 's' (skip), 'q' (quit).")

    while True:
        cmd = prompt("> ").strip().lower()
        if not cmd:
            continue
        if cmd == "q":
            return "__quit__"
        if cmd == "s":
            return "__skip__"
        if cmd in ("f", "full"):
            return "full"
        if cmd in ("c", "color", "color_only"):
            return "color_only"
        print(f"  '{cmd}' is not a valid command. Use f/c/s/q.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true",
                        help="Revisit images that already have a level set.")
    parser.add_argument("--display-max", type=int, default=DEFAULT_MAX_DISPLAY,
                        help="Longest-edge display size in pixels "
                             "(default %(default)s).")
    args = parser.parse_args()

    if not BOARDS_DIR.is_dir():
        print(f"Boards directory not found: {BOARDS_DIR}")
        sys.exit(1)

    store = corners_store.load()

    # Snapshot which images had an explicit augment_level *before* we
    # started. Without this, the first save would write the default
    # "full" into every entry and immediately filter them out.
    raw = {}
    if corners_store.CORNERS_FILE.is_file():
        import json
        raw = json.loads(corners_store.CORNERS_FILE.read_text())
    prior_levels = {
        k: (v.get("augment_level") if isinstance(v, dict) else None)
        for k, v in raw.items()
    }

    targets = find_pending(store, args.force, prior_levels)
    if not targets:
        print("Nothing to label -- every verified image already has "
              "an augment_level. Pass --force to revisit them.")
        return

    n_full, n_color = counts(store)
    print(f"Found {len(targets)} image(s) to label. "
          f"Current totals: full={n_full}  color_only={n_color}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    assigned = skipped = 0
    try:
        for i, img_path in enumerate(targets, start=1):
            result = assign_one(img_path, i, len(targets), store,
                                args.display_max)
            if result == "__quit__":
                print("\nQuitting.")
                break
            if result == "__skip__":
                skipped += 1
                print("  Skipped.")
                continue
            store[img_path.name]["augment_level"] = result
            corners_store.save(store)
            assigned += 1
            print(f"  Set to {result}.")
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass

    n_full, n_color = counts(store)
    print(f"\nAssigned {assigned}, skipped {skipped}, "
          f"{len(targets) - assigned - skipped} not yet seen.")
    print(f"Verified totals now: full={n_full}  color_only={n_color}  "
          f"(of {sum(1 for e in store.values() if e['corners_verified'])} "
          f"verified entries; valid levels: {AUGMENT_LEVELS})")


if __name__ == "__main__":
    main()
