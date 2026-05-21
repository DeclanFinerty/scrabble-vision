"""
Stage 1 of the labeling pipeline — confirm the 4 board corners for
each raw image, one at a time.

All interaction happens in the cv2 window:
  - Drag the corner handles.
  - Click one of the buttons below the image:
      Verify  save adjusted corners and mark verified
      Reset   restore the seeded corners
      Skip    leave the entry alone (ask again next run)
      Reject  mark as unusable (won't be asked again, also
              blocks tile labeling for this board)
      Quit    stop the batch

Corner seeds come from data/annotations/corners.json — typically
import_coco_dataset.py and/or batch_detect_corners.py have already
filled those in. If no seed exists for an image, GridFitUI starts
from a fresh auto-detect.

Usage:
    uv run scripts/verify_corners.py --batch
    uv run scripts/verify_corners.py SCRABBLE_10
    uv run scripts/verify_corners.py SCRABBLE_10 --redo
    uv run scripts/verify_corners.py --batch --display-max 700
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.corners_store import CornerEntry, empty_entry  # noqa: E402
from src.detection.grid_detect import GridFitUI, auto_detect_corners  # noqa: E402
from src.image_io import BOARD_IMAGE_EXTS, is_board_image, load_board_image  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")


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


def initial_corners(image: np.ndarray, seed: np.ndarray | None) -> np.ndarray:
    if seed is not None:
        return seed
    auto = auto_detect_corners(image)
    if auto is not None:
        return auto
    h, w = image.shape[:2]
    return np.array([
        [w * 0.05, h * 0.05],
        [w * 0.95, h * 0.05],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.95],
    ], dtype=np.float32)


def process_one(img_path: Path, store: dict[str, CornerEntry],
                title: str, max_display: int) -> str:
    """Returns 'verified', 'rejected', 'skipped', or 'quit'."""
    fname = img_path.name
    image = load_board_image(img_path)
    if image is None:
        print(f"  Could not read image: {img_path}")
        return "skipped"

    entry = store.get(fname) or empty_entry()
    seed = np.array(entry["corners"], dtype=np.float32) \
        if entry["corners"] is not None else None
    source = "seed" if seed is not None else "auto-detect"
    print(f"  ({source}) waiting for click in window...")

    ui = GridFitUI(image,
                   initial_corners=initial_corners(image, seed),
                   max_display=max_display)
    action, corners = ui.run_until_action(title=title)

    if action == "verify":
        entry["corners"] = corners.tolist()
        entry["corners_verified"] = True
        entry["corners_rejected"] = False
        store[fname] = entry
        corners_store.save(store)
        print(f"  Verified.")
        return "verified"
    if action == "reject":
        entry["corners_rejected"] = True
        entry["corners_verified"] = False
        store[fname] = entry
        corners_store.save(store)
        print(f"  Rejected (won't be asked again).")
        return "rejected"
    if action == "quit":
        return "quit"
    print(f"  Skipped for now.")
    return "skipped"


def find_pending(store: dict[str, CornerEntry]) -> list[Path]:
    if not BOARDS_DIR.is_dir():
        return []
    targets = []
    for p in sorted(BOARDS_DIR.rglob("*")):
        if not p.is_file() or not is_board_image(p):
            continue
        entry = store.get(p.name)
        if entry and (entry["corners_verified"] or entry["corners_rejected"]):
            continue
        targets.append(p)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("image", nargs="?", default=None,
                        help="Single image (filename or stem)")
    parser.add_argument("--batch", action="store_true",
                        help="Process every unverified, non-rejected board")
    parser.add_argument("--redo", action="store_true",
                        help="With a single image, re-verify even if already done")
    parser.add_argument("--display-max", type=int,
                        default=GridFitUI.DEFAULT_MAX_DISPLAY,
                        help="Longest-edge display size in pixels "
                             "(default %(default)s; shrink for small screens)")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("provide an image name or --batch")
    if args.image and args.batch:
        parser.error("--batch and a positional image are mutually exclusive")

    store = corners_store.load()

    if args.batch:
        targets = find_pending(store)
        if not targets:
            print("Nothing to verify — every board is already verified or rejected.")
            return
        print(f"Found {len(targets)} boards to verify.\n")
        verified = rejected = skipped = 0
        for i, img_path in enumerate(targets):
            print(f"\n=== [{i + 1}/{len(targets)}] {img_path.name} ===")
            title = f"{i + 1}/{len(targets)}  {img_path.name}"
            result = process_one(img_path, store, title, args.display_max)
            if result == "verified":
                verified += 1
            elif result == "rejected":
                rejected += 1
            elif result == "skipped":
                skipped += 1
            elif result == "quit":
                print("\nQuitting batch.")
                break
        print(f"\nVerified {verified}, rejected {rejected}, skipped {skipped}, "
              f"{len(targets) - verified - rejected - skipped} remaining.")
        return

    img_path = find_image(args.image)
    if img_path is None:
        print(f"Image not found in {BOARDS_DIR}: {args.image}")
        sys.exit(1)
    entry = store.get(img_path.name)
    if entry and (entry["corners_verified"] or entry["corners_rejected"]) and not args.redo:
        state = "verified" if entry["corners_verified"] else "rejected"
        print(f"{img_path.name} is already {state}. Pass --redo to override.")
        return
    print(f"\n=== {img_path.name} ===")
    process_one(img_path, store, img_path.name, args.display_max)


if __name__ == "__main__":
    main()
