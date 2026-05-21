"""
Run auto_detect_corners on every image in data/raw/boards/ and write
the results to data/annotations/corners.json (via corners_store).

This script only fills in the `corners` field. It never sets or
clears verification flags — verify_corners.py owns those. By default
it skips:
  - entries that already have corners (verified, rejected, or just
    auto-detected previously), and
  - entries that have been verified or rejected by a human.

Use --force to re-detect entries that have corners but are not yet
verified. Verified entries are NEVER overwritten — the human has
spoken. --single NAME targets a specific image.

Usage:
    uv run scripts/batch_detect_corners.py
    uv run scripts/batch_detect_corners.py --force
    uv run scripts/batch_detect_corners.py --single board_001.jpeg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.corners_store import CornerEntry, empty_entry  # noqa: E402
from src.detection.grid_detect import auto_detect_corners  # noqa: E402
from src.image_io import BOARD_IMAGE_EXTS, is_board_image, load_board_image  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")


def detect_one(img_path: Path) -> list | None:
    image = load_board_image(img_path)
    if image is None:
        return None
    corners = auto_detect_corners(image)
    if corners is None:
        return None
    return corners.tolist()


def resolve_single(name: str) -> Path | None:
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


def should_skip(entry: CornerEntry | None, force: bool) -> str | None:
    """Return a reason to skip, or None if we should process."""
    if entry is None:
        return None
    if entry["corners_verified"]:
        return "verified"
    if entry["corners_rejected"]:
        return "rejected"
    if entry["corners"] is not None and not force:
        return "has corners (use --force)"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--single", default=None,
                        help="Process a single image by name (with or without extension)")
    parser.add_argument("--force", action="store_true",
                        help="Re-detect unverified entries that already have corners")
    args = parser.parse_args()

    if not BOARDS_DIR.is_dir():
        print(f"Boards directory not found: {BOARDS_DIR}")
        sys.exit(1)

    if args.single:
        path = resolve_single(args.single)
        if path is None:
            print(f"Image not found in {BOARDS_DIR}: {args.single}")
            sys.exit(1)
        images = [path]
    else:
        images = sorted(
            p for p in BOARDS_DIR.rglob("*")
            if p.is_file() and is_board_image(p)
        )

    if not images:
        print(f"No images found in {BOARDS_DIR}")
        return

    store = corners_store.load()

    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[tuple[str, str]] = []

    for img_path in images:
        fname = img_path.name
        reason = should_skip(store.get(fname), args.force)
        if reason is not None:
            skipped.append((fname, reason))
            continue

        corners = detect_one(img_path)
        entry = store.get(fname) or empty_entry()
        entry["corners"] = corners
        store[fname] = entry

        if corners is None:
            failed.append(fname)
            print(f"  [FAIL] {fname}")
        else:
            succeeded.append(fname)
            print(f"  [ OK ] {fname}")

    corners_store.save(store)

    print()
    print(f"Detected:  {len(succeeded)}")
    print(f"Failed:    {len(failed)}")
    if skipped:
        print(f"Skipped:   {len(skipped)}")
        # Summarise skip reasons
        by_reason: dict[str, int] = {}
        for _, r in skipped:
            by_reason[r] = by_reason.get(r, 0) + 1
        for r, n in sorted(by_reason.items()):
            print(f"           {n}  {r}")
    print(f"Total entries: {len(store)}")

    if failed:
        print("\nFailed filenames:")
        for fname in failed:
            print(f"  {fname}")


if __name__ == "__main__":
    main()
