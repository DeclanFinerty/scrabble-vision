"""
Run auto_detect_corners on every image in data/raw/boards/ and write
results to data/annotations/corners.json.

The JSON is keyed by image filename (with extension); each value is
either a 4x2 list of [x, y] pairs (TL, TR, BR, BL order from
order_corners) or null on failure.

Successive runs merge with existing entries — only newly-seen images
are detected. Use --force to redetect everything, or --single NAME
to redetect a specific image.

Usage:
    uv run scripts/batch_detect_corners.py
    uv run scripts/batch_detect_corners.py --force
    uv run scripts/batch_detect_corners.py --single board_001.jpeg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.grid_detect import auto_detect_corners  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")
CORNERS_FILE = Path("data/annotations/corners.json")
IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def load_existing() -> dict:
    if not CORNERS_FILE.exists():
        return {}
    with open(CORNERS_FILE) as f:
        return json.load(f)


def save_corners(data: dict):
    CORNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORNERS_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def detect_one(img_path: Path) -> list | None:
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    corners = auto_detect_corners(image)
    if corners is None:
        return None
    return corners.tolist()


def resolve_single(name: str) -> Path | None:
    """Accept either a full filename or a stem; return the resolved Path."""
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    for ext in IMAGE_EXTS:
        p = BOARDS_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--single", default=None,
                        help="Process a single image by name (with or without extension)")
    parser.add_argument("--force", action="store_true",
                        help="Re-detect even if an entry already exists")
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
            p for p in BOARDS_DIR.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

    if not images:
        print(f"No images found in {BOARDS_DIR}")
        return

    existing = load_existing()

    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for img_path in images:
        fname = img_path.name
        if not args.force and fname in existing:
            skipped.append(fname)
            continue

        corners = detect_one(img_path)
        existing[fname] = corners
        if corners is None:
            failed.append(fname)
            print(f"  [FAIL] {fname}")
        else:
            succeeded.append(fname)
            print(f"  [ OK ] {fname}")

    save_corners(existing)

    print()
    print(f"Detected:  {len(succeeded)}")
    print(f"Failed:    {len(failed)}")
    if skipped:
        print(f"Skipped:   {len(skipped)} (already had entries; use --force to redo)")
    print(f"Total entries in {CORNERS_FILE}: {len(existing)}")

    if failed:
        print("\nFailed filenames:")
        for fname in failed:
            print(f"  {fname}")


if __name__ == "__main__":
    main()
