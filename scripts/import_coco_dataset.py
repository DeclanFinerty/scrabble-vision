"""
Import a COCO-format dataset of board images with CORNER bbox
annotations.

Two side effects on the dataset folder:
  1. Each image is renamed in place to its `extra.name` (the short
     human-friendly name from the COCO record), and the COCO json is
     rewritten so `file_name` matches.
  2. For every board with 4 CORNER bboxes, the bbox centers are
     ordered TL/TR/BR/BL and merged into
     data/annotations/corners.json (keyed by the new short
     filename), so the labeling tool can use them as seeds.

Re-running is safe — already-renamed files are detected and skipped,
and existing corner entries are preserved unless --force-corners is
passed.

Usage:
    uv run scripts/import_coco_dataset.py \\
        --coco data/raw/boards/scrabble-dataset_1/_annotations.coco.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.corners_store import empty_entry  # noqa: E402
from src.detection.grid_detect import order_corners  # noqa: E402


CORNER_CATEGORY = "CORNER"


def bbox_center(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x + w / 2.0, y + h / 2.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--coco", required=True, type=Path,
                        help="Path to _annotations.coco.json")
    parser.add_argument("--force-corners", action="store_true",
                        help="Overwrite existing unverified entries in corners.json "
                             "(verified entries are NEVER overwritten)")
    args = parser.parse_args()

    if not args.coco.is_file():
        print(f"COCO file not found: {args.coco}")
        sys.exit(1)

    coco = json.loads(args.coco.read_text())
    img_dir = args.coco.parent

    corner_cat_ids = {c["id"] for c in coco["categories"]
                      if c["name"] == CORNER_CATEGORY}
    if not corner_cat_ids:
        print(f"No '{CORNER_CATEGORY}' category in {args.coco}")
        sys.exit(1)

    corners_by_image: dict[int, list[list[float]]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in corner_cat_ids:
            continue
        corners_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    renamed = already = skipped = 0
    new_corners: dict[str, list[list[float]]] = {}
    incomplete: list[tuple[str, int]] = []
    missing: list[str] = []

    for img in coco["images"]:
        old_name = img["file_name"]
        new_name = img.get("extra", {}).get("name")
        if not new_name:
            print(f"  no extra.name for {old_name}, skipping")
            skipped += 1
            continue

        old_path = img_dir / old_name
        new_path = img_dir / new_name

        if old_path != new_path:
            if old_path.is_file():
                if new_path.exists():
                    print(f"  target exists, refusing to overwrite: {new_path}")
                    skipped += 1
                    continue
                old_path.rename(new_path)
                renamed += 1
            elif new_path.is_file():
                already += 1
            else:
                missing.append(old_name)
                skipped += 1
                continue
            img["file_name"] = new_name
        else:
            already += 1

        bboxes = corners_by_image.get(img["id"], [])
        if len(bboxes) == 4:
            pts = np.array([bbox_center(b) for b in bboxes], dtype=np.float32)
            ordered = order_corners(pts)
            new_corners[new_name] = ordered.tolist()
        else:
            incomplete.append((new_name, len(bboxes)))

    args.coco.write_text(json.dumps(coco, indent=2))

    print(f"\nFiles: {renamed} renamed, {already} already named, {skipped} skipped.")
    if missing:
        print(f"Missing files (no old or new path on disk):")
        for n in missing:
            print(f"  {n}")

    store = corners_store.load()
    added = updated = preserved_verified = preserved_other = 0
    for name, pts in new_corners.items():
        entry = store.get(name)
        if entry is None:
            entry = empty_entry()
            entry["corners"] = pts
            store[name] = entry
            added += 1
            continue
        if entry["corners_verified"]:
            preserved_verified += 1
            continue
        if entry["corners"] is not None and not args.force_corners:
            preserved_other += 1
            continue
        entry["corners"] = pts
        store[name] = entry
        updated += 1
    corners_store.save(store)

    print(f"\nCorners -> {corners_store.CORNERS_FILE}")
    print(f"  added:                {added}")
    print(f"  updated unverified:   {updated}")
    print(f"  preserved (verified): {preserved_verified}")
    print(f"  preserved (existing): {preserved_other} (use --force-corners to overwrite)")
    print(f"  incomplete:           {len(incomplete)} (not enough corners to seed)")
    if incomplete:
        for name, n in incomplete:
            print(f"    {name}: {n}/4")


if __name__ == "__main__":
    main()
