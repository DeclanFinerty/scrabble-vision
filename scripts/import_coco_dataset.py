"""
Import a COCO-format dataset of board images with corner annotations.

Two annotation shapes are auto-detected:
  - Roboflow "object detection" task with a `CORNER` category
    contributing 4 small bboxes per image (one per corner).
  - Roboflow "keypoint detection" task with one annotation per
    image whose `keypoints` array holds 4 (x, y, visibility) triples.

Two side effects on the dataset folder:
  1. Each image is renamed in place to its `extra.name` (the short
     human-friendly name from the COCO record), and the COCO json is
     rewritten so `file_name` matches.
  2. For every image with 4 visible corner points, the corners are
     ordered TL/TR/BR/BL via order_corners and merged into
     data/annotations/corners.json (keyed by the new short
     filename), so the labeling tool can use them as seeds.

Re-running is safe — already-renamed files are detected and skipped,
and existing corner entries are preserved unless --force-corners is
passed.

Usage:
    uv run scripts/import_coco_dataset.py \\
        --coco data/raw/boards/scrabble-dataset_1/_annotations.coco.json
    uv run scripts/import_coco_dataset.py \\
        --coco data/raw/boards/scrabble-dataset_3/_annotations.coco.json
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


def extract_corners_by_image(coco: dict
                             ) -> tuple[dict[int, list[list[float]]], str | None]:
    """Pull corner points out of a COCO json. Auto-detects two
    Roboflow export shapes:

      1. "CORNER" category with 4 small bboxes per image (one per
         corner). Used by Roboflow's object-detection task.
      2. COCO keypoints: one annotation per image whose `keypoints`
         array holds 4 (x, y, visibility) triples. Used by
         Roboflow's keypoint-detection task.

    Returns (corners_by_image_id, format_name). Empty + None if
    neither shape is present in the file."""
    # Format 1: CORNER bboxes.
    corner_cat_ids = {c["id"] for c in coco["categories"]
                      if c["name"] == CORNER_CATEGORY}
    if corner_cat_ids:
        out: dict[int, list[list[float]]] = {}
        for ann in coco["annotations"]:
            if ann["category_id"] not in corner_cat_ids:
                continue
            out.setdefault(ann["image_id"], []).append(bbox_center(ann["bbox"]))
        return out, f"{CORNER_CATEGORY} bboxes"

    # Format 2: COCO keypoints. Visibility 0 means "not labeled" --
    # we skip those triples so partially-labeled images surface in
    # the incomplete report instead of getting silently bad corners.
    saw_keypoints = False
    out = {}
    for ann in coco["annotations"]:
        kp = ann.get("keypoints") or []
        if len(kp) < 12:
            continue
        saw_keypoints = True
        pts = []
        for i in range(4):
            x, y, v = kp[i * 3], kp[i * 3 + 1], kp[i * 3 + 2]
            if v == 0:
                continue
            pts.append([float(x), float(y)])
        if pts:
            out.setdefault(ann["image_id"], []).extend(pts)
    if saw_keypoints:
        return out, "keypoints"
    return {}, None


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

    corners_by_image, fmt = extract_corners_by_image(coco)
    if fmt is None:
        print(f"No usable corner annotations found in {args.coco}.")
        print(f"Expected one of:")
        print(f"  - a '{CORNER_CATEGORY}' category with one bbox per corner, or")
        print(f"  - keypoint annotations with 4 (x, y, visibility) triples")
        sys.exit(1)
    print(f"Annotation format: {fmt}")

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

        pts_list = corners_by_image.get(img["id"], [])
        if len(pts_list) == 4:
            pts = np.array(pts_list, dtype=np.float32)
            ordered = order_corners(pts)
            new_corners[new_name] = ordered.tolist()
        else:
            incomplete.append((new_name, len(pts_list)))

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
