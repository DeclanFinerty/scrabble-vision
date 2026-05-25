"""
Rename images in a COCO dataset folder to their short `extra.name`
and remap any existing entries in data/annotations/corners.json and
data/annotations/board_groups.json to the new filename.

This is the "I already labeled the long filenames and don't want to
lose those labels" path. Unlike import_coco_dataset.py, this script:

  - NEVER imports CORNER bbox annotations as seeds
  - NEVER writes a fresh entry into corners.json
  - NEVER modifies a corners.json entry's content (verification
    flags, augment_level, corner coordinates -- all untouched)
  - ONLY moves an existing entry from the long filename key to the
    short filename key (and the same for board_groups.json)

Safety:
  - Defaults to dry-run. Pass --apply to actually rename / remap.
  - Refuses to merge: if both the old and the new filename already
    have entries in corners.json (or board_groups.json), that image
    is skipped with a warning -- you fix it by hand.
  - Refuses to overwrite a file: if both old and new exist on disk,
    that image is skipped.
  - Prints the verified-corner count before and after; they must
    match, since this script only moves keys.

Usage:
    uv run scripts/rename_coco_files.py --coco data/raw/boards/.../_annotations.coco.json
    uv run scripts/rename_coco_files.py --coco ... --apply
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402


BOARD_GROUPS_FILE = Path("data/annotations/board_groups.json")


def load_board_groups() -> dict:
    if not BOARD_GROUPS_FILE.is_file():
        return {"groups": {}, "assignments": {}}
    data = json.loads(BOARD_GROUPS_FILE.read_text())
    data.setdefault("groups", {})
    data.setdefault("assignments", {})
    return data


def n_verified(store: dict) -> int:
    return sum(1 for e in store.values() if e["corners_verified"])


def describe_corner(entry) -> str:
    parts = []
    if entry["corners_verified"]:
        parts.append("verified")
    if entry["corners_rejected"]:
        parts.append("rejected")
    if entry["corners"] is not None and not parts:
        parts.append("has-corners")
    if not parts:
        parts.append("empty")
    parts.append(f"aug={entry['augment_level']}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--coco", required=True, type=Path,
                        help="Path to _annotations.coco.json")
    parser.add_argument("--apply", action="store_true",
                        help="Apply the renames + remaps. Default is dry-run.")
    args = parser.parse_args()

    if not args.coco.is_file():
        print(f"COCO file not found: {args.coco}")
        sys.exit(1)

    coco = json.loads(args.coco.read_text())
    img_dir = args.coco.parent

    store = corners_store.load()
    groups_data = load_board_groups()
    n_verified_before = n_verified(store)
    n_assignments_before = len(groups_data["assignments"])

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"COCO file: {args.coco}")
    print(f"corners.json verified entries:    {n_verified_before}")
    print(f"board_groups.json assignments:    {n_assignments_before}")
    print()

    plan: list[dict] = []
    skipped: list[tuple[str, str, str]] = []
    nothing_to_do: list[str] = []

    for idx, img in enumerate(coco["images"]):
        old_name = img["file_name"]
        new_name = img.get("extra", {}).get("name")
        if not new_name:
            skipped.append((old_name, "<none>", "no extra.name in COCO record"))
            continue

        # Identical names: COCO already in sync, just confirm file is there.
        if old_name == new_name:
            if (img_dir / new_name).is_file():
                nothing_to_do.append(old_name)
            else:
                skipped.append((old_name, new_name,
                                "COCO file_name already short but file is missing"))
            continue

        old_path = img_dir / old_name
        new_path = img_dir / new_name

        if old_path.is_file() and new_path.is_file():
            skipped.append((old_name, new_name,
                            "BOTH old and new files exist on disk -- refusing"))
            continue
        if not old_path.is_file() and not new_path.is_file():
            skipped.append((old_name, new_name,
                            "neither old nor new file exists on disk"))
            continue

        file_op = "rename" if old_path.is_file() else "already-renamed"

        corner_old = store.get(old_name)
        corner_new = store.get(new_name)
        if corner_old is not None and corner_new is not None:
            skipped.append((old_name, new_name,
                            f"corners.json has BOTH keys "
                            f"(old: {describe_corner(corner_old)}; "
                            f"new: {describe_corner(corner_new)}) -- refusing to clobber"))
            continue

        old_in_groups = old_name in groups_data["assignments"]
        new_in_groups = new_name in groups_data["assignments"]
        if old_in_groups and new_in_groups:
            skipped.append((old_name, new_name,
                            f"board_groups.json has BOTH keys "
                            f"(old: {groups_data['assignments'][old_name]}; "
                            f"new: {groups_data['assignments'][new_name]}) -- refusing"))
            continue

        plan.append({
            "img_idx": idx,
            "old_name": old_name,
            "new_name": new_name,
            "file_op": file_op,
            "remap_corner": corner_old is not None,
            "corner_state": describe_corner(corner_old) if corner_old else None,
            "remap_group": old_in_groups,
            "group_key": groups_data["assignments"][old_name] if old_in_groups else None,
        })

    print(f"Plan: {len(plan)} renames, {len(skipped)} skipped, "
          f"{len(nothing_to_do)} already done.")
    print()
    for p in plan:
        print(f"  {p['old_name']}")
        print(f"   -> {p['new_name']}  [{p['file_op']}]")
        if p["remap_corner"]:
            print(f"      corners.json: move key  ({p['corner_state']})")
        if p["remap_group"]:
            print(f"      board_groups.json: move assignment  "
                  f"(group {p['group_key']})")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for old, new, reason in skipped:
            print(f"  {old} -> {new}")
            print(f"     {reason}")

    if not args.apply:
        print("\n(dry-run -- pass --apply to make these changes)")
        return

    if not plan:
        print("\nNothing to do.")
        return

    print(f"\nApplying {len(plan)} renames...")
    for p in plan:
        if p["file_op"] == "rename":
            (img_dir / p["old_name"]).rename(img_dir / p["new_name"])
        if p["remap_corner"]:
            store[p["new_name"]] = store.pop(p["old_name"])
        if p["remap_group"]:
            groups_data["assignments"][p["new_name"]] = \
                groups_data["assignments"].pop(p["old_name"])
        coco["images"][p["img_idx"]]["file_name"] = p["new_name"]

    # Sanity-check that we only moved keys -- counts must be unchanged.
    n_verified_after = n_verified(store)
    n_assignments_after = len(groups_data["assignments"])
    if n_verified_after != n_verified_before:
        print(f"\nABORT: verified count changed "
              f"({n_verified_before} -> {n_verified_after}). "
              f"NOT writing files. Investigate before retrying.")
        sys.exit(2)
    if n_assignments_after != n_assignments_before:
        print(f"\nABORT: board_groups assignment count changed "
              f"({n_assignments_before} -> {n_assignments_after}). "
              f"NOT writing files. Investigate before retrying.")
        sys.exit(2)

    corners_store.save(store)
    BOARD_GROUPS_FILE.write_text(
        json.dumps(groups_data, indent=2, sort_keys=True))
    args.coco.write_text(json.dumps(coco, indent=2))

    print(f"\nDone. corners.json verified: {n_verified_before} -> {n_verified_after}")
    print(f"      board_groups.json assignments: "
          f"{n_assignments_before} -> {n_assignments_after}")
    print(f"      (counts must match -- if they don't, the script would have aborted)")


if __name__ == "__main__":
    main()
