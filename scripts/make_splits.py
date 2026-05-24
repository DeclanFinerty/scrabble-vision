"""
Build train/val/test split manifests for corner regressor training.

Reads:
  data/annotations/board_groups.json  -- which physical board each
                                         image came from
  data/annotations/corners.json       -- corner labels + verification

Only images with `corners_verified: true` are eligible. The board
groups are used purely to keep photos of the same physical board in
the same split (preventing leakage) -- the corner coordinates are
the actual training target.

Splits are built at the group level. Defaults match the current
project state:
  test:  group D
  val:   group A + 2 unique groups (chosen by seed)
  train: everything else (groups B, C, remaining unique groups)

Override with --test-groups, --val-groups, --val-unique-count, --seed.

Usage:
    uv run scripts/make_splits.py
    uv run scripts/make_splits.py --seed 42
    uv run scripts/make_splits.py --val-unique-count 3
    uv run scripts/make_splits.py --test-groups D --val-groups A
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402


GROUPS_FILE = Path("data/annotations/board_groups.json")
SPLITS_DIR = Path("data/splits")
SPLIT_FILES = {
    "train": SPLITS_DIR / "corner_train.txt",
    "val": SPLITS_DIR / "corner_val.txt",
    "test": SPLITS_DIR / "corner_test.txt",
}
# Unique groups (one-off boards) are auto-keyed U01, U02, ... by
# label_boards.py. Everything else is a "named" multi-photo group.
UNIQUE_KEY = re.compile(r"^U\d+$")


def is_unique_key(key: str) -> bool:
    return bool(UNIQUE_KEY.match(key))


def load_board_groups() -> tuple[dict[str, str], dict[str, str]]:
    if not GROUPS_FILE.is_file():
        print(f"Board groups file not found: {GROUPS_FILE}")
        print("Run scripts/label_boards.py first.")
        sys.exit(1)
    data = json.loads(GROUPS_FILE.read_text())
    return data.get("groups", {}), data.get("assignments", {})


def parse_keys(arg: str | None) -> list[str]:
    if not arg:
        return []
    return [k.strip() for k in arg.split(",") if k.strip()]


def write_split(path: Path, filenames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(filenames)
    if filenames:
        body += "\n"
    path.write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--test-groups", default="D",
                        help="Comma-separated group keys for the test set "
                             "(default: %(default)s).")
    parser.add_argument("--val-groups", default="A",
                        help="Comma-separated *named* group keys for val. "
                             "Unique groups are added separately via "
                             "--val-unique-count (default: %(default)s).")
    parser.add_argument("--val-unique-count", type=int, default=2,
                        help="Unique boards to add to val on top of "
                             "--val-groups (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for choosing the val unique "
                             "boards (default: %(default)s).")
    args = parser.parse_args()

    groups_meta, assignments = load_board_groups()
    corner_entries = corners_store.load()
    verified = {f for f, e in corner_entries.items() if e["corners_verified"]}

    if not verified:
        print("No verified corner entries -- nothing to split.")
        print("Run scripts/verify_corners.py first.")
        sys.exit(1)

    by_group: dict[str, list[str]] = defaultdict(list)
    unassigned: list[str] = []
    for fname in sorted(verified):
        key = assignments.get(fname)
        if key is None:
            unassigned.append(fname)
            continue
        by_group[key].append(fname)
    by_group = {k: sorted(v) for k, v in by_group.items() if v}

    if unassigned:
        print(f"Warning: {len(unassigned)} verified image(s) have no board "
              f"group assignment -- excluded from splits:")
        for f in unassigned[:10]:
            print(f"  {f}")
        if len(unassigned) > 10:
            print(f"  ... and {len(unassigned) - 10} more")
        print("(Run scripts/label_boards.py to assign them.)\n")

    test_keys = parse_keys(args.test_groups)
    val_named_keys = parse_keys(args.val_groups)

    available = set(by_group.keys())
    missing_test = [k for k in test_keys if k not in available]
    missing_val = [k for k in val_named_keys if k not in available]
    if missing_test:
        print(f"Error: test group(s) not present in data: {missing_test}")
        sys.exit(1)
    if missing_val:
        print(f"Error: val group(s) not present in data: {missing_val}")
        sys.exit(1)
    overlap = set(test_keys) & set(val_named_keys)
    if overlap:
        print(f"Error: group(s) in both test and val: {sorted(overlap)}")
        sys.exit(1)

    rng = random.Random(args.seed)
    candidate_unique = sorted(
        k for k in available
        if is_unique_key(k) and k not in test_keys and k not in val_named_keys
    )
    if args.val_unique_count > len(candidate_unique):
        print(f"Error: requested {args.val_unique_count} unique val boards "
              f"but only {len(candidate_unique)} available.")
        sys.exit(1)
    rng.shuffle(candidate_unique)
    val_unique_keys = sorted(candidate_unique[:args.val_unique_count])

    test_group_set = set(test_keys)
    val_group_set = set(val_named_keys) | set(val_unique_keys)
    train_group_set = available - test_group_set - val_group_set

    split_groups = {
        "test": sorted(test_group_set),
        "val": sorted(val_group_set),
        "train": sorted(train_group_set),
    }
    split_files = {
        name: sorted(f for k in keys for f in by_group[k])
        for name, keys in split_groups.items()
    }

    total = sum(len(v) for v in split_files.values())
    print(f"Verified images:    {len(verified)}")
    print(f"In splits (assigned & verified): {total}")
    print(f"Seed: {args.seed}\n")

    for name in ("test", "val", "train"):
        keys = split_groups[name]
        files = split_files[name]
        pct = (100 * len(files) / total) if total else 0.0
        print(f"[{name.upper():5}] {len(files):3} images  "
              f"({pct:5.1f}%)  across {len(keys)} group(s)")
        for k in keys:
            desc = groups_meta.get(k, "")
            print(f"          {k:>4} ({len(by_group[k]):3})  {desc}")
        print()

    for name, path in SPLIT_FILES.items():
        write_split(path, split_files[name])
        print(f"Wrote {path}  ({len(split_files[name])} lines)")


if __name__ == "__main__":
    main()
