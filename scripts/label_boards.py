"""
Group raw board images by physical board.

For every image under data/raw/boards/ that doesn't yet have a group
assignment in data/annotations/board_groups.json, show the photo and
prompt for which group it belongs to. Two flavours of group key:

  - Named groups (A, B, ..., Z, AA, AB, ...) for boards you'll see
    more than once. Get a short description when created.
  - Unique groups (U01, U02, ...) for one-off boards.

The point of the grouping is the train/val/test split rule for the
corner regressor: every image with the same key must land in the
same split (see CLAUDE.md "Corner detection training").

At each prompt:
  <key>   assign this image to an existing group
  n       create a new named group (prompts for description)
  u       create a new unique group (auto-keyed U01, U02, ...)
  s       skip this image (ask again next run)
  q       quit (already-saved assignments are kept)

Assignments are written to disk after every choice so quitting or
Ctrl+C never loses work.

Usage:
    uv run scripts/label_boards.py
    uv run scripts/label_boards.py --display-max 700
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_io import is_board_image, load_board_image  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")
GROUPS_FILE = Path("data/annotations/board_groups.json")
WINDOW_NAME = "Board - which group?"
DEFAULT_MAX_DISPLAY = 800


def load_store() -> tuple[dict[str, str], dict[str, str]]:
    if not GROUPS_FILE.is_file():
        return {}, {}
    data = json.loads(GROUPS_FILE.read_text())
    return data.get("groups", {}), data.get("assignments", {})


def save_store(groups: dict[str, str], assignments: dict[str, str]) -> None:
    GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"groups": groups, "assignments": assignments}
    GROUPS_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))


def column_name(n: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    result = ""
    n += 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        result = chr(ord("A") + r) + result
    return result


def next_named_key(groups: dict[str, str]) -> str:
    i = 0
    while True:
        key = column_name(i)
        if key not in groups:
            return key
        i += 1


def next_unique_key(groups: dict[str, str]) -> str:
    i = 1
    while True:
        key = f"U{i:02d}"
        if key not in groups:
            return key
        i += 1


def fit_to_screen(image, max_dim: int):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))


def print_groups(groups: dict[str, str], assignments: dict[str, str]) -> None:
    if not groups:
        print("(no groups yet -- type 'n' to create one or 'u' for unique)")
        return
    counts: dict[str, int] = {}
    for key in assignments.values():
        counts[key] = counts.get(key, 0) + 1
    print("Groups:")
    for key in sorted(groups):
        print(f"  {key:>4}  ({counts.get(key, 0):2})  {groups[key]}")


def find_pending(assignments: dict[str, str]) -> list[Path]:
    if not BOARDS_DIR.is_dir():
        return []
    targets = []
    for p in sorted(BOARDS_DIR.rglob("*")):
        if not p.is_file() or not is_board_image(p):
            continue
        if p.name in assignments:
            continue
        targets.append(p)
    return targets


def prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return "q"


def show(image, max_display: int) -> None:
    cv2.imshow(WINDOW_NAME, fit_to_screen(image, max_display))
    # Two short waitKeys give the Qt backend a chance to actually
    # render the frame before input() blocks the main thread.
    cv2.waitKey(1)
    cv2.waitKey(30)


def assign_one(img_path: Path, idx: int, total: int,
               groups: dict[str, str], assignments: dict[str, str],
               max_display: int) -> str | None:
    """Return the chosen key, "__skip__", or "__quit__"."""
    image = load_board_image(img_path)
    if image is None:
        print(f"  Could not read image: {img_path}")
        return "__skip__"

    show(image, max_display)

    print(f"\n=== [{idx}/{total}] {img_path.name} ===")
    print_groups(groups, assignments)
    print("Type a group key, 'n' (new), 'u' (unique), 's' (skip), 'q' (quit).")

    while True:
        cmd = prompt("> ").strip()
        if not cmd:
            continue
        if cmd == "q":
            return "__quit__"
        if cmd == "s":
            return "__skip__"
        if cmd == "n":
            desc = prompt("Description: ").strip()
            if not desc:
                print("  (empty description -- try again)")
                continue
            key = next_named_key(groups)
            groups[key] = desc
            print(f"  Created group {key}: {desc}")
            return key
        if cmd == "u":
            desc = prompt("Description (optional): ").strip() or "(unique)"
            key = next_unique_key(groups)
            groups[key] = desc
            print(f"  Created unique group {key}: {desc}")
            return key
        if cmd in groups:
            return cmd
        print(f"  '{cmd}' is not a group key. Type 'n' for a new group, "
              f"'u' for unique, 's' to skip, 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--display-max", type=int, default=DEFAULT_MAX_DISPLAY,
                        help="Longest-edge display size in pixels "
                             "(default %(default)s).")
    args = parser.parse_args()

    if not BOARDS_DIR.is_dir():
        print(f"Boards directory not found: {BOARDS_DIR}")
        sys.exit(1)

    groups, assignments = load_store()
    targets = find_pending(assignments)
    if not targets:
        print("Nothing to label -- every image is already assigned.")
        return

    print(f"Found {len(targets)} unassigned image(s). "
          f"({len(assignments)} already assigned across "
          f"{len(groups)} group(s).)")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    assigned = skipped = 0
    try:
        for i, img_path in enumerate(targets, start=1):
            result = assign_one(img_path, i, len(targets),
                                groups, assignments, args.display_max)
            if result == "__quit__":
                print("\nQuitting.")
                break
            if result == "__skip__":
                skipped += 1
                print("  Skipped.")
                continue
            assignments[img_path.name] = result
            save_store(groups, assignments)
            assigned += 1
            print(f"  Assigned to {result}.")
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass

    print(f"\nAssigned {assigned}, skipped {skipped}, "
          f"{len(targets) - assigned - skipped} not yet seen.")
    print(f"Total groups: {len(groups)}, total assignments: {len(assignments)}.")


if __name__ == "__main__":
    main()
