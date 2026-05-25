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
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.corners_store import CornerEntry, empty_entry  # noqa: E402
from src.detection.grid_detect import GridFitUI, auto_detect_corners  # noqa: E402
from src.image_io import (  # noqa: E402
    BOARD_IMAGE_EXTS, is_board_image, load_board_image, save_board_image,
)


BOARDS_DIR = Path("data/raw/boards")


def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    # Sort so the top-level file is preferred over any `original/`
    # backup that group-mode created during a flip-and-save.
    matches = sorted(p for p in BOARDS_DIR.rglob(name) if p.is_file())
    if matches:
        return matches[0]
    for ext in BOARD_IMAGE_EXTS:
        matches = sorted(p for p in BOARDS_DIR.rglob(f"{name}{ext}")
                         if p.is_file())
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
        # If Flip H or Flip V was applied during this session, persist
        # the flipped image back to disk so future loads see the same
        # orientation the corners were verified against.
        if ui.image_modified:
            save_board_image(img_path, ui.current_image)
            print(f"  Wrote flipped image: {img_path}")
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


# ─── Group-mode helpers ────────────────────────────────────────────

BOARD_GROUPS_FILE = Path("data/annotations/board_groups.json")


def _is_settled(entry: CornerEntry) -> bool:
    """True once a human has decided about this image -- whether they
    verified it or rejected it, we shouldn't ask again in normal flow."""
    return entry["corners_verified"] or entry["corners_rejected"]


def save_with_backup(path: Path, image_bgr: np.ndarray) -> None:
    """Save image_bgr to `path`, moving the existing file to
    `<path.parent>/original/<path.name>` first. One-shot backup --
    if an original already exists for this filename we keep that
    one and just overwrite the working copy (the true original is
    still preserved)."""
    backup_dir = path.parent / "original"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / path.name
    if backup_path.exists():
        print(f"  (backup already exists, keeping it: {backup_path})")
    else:
        path.rename(backup_path)
    save_board_image(path, image_bgr)


def load_board_group_map() -> dict[str, list[str]]:
    """Returns {group_key: [filename, ...]} from board_groups.json."""
    if not BOARD_GROUPS_FILE.is_file():
        return {}
    data = json.loads(BOARD_GROUPS_FILE.read_text())
    out: dict[str, list[str]] = {}
    for fname, key in data.get("assignments", {}).items():
        out.setdefault(key, []).append(fname)
    for key in out:
        out[key].sort()
    return out


def resolve_paths(filenames: list[str]) -> list[Path]:
    """Resolve filenames to on-disk paths, dropping any that vanished."""
    paths = []
    for fname in filenames:
        p = find_image(fname)
        if p is not None:
            paths.append(p)
        else:
            print(f"  (group member not found on disk: {fname})")
    return paths


def apply_to_other(path: Path, first_w: int, first_h: int,
                   corners: np.ndarray, h_flipped: bool, v_flipped: bool,
                   store: dict[str, CornerEntry]) -> bool:
    """Replicate the first-image transforms onto a sibling image and
    record the (scaled) corners as verified. Returns False if the
    image couldn't be loaded."""
    image = load_board_image(path)
    if image is None:
        print(f"    [skip] could not load: {path}")
        return False

    if h_flipped:
        image = image[:, ::-1].copy()
    if v_flipped:
        image = image[::-1, :].copy()
    if h_flipped or v_flipped:
        save_with_backup(path, image)

    h, w = image.shape[:2]
    scaled = corners.copy().astype(np.float64)
    scaled[:, 0] *= (w / first_w)
    scaled[:, 1] *= (h / first_h)

    entry = store.get(path.name) or empty_entry()
    entry["corners"] = scaled.tolist()
    entry["corners_verified"] = True
    entry["corners_rejected"] = False
    store[path.name] = entry
    return True


def review_one_in_group(path: Path, first_w: int, first_h: int,
                        corners: np.ndarray, h_flipped: bool, v_flipped: bool,
                        store: dict[str, CornerEntry],
                        title: str, max_display: int) -> str:
    """Pre-apply the group's flips, seed GridFitUI with the scaled
    corners, then let the user fine-tune/verify just this image.
    Returns 'verified', 'rejected', 'skipped', or 'quit'."""
    image = load_board_image(path)
    if image is None:
        print(f"  Could not load: {path}")
        return "skipped"

    pre_flipped = h_flipped or v_flipped
    if h_flipped:
        image = image[:, ::-1].copy()
    if v_flipped:
        image = image[::-1, :].copy()

    h, w = image.shape[:2]
    seed = corners.copy().astype(np.float64)
    seed[:, 0] *= (w / first_w)
    seed[:, 1] *= (h / first_h)

    ui = GridFitUI(image,
                   initial_corners=seed.astype(np.float32),
                   max_display=max_display)
    action, final_corners = ui.run_until_action(title=title)

    if action == "quit":
        return "quit"
    if action == "skip":
        return "skipped"
    if action == "reject":
        entry = store.get(path.name) or empty_entry()
        entry["corners_rejected"] = True
        entry["corners_verified"] = False
        store[path.name] = entry
        corners_store.save(store)
        return "rejected"

    # action == "verify"
    if pre_flipped or ui.image_modified:
        save_with_backup(path, ui.current_image)
    entry = store.get(path.name) or empty_entry()
    entry["corners"] = final_corners.tolist()
    entry["corners_verified"] = True
    entry["corners_rejected"] = False
    store[path.name] = entry
    corners_store.save(store)
    return "verified"


def prompt_apply_choice() -> str:
    while True:
        ans = input("  > ").strip().lower()
        if ans in ("y", "n", "r"):
            return ans
        print("  Type y (yes apply to all), n (no, first only), "
              "or r (review each).")


def process_group(group_key: str, paths: list[Path],
                  store: dict[str, CornerEntry],
                  group_idx: int, group_total: int,
                  max_display: int) -> str:
    """Verify one board group. Returns 'continue' or 'quit'."""
    pending = [p for p in paths
               if not _is_settled(store.get(p.name) or empty_entry())]
    if not pending:
        return "continue"

    first_path = pending[0]
    others = pending[1:]

    group_label = f"[group {group_idx}/{group_total} {group_key!r}]"
    print(f"\n=== {group_label} {len(paths)} image(s), "
          f"{len(pending)} pending ===")

    # Fall back to normal flow for single-image groups (whether by
    # total or by remaining unverified count).
    if not others:
        title = f"{group_label} image 1/1  {first_path.name}"
        result = process_one(first_path, store, title, max_display)
        return "quit" if result == "quit" else "continue"

    # ── Verify the first image (with full GridFitUI affordances) ──
    image = load_board_image(first_path)
    if image is None:
        print(f"  Could not load first image: {first_path}")
        return "continue"
    entry = store.get(first_path.name) or empty_entry()
    seed_pts = (np.array(entry["corners"], dtype=np.float32)
                if entry["corners"] is not None else None)
    ui = GridFitUI(image,
                   initial_corners=initial_corners(image, seed_pts),
                   max_display=max_display)
    title = (f"{group_label} image 1/{len(pending)}  "
             f"FIRST: {first_path.name}")
    print(f"  ({'seed' if seed_pts is not None else 'auto-detect'}) "
          f"verifying {first_path.name}...")
    action, corners = ui.run_until_action(title=title)

    if action == "quit":
        return "quit"
    if action == "skip":
        print(f"  Skipped group {group_key}.")
        return "continue"
    if action == "reject":
        entry["corners_rejected"] = True
        entry["corners_verified"] = False
        store[first_path.name] = entry
        corners_store.save(store)
        print(f"  Rejected {first_path.name}; NOT propagating to "
              f"{len(others)} other(s).")
        return "continue"

    # action == "verify"
    h_flipped = bool(ui._h_flipped)
    v_flipped = bool(ui._v_flipped)
    first_h, first_w = ui.current_image.shape[:2]

    if ui.image_modified:
        save_with_backup(first_path, ui.current_image)
        print(f"  Wrote flipped image: {first_path.name}")
    entry["corners"] = corners.tolist()
    entry["corners_verified"] = True
    entry["corners_rejected"] = False
    store[first_path.name] = entry
    corners_store.save(store)
    print(f"  Verified {first_path.name}.")

    # ── Propagation prompt ──
    transforms = []
    if h_flipped:
        transforms.append("flip H")
    if v_flipped:
        transforms.append("flip V")
    xstr = (" + ".join(transforms) if transforms
            else "(no image transform, just corner positions)")
    print(f"\n  Apply to all {len(others)} other unverified image(s) in "
          f"group {group_key!r}? Transforms: {xstr}")
    print(f"  y = yes apply to all, n = no first only, r = review each")
    choice = prompt_apply_choice()

    if choice == "n":
        print(f"  Done with group {group_key}: only first verified.")
        return "continue"

    if choice == "y":
        n_ok = 0
        for j, other in enumerate(others, start=2):
            print(f"  [{group_label} image {j}/{len(pending)}] "
                  f"applying to {other.name}")
            if apply_to_other(other, first_w, first_h, corners,
                              h_flipped, v_flipped, store):
                n_ok += 1
        corners_store.save(store)
        print(f"  Auto-applied to {n_ok}/{len(others)} other image(s).")
        return "continue"

    # choice == "r" -- step through each remaining image with GridFitUI
    for j, other in enumerate(others, start=2):
        per_title = (f"{group_label} image {j}/{len(pending)}  "
                     f"REVIEW: {other.name}")
        print(f"  [{j}/{len(pending)}] {other.name}")
        result = review_one_in_group(other, first_w, first_h, corners,
                                     h_flipped, v_flipped, store,
                                     per_title, max_display)
        if result == "quit":
            return "quit"
    return "continue"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("image", nargs="?", default=None,
                        help="Single image (filename or stem)")
    parser.add_argument("--batch", action="store_true",
                        help="Process every unverified, non-rejected board")
    parser.add_argument("--group-mode", action="store_true",
                        help="Iterate by board group: verify first unlabeled "
                             "image in each group, then optionally apply the "
                             "same corners + image flips to the rest of the "
                             "group (with y/n/r prompt). Single-image groups "
                             "fall back to the normal flow.")
    parser.add_argument("--redo", action="store_true",
                        help="With a single image, re-verify even if already done")
    parser.add_argument("--display-max", type=int,
                        default=GridFitUI.DEFAULT_MAX_DISPLAY,
                        help="Longest-edge display size in pixels "
                             "(default %(default)s; shrink for small screens)")
    args = parser.parse_args()

    modes = sum([bool(args.image), args.batch, args.group_mode])
    if modes == 0:
        parser.error("provide an image name, --batch, or --group-mode")
    if modes > 1:
        parser.error("--batch, --group-mode, and a positional image are "
                     "mutually exclusive")

    store = corners_store.load()

    if args.group_mode:
        group_map = load_board_group_map()
        if not group_map:
            print(f"No board groups in {BOARD_GROUPS_FILE}. "
                  f"Run scripts/label_boards.py first.")
            sys.exit(1)
        # Resolve paths up-front so we can report counts honestly,
        # then drop groups that have no unverified members left.
        pending_groups: list[tuple[str, list[Path]]] = []
        for key in sorted(group_map):
            paths = resolve_paths(group_map[key])
            if not paths:
                continue
            n_pending = sum(
                1 for p in paths
                if not _is_settled(store.get(p.name) or empty_entry())
            )
            if n_pending == 0:
                continue
            pending_groups.append((key, paths))
        if not pending_groups:
            print("Nothing to verify -- every group is fully decided "
                  "(verified or rejected).")
            return
        print(f"Group mode: {len(pending_groups)} group(s) have pending "
              f"image(s).")
        for idx, (key, paths) in enumerate(pending_groups, start=1):
            result = process_group(key, paths, store,
                                   group_idx=idx,
                                   group_total=len(pending_groups),
                                   max_display=args.display_max)
            if result == "quit":
                print("\nQuitting group-mode batch.")
                break
        return

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
