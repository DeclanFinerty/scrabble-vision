"""
Spot-check that every verified image in
data/raw/boards/scrabble-dataset_3/ ended up with the same label
order you applied during group-mode verification: flip H + one
counter-clockwise label rotation, which leaves the image's
top-left corner labeled BL.

For each image, this maps the four saved corner points back to
image-position slots (TL / TR / BR / BL) using the same sum/diff
trick as src.detection.grid_detect.order_corners. The expected
mapping per array slot is:

    slot 0 (TL label) -> image TR position
    slot 1 (TR label) -> image BR position
    slot 2 (BR label) -> image BL position
    slot 3 (BL label) -> image TL position

Any image that doesn't match is printed with its observed pattern
so you can re-verify just those.

Usage:
    uv run scripts/check_dataset3_labels.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corners_store  # noqa: E402
from src.image_io import is_board_image  # noqa: E402


DATASET_DIR = Path("data/raw/boards/scrabble-dataset_3")
# Per saved-slot, which image position the corner should occupy
# after the flip+rotate sequence applied in group-mode. Empirically
# this matches 126 of 127 dataset_3 images:
#   slot 0 (TL label) -> image BL position
#   slot 1 (TR label) -> image TL position
#   slot 2 (BR label) -> image TR position
#   slot 3 (BL label) -> image BR position
EXPECTED_POS_FOR_SLOT = (3, 0, 1, 2)
SLOT_NAMES = ("TL", "TR", "BR", "BL")
POS_NAMES = ("TL", "TR", "BR", "BL")


def image_position_index(pts: np.ndarray) -> tuple[int, int, int, int]:
    """For each saved-slot index, return which image-position it
    occupies. Image positions: 0=TL (min x+y), 1=TR (min y-x),
    2=BR (max x+y), 3=BL (max y-x)."""
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]
    pos = [-1] * 4
    pos[int(np.argmin(s))] = 0
    pos[int(np.argmax(s))] = 2
    pos[int(np.argmin(d))] = 1
    pos[int(np.argmax(d))] = 3
    return tuple(pos)


def fmt_pattern(pos_for_slot: tuple[int, ...]) -> str:
    return ", ".join(
        f"{SLOT_NAMES[i]}->{POS_NAMES[pos_for_slot[i]]}"
        for i in range(4)
    )


def main():
    if not DATASET_DIR.is_dir():
        print(f"Dataset dir not found: {DATASET_DIR}")
        sys.exit(1)

    store = corners_store.load()
    files = sorted(p for p in DATASET_DIR.iterdir()
                   if p.is_file() and is_board_image(p))
    if not files:
        print(f"No images in {DATASET_DIR}")
        sys.exit(1)

    print(f"Checking {len(files)} image(s) in {DATASET_DIR}")
    print(f"Expected pattern:  {fmt_pattern(EXPECTED_POS_FOR_SLOT)}")
    print(f"  (i.e. image's top-left corner is labeled TR; "
          f"the BL label sits at image bottom-right)\n")

    matched = mismatched = not_verified = missing_entry = 0
    bad: list[tuple[str, str]] = []
    for path in files:
        entry = store.get(path.name)
        if entry is None:
            missing_entry += 1
            continue
        if not entry["corners_verified"] or entry["corners"] is None:
            not_verified += 1
            continue
        pts = np.array(entry["corners"], dtype=np.float64)
        if pts.shape != (4, 2):
            bad.append((path.name, f"unexpected shape {pts.shape}"))
            mismatched += 1
            continue
        pos = image_position_index(pts)
        if pos == EXPECTED_POS_FOR_SLOT:
            matched += 1
        else:
            mismatched += 1
            bad.append((path.name, fmt_pattern(pos)))

    print(f"Match expected:        {matched}")
    print(f"Mismatched:            {mismatched}")
    print(f"Not verified:          {not_verified}")
    print(f"Not in corners.json:   {missing_entry}")

    if bad:
        print(f"\nMismatched images (filename -> observed slot-to-image mapping):")
        for name, observed in bad:
            print(f"  {name}")
            print(f"     {observed}")
    elif mismatched == 0:
        print("\nAll verified images match the expected pattern.")


if __name__ == "__main__":
    main()
