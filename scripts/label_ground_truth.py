"""
Label the 15x15 letter grid for a board image.

Warps the board using the corners stored in
data/annotations/corners.json, displays it in an OpenCV window with a
grid overlay, and prompts row by row in the terminal. Each row must
be exactly 15 chars from {A-Z, a-z, '.'}:
  - Uppercase letter = printed tile face
  - Lowercase letter = blank tile playing as that letter
  - '.'              = empty / bonus square

Output is written to data/annotations/{stem}.txt (one row per line).

Usage:
    uv run scripts/label_ground_truth.py board_001.jpeg
    uv run scripts/label_ground_truth.py board_001
    uv run scripts/label_ground_truth.py --batch
"""

import argparse
import json
import string
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.grid_detect import perspective_correct, GRID_SIZE  # noqa: E402


BOARDS_DIR = Path("data/raw/boards")
ANNOTATIONS_DIR = Path("data/annotations")
CORNERS_FILE = ANNOTATIONS_DIR / "corners.json"
IMAGE_EXTS = (".jpeg", ".jpg", ".png")
OUTPUT_SIZE = 900
VALID_CHARS = set(string.ascii_letters) | {"."}


def load_corners() -> dict:
    if not CORNERS_FILE.exists():
        print(f"No corners file at {CORNERS_FILE}. "
              f"Run scripts/batch_detect_corners.py first.")
        sys.exit(1)
    with open(CORNERS_FILE) as f:
        return json.load(f)


def find_image(name: str) -> Path | None:
    direct = BOARDS_DIR / name
    if direct.is_file():
        return direct
    for ext in IMAGE_EXTS:
        p = BOARDS_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def render_overlay(warped: np.ndarray, highlight_row: int | None,
                   filename: str) -> np.ndarray:
    out = warped.copy()
    h, w = out.shape[:2]
    cell = w / GRID_SIZE

    if highlight_row is not None:
        y1 = int(highlight_row * cell)
        y2 = int((highlight_row + 1) * cell)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), (0, 220, 255), -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    for i in range(GRID_SIZE + 1):
        p = int(i * cell)
        cv2.line(out, (p, 0), (p, h), (0, 255, 0), 1)
        cv2.line(out, (0, p), (w, p), (0, 255, 0), 1)

    for r in range(GRID_SIZE):
        y = int((r + 0.5) * cell) + 4
        cv2.putText(out, f"{r:2d}", (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for c in range(GRID_SIZE):
        x = int((c + 0.5) * cell) - 6
        cv2.putText(out, f"{c}", (x, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    label = filename
    if highlight_row is not None:
        label += f"  —  row {highlight_row}"
    cv2.putText(out, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(out, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


def validate_row(s: str) -> str | None:
    if len(s) != GRID_SIZE:
        return f"row must be exactly {GRID_SIZE} chars (got {len(s)})"
    for ch in s:
        if ch not in VALID_CHARS:
            return f"invalid character {ch!r} (allowed: A-Z, a-z, .)"
    return None


def read_existing_gt(stem: str) -> list[str] | None:
    path = ANNOTATIONS_DIR / f"{stem}.txt"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) != GRID_SIZE:
        return None
    return lines


def prompt_existing(existing: list[str]) -> str:
    print("Existing ground truth:")
    for i, row in enumerate(existing):
        print(f"  {i:2d}: {row}")
    while True:
        choice = input("Re-enter (r), skip (s), or quit (q)? ").strip().lower()
        if choice in ("r", "re", "redo", "re-enter"):
            return "redo"
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("q", "quit"):
            return "quit"


def label_one(img_path: Path, corners_data: dict) -> str:
    """Returns 'saved', 'skipped', or 'quit'."""
    fname = img_path.name
    stem = img_path.stem

    if fname not in corners_data:
        print(f"  No corners for {fname} in {CORNERS_FILE}. "
              f"Run batch_detect_corners.py with --single {fname}.")
        return "skipped"
    corners = corners_data[fname]
    if corners is None:
        print(f"  Corners for {fname} are null (auto-detect failed). "
              f"Manually adjust before labeling.")
        return "skipped"

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  Could not read image: {img_path}")
        return "skipped"

    existing = read_existing_gt(stem)
    if existing is not None:
        choice = prompt_existing(existing)
        if choice == "skip":
            return "skipped"
        if choice == "quit":
            return "quit"

    corners_arr = np.array(corners, dtype=np.float32)
    warped = perspective_correct(image, corners_arr, output_size=OUTPUT_SIZE)

    window = "Ground Truth Labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, OUTPUT_SIZE, OUTPUT_SIZE)

    print(f"\nLabeling {fname}")
    print(f"  15 chars per row. A-Z=letter, a-z=blank-as-letter, .=empty.")
    print(f"  Type 'quit' on any row prompt to abort without saving.")

    rows: list[str] = []
    try:
        for r in range(GRID_SIZE):
            display = render_overlay(warped, highlight_row=r, filename=fname)
            cv2.imshow(window, display)
            cv2.waitKey(1)

            hint = ""
            if existing and r < len(existing):
                hint = f"  [prev: {existing[r]}]"
            while True:
                try:
                    s = input(f"  Row {r:2d}{hint}: ")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Aborted, not saving.")
                    return "quit"
                if s.strip().lower() == "quit":
                    print("  Aborted, not saving.")
                    return "quit"
                err = validate_row(s)
                if err:
                    print(f"    {err}")
                    continue
                rows.append(s)
                break
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    out_path = ANNOTATIONS_DIR / f"{stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"  Saved: {out_path}")
    return "saved"


def find_unlabeled(corners_data: dict) -> list[Path]:
    targets = []
    for fname, corners in sorted(corners_data.items()):
        if corners is None:
            continue
        img_path = BOARDS_DIR / fname
        if not img_path.is_file():
            continue
        if (ANNOTATIONS_DIR / f"{img_path.stem}.txt").exists():
            continue
        targets.append(img_path)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("image", nargs="?", default=None,
                        help="Board image filename (with or without extension)")
    parser.add_argument("--batch", action="store_true",
                        help="Process every board with corners but no ground truth")
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.error("provide an image name or --batch")
    if args.image and args.batch:
        parser.error("--batch and a positional image are mutually exclusive")

    corners_data = load_corners()

    if args.batch:
        targets = find_unlabeled(corners_data)
        if not targets:
            print("Nothing to do — every board with corners is already labeled.")
            return
        print(f"Found {len(targets)} unlabeled boards.\n")
        saved = skipped = 0
        for i, img_path in enumerate(targets):
            print(f"\n=== [{i + 1}/{len(targets)}] {img_path.name} ===")
            result = label_one(img_path, corners_data)
            if result == "saved":
                saved += 1
            elif result == "skipped":
                skipped += 1
            elif result == "quit":
                print("\nQuitting batch.")
                break
        print(f"\nSaved {saved}, skipped {skipped}, "
              f"{len(targets) - saved - skipped} remaining.")
        return

    img_path = find_image(args.image)
    if img_path is None:
        print(f"Image not found in {BOARDS_DIR}: {args.image}")
        sys.exit(1)
    label_one(img_path, corners_data)


if __name__ == "__main__":
    main()
