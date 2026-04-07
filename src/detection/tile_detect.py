"""
Tile presence detector.

Instead of asking "does this look like a tile?", we ask "does this
look like its expected empty state?" Each position has a known empty
appearance based on the bonus square map. Bonus squares (TW, DW, TL,
DL) are colored and have higher saturation than wooden tiles. Normal
squares are detected by checking for the absence of letter content.
"""

import cv2
import numpy as np


# ── Hardcoded Bonus Square Map ───────────────────────────────────
# Standard 15x15 Scrabble board layout.
# These positions are the same on EVERY standard board.
# Key: (row, col) -> square type

BONUS_SQUARES = {}

# Triple Word Score (red corners and edges)
for r, c in [(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)]:
    BONUS_SQUARES[(r,c)] = "TW"

# Double Word Score (pink diagonal and edges)
for r, c in [(1,1),(2,2),(3,3),(4,4),(10,10),(11,11),(12,12),(13,13),
             (1,13),(2,12),(3,11),(4,10),(10,4),(11,3),(12,2),(13,1),
             (7,7)]:  # center star is also DW
    BONUS_SQUARES[(r,c)] = "DW"

# Triple Letter Score (blue)
for r, c in [(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),
             (9,1),(9,5),(9,9),(9,13),(13,5),(13,9)]:
    BONUS_SQUARES[(r,c)] = "TL"

# Double Letter Score (light blue)
for r, c in [(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),
             (6,2),(6,6),(6,8),(6,12),
             (7,3),(7,11),
             (8,2),(8,6),(8,8),(8,12),
             (11,0),(11,7),(11,14),(12,6),(12,8),
             (14,3),(14,11)]:
    BONUS_SQUARES[(r,c)] = "DL"


def is_bonus_square(row: int, col: int) -> bool:
    """Check if a position is a bonus square."""
    return (row, col) in BONUS_SQUARES


def get_square_type(row: int, col: int) -> str:
    """Get the bonus type: 'TW', 'DW', 'TL', 'DL', or 'normal'."""
    return BONUS_SQUARES.get((row, col), "normal")


# ── Expected empty hue ranges per square type (OpenCV H: 0-179) ──
# These are broad ranges — the key signal is saturation, not exact hue.
EMPTY_HUE_RANGES = {
    "TW": ((0, 15), (160, 179)),     # red
    "DW": ((0, 20), (150, 179)),     # pink/rose
    "TL": ((95, 135),),              # dark blue
    "DL": ((85, 125),),              # light blue
}

EMPTY_SAT_MIN = {
    "TW": 40,
    "DW": 25,
    "TL": 35,
    "DL": 25,
}


def _hue_in_ranges(hue: float, ranges: tuple) -> bool:
    for lo, hi in ranges:
        if lo <= hue <= hi:
            return True
    return False


# ── Tile Detection ───────────────────────────────────────────────

def detect_tile_presence(cell_image: np.ndarray,
                          row: int = -1, col: int = -1) -> tuple[bool, float]:
    """
    Determine if a cell contains a tile by checking whether it looks
    like its expected empty state.

    Bonus squares: empty = colored (high saturation, matching hue).
      If saturated + correct hue → empty. Otherwise → tile.
    Normal squares: empty = plain (no letter content).
      If significant dark letter-like content → tile. Otherwise → empty.
    """
    if cell_image.size == 0 or cell_image.shape[0] < 4 or cell_image.shape[1] < 4:
        return False, 0.0

    h, w = cell_image.shape[:2]
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    square_type = get_square_type(row, col) if row >= 0 and col >= 0 else "normal"

    if square_type != "normal":
        return _detect_bonus_square(hsv, square_type)
    else:
        return _detect_normal_square(gray, hsv)


def _detect_bonus_square(hsv: np.ndarray, square_type: str) -> tuple[bool, float]:
    """
    Bonus square detection: if the cell is saturated and matches the
    expected hue for this square type, it's empty (the colored square
    is showing). If not, a tile is covering it.
    """
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_hue = float(np.mean(hsv[:, :, 0]))

    hue_ranges = EMPTY_HUE_RANGES.get(square_type)
    sat_min = EMPTY_SAT_MIN.get(square_type, 30)

    if hue_ranges is None:
        return False, 0.5

    hue_match = _hue_in_ranges(mean_hue, hue_ranges)
    sat_match = mean_sat >= sat_min

    if hue_match and sat_match:
        # Matches expected empty color → no tile
        confidence = min(1.0, 0.5 + (mean_sat - sat_min) / 100)
        return False, confidence
    elif sat_match and not hue_match:
        # Saturated but wrong hue — probably still empty (board color variation)
        # or could be a tile. Low confidence either way.
        return False, 0.4
    else:
        # Low saturation = wooden tile covering the colored square
        confidence = min(1.0, 0.5 + (sat_min - mean_sat) / 80)
        return True, confidence


def _detect_normal_square(gray: np.ndarray, hsv: np.ndarray) -> tuple[bool, float]:
    """
    Normal square detection: empty normal squares are plain and uniform.
    Tiles on normal squares have dark letter content. Also check
    saturation — a saturated cell at a "normal" position might be a
    mis-aligned bonus square, so lean toward empty.
    """
    h, w = gray.shape[:2]

    # Check saturation first — if unexpectedly saturated, probably
    # grid line or color bleed from adjacent bonus square, not a tile
    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > 70:
        return False, 0.6

    # Look for dark letter content in the center
    margin = max(3, min(h, w) // 5)
    center = gray[margin:-margin, margin:-margin]
    if center.size == 0:
        return False, 0.3

    center_mean = float(np.mean(center))
    center_std = float(np.std(center))

    # Adaptive threshold to find dark marks
    block_size = max(11, (min(h, w) // 3)) | 1
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )

    # Only look at the center region for letter content
    center_thresh = thresh[margin:-margin, margin:-margin]
    dark_ratio = np.count_nonzero(center_thresh) / center_thresh.size

    # Tiles have a letter: moderate dark content (5-40%) with some contrast
    # Empty squares: very little dark content (<3%) and low std dev
    if dark_ratio > 0.06 and center_std > 20:
        confidence = min(1.0, 0.5 + dark_ratio * 2)
        return True, confidence
    elif dark_ratio < 0.03 and center_std < 15:
        return False, 0.8
    elif dark_ratio < 0.03:
        return False, 0.6
    else:
        # Borderline — small amount of dark content could be noise or faint text
        has_tile = dark_ratio > 0.05
        confidence = 0.45
        return has_tile, confidence


def detect_tiles_batch(cell_images: list[tuple]) -> list[tuple]:
    """
    Run tile detection on all cells.

    Args:
        cell_images: list of (row, col, cell_image).

    Returns:
        list of (row, col, cell_image, has_tile, confidence).
    """
    results = []
    for row, col, img in cell_images:
        has_tile, conf = detect_tile_presence(img, row, col)
        results.append((row, col, img, has_tile, conf))
    return results


def debug_tile_detection(cell_images: list[tuple],
                          board_image: np.ndarray,
                          grid_cells: list[tuple],
                          output_path: str = "debug_tiles.jpg"):
    """
    Save debug image.
    Green = tile detected, Red = empty.
    Blue outline = bonus square position.
    """
    debug = board_image.copy()
    detections = detect_tiles_batch(cell_images)

    tile_count = 0
    for det, gc in zip(detections, grid_cells):
        row, col, img, has_tile, conf = det
        _, _, x1, y1, x2, y2 = gc

        # Tile/empty color
        color = (0, 255, 0) if has_tile else (0, 0, 255)
        thickness = 2 if has_tile else 1
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)

        # Mark bonus squares with blue corners
        if is_bonus_square(row, col):
            sq_type = get_square_type(row, col)
            cv2.putText(debug, sq_type, (x1 + 2, y1 + 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 100, 0), 1)

        cv2.putText(debug, f"{conf:.0%}", (x1 + 2, y2 - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        if has_tile:
            tile_count += 1

    cv2.imwrite(output_path, debug)
    print(f"Tile debug: {output_path}")
    print(f"  Tiles: {tile_count}, Empty: {len(detections) - tile_count}")
    print(f"  (Expected: ~60-80 tiles in a typical mid-game board)")