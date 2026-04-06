"""
Tile presence detector.

Key insight: bonus square positions are IDENTICAL on every standard
Scrabble board (they're part of the game rules, not the board design).
We hardcode these positions and focus detection on one question:
"Is there a wooden tile sitting on this square, or is the square exposed?"

This generalizes across all board styles (black grid, white grid,
colored boards, etc.) because tiles always look similar — wooden,
raised, with a dark letter — while board squares vary wildly.
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


# ── Tile Detection ───────────────────────────────────────────────

def detect_tile_presence(cell_image: np.ndarray,
                          row: int = -1, col: int = -1) -> tuple[bool, float]:
    """
    Determine if a cell contains a tile.

    Focuses on detecting the TILE (wooden, raised, letter imprint)
    rather than trying to identify board square types, which vary
    across board designs.

    Signals used:
    1. Color: tiles are warm beige/tan in a narrow HSV range
    2. Saturation: tiles have low saturation (wood); colored bonus
       squares (blue, pink, red) have higher saturation
    3. Texture: tiles have a uniform wood center with one large
       dark mark (the letter); empty squares have either nothing
       or scattered printed text
    4. Edge contrast: tiles are raised and create border shadows

    Args:
        cell_image: BGR image of a single cell.
        row, col: grid position (used for bonus square prior).

    Returns:
        (has_tile, confidence)
    """
    if cell_image.size == 0 or cell_image.shape[0] < 4 or cell_image.shape[1] < 4:
        return False, 0.0

    h, w = cell_image.shape[:2]
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    score = 0.0

    # ── Signal 1: Tile color match ──
    # Scrabble tiles are warm beige/tan wood.
    # This is consistent across all board types.
    tile_mask = cv2.inRange(
        hsv,
        np.array([10, 15, 145]),   # H, S, V low
        np.array([32, 100, 245])   # H, S, V high
    )
    tile_fraction = np.count_nonzero(tile_mask) / tile_mask.size

    if tile_fraction > 0.55:
        score += 0.35
    elif tile_fraction > 0.35:
        score += 0.20
    elif tile_fraction > 0.20:
        score += 0.08

    # ── Signal 2: Low saturation ──
    # Wood tiles have low saturation. Colored bonus squares
    # (blue, pink, red) have much higher saturation.
    mean_sat = float(np.mean(hsv[:, :, 1]))

    if mean_sat < 30:
        score += 0.20    # very desaturated = likely tile or plain board
    elif mean_sat < 50:
        score += 0.10
    elif mean_sat > 70:
        score -= 0.25    # saturated color = bonus square, not a tile

    # ── Signal 3: Dark content in center ──
    # Tiles have a dark letter imprint in the center.
    # Empty squares either have nothing (plain) or small scattered text.
    margin = max(3, min(h, w) // 4)
    center = gray[margin:-margin, margin:-margin]

    if center.size > 0:
        center_mean = float(np.mean(center))
        center_min = float(np.min(center))

        # Tiles have a bright center (wood) with dark marks (letter)
        # The minimum pixel value in the center indicates dark letter strokes
        dark_contrast = center_mean - center_min

        if dark_contrast > 60 and center_mean > 140:
            # Bright background with dark marks = likely tile with letter
            score += 0.25
        elif dark_contrast > 30 and center_mean > 130:
            score += 0.10
        elif center_mean < 100:
            # Very dark cell = probably a dark-colored bonus square
            score -= 0.10

    # ── Signal 4: Border contrast ──
    # Tiles are raised above the board and create edge shadows.
    # Compute brightness difference between border and center.
    border_width = max(2, min(h, w) // 6)

    top_strip = gray[:border_width, :]
    bottom_strip = gray[-border_width:, :]
    left_strip = gray[:, :border_width]
    right_strip = gray[:, -border_width:]

    border_mean = np.mean(np.concatenate([
        top_strip.flatten(), bottom_strip.flatten(),
        left_strip.flatten(), right_strip.flatten()
    ]))

    if center.size > 0:
        border_center_diff = abs(float(np.mean(center)) - border_mean)
        if border_center_diff > 15:
            score += 0.10  # visible border = raised tile
        elif border_center_diff > 8:
            score += 0.05

    # ── Bonus square prior ──
    # If we know this is a bonus square position and the score is
    # borderline, lean toward "empty" since bonus squares are more
    # often empty than covered (most games use ~60-80 of 225 squares).
    if row >= 0 and col >= 0 and is_bonus_square(row, col):
        if 0.30 < score < 0.55:
            score -= 0.08  # nudge borderline bonus squares toward empty

    has_tile = score >= 0.45
    confidence = max(0.0, min(1.0, score))

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