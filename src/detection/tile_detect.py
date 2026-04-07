"""
Tile presence detector with adaptive brightness calibration.

Two-pass detection: the primary detector uses per-square-type HSV
checks (high precision). An adaptive calibration pass learns the
board's brightness/saturation range and provides a secondary opinion.

A cell is a tile only if BOTH methods agree, or one is very confident.
A cell is empty if EITHER method says empty with reasonable confidence.
Conservative by design — better to miss a few tiles than flood with
false positives.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


# ── Hardcoded Bonus Square Map ───────────────────────────────────

BONUS_SQUARES = {}

for r, c in [(0,0),(0,7),(0,14),(7,0),(7,14),(14,0),(14,7),(14,14)]:
    BONUS_SQUARES[(r,c)] = "TW"

for r, c in [(1,1),(2,2),(3,3),(4,4),(10,10),(11,11),(12,12),(13,13),
             (1,13),(2,12),(3,11),(4,10),(10,4),(11,3),(12,2),(13,1),
             (7,7)]:
    BONUS_SQUARES[(r,c)] = "DW"

for r, c in [(1,5),(1,9),(5,1),(5,5),(5,9),(5,13),
             (9,1),(9,5),(9,9),(9,13),(13,5),(13,9)]:
    BONUS_SQUARES[(r,c)] = "TL"

for r, c in [(0,3),(0,11),(2,6),(2,8),(3,0),(3,7),(3,14),
             (6,2),(6,6),(6,8),(6,12),
             (7,3),(7,11),
             (8,2),(8,6),(8,8),(8,12),
             (11,0),(11,7),(11,14),(12,6),(12,8),
             (14,3),(14,11)]:
    BONUS_SQUARES[(r,c)] = "DL"


def is_bonus_square(row: int, col: int) -> bool:
    return (row, col) in BONUS_SQUARES


def get_square_type(row: int, col: int) -> str:
    return BONUS_SQUARES.get((row, col), "normal")


# ── Expected empty hue ranges per square type (OpenCV H: 0-179) ──

EMPTY_HUE_RANGES = {
    "TW": ((0, 15), (160, 179)),
    "DW": ((0, 20), (150, 179)),
    "TL": ((95, 135),),
    "DL": ((85, 125),),
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


# ── Adaptive Calibration ─────────────────────────────────────────

LIKELY_TILE_POSITIONS = [
    (r, c) for r in range(5, 10) for c in range(5, 10)
    if (r, c) not in BONUS_SQUARES
]

LIKELY_EMPTY_POSITIONS = [
    (0, 0), (0, 14), (14, 0), (14, 14),
    (0, 7), (14, 7), (7, 0), (7, 14),
]


@dataclass
class BoardProfile:
    """Learned brightness/color profile for a specific board image."""
    tile_brightness: float = 160.0
    tile_saturation: float = 30.0
    empty_brightness: float = 140.0
    empty_saturation: float = 60.0
    brightness_gap: float = 20.0
    calibrated: bool = False
    debug_info: dict = field(default_factory=dict)


def calibrate(cell_images: list[tuple]) -> BoardProfile:
    """Learn this board's brightness/saturation by sampling known positions."""
    cell_map = {(r, c): img for r, c, img in cell_images}
    profile = BoardProfile()

    tile_vals = []
    for r, c in LIKELY_TILE_POSITIONS:
        if (r, c) in cell_map:
            img = cell_map[(r, c)]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            tile_vals.append({
                "brightness": float(np.mean(gray)),
                "saturation": float(np.mean(hsv[:, :, 1])),
            })

    empty_vals = []
    for r, c in LIKELY_EMPTY_POSITIONS:
        if (r, c) in cell_map:
            img = cell_map[(r, c)]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            empty_vals.append({
                "brightness": float(np.mean(gray)),
                "saturation": float(np.mean(hsv[:, :, 1])),
            })

    if tile_vals:
        profile.tile_brightness = np.median([v["brightness"] for v in tile_vals])
        profile.tile_saturation = np.median([v["saturation"] for v in tile_vals])

    if empty_vals:
        profile.empty_brightness = np.median([v["brightness"] for v in empty_vals])
        profile.empty_saturation = np.median([v["saturation"] for v in empty_vals])

    profile.brightness_gap = abs(profile.tile_brightness - profile.empty_brightness)
    profile.calibrated = bool(tile_vals and empty_vals)

    profile.debug_info = {
        "tile_samples": len(tile_vals),
        "empty_samples": len(empty_vals),
        "tile_brightness": f"{profile.tile_brightness:.0f}",
        "tile_saturation": f"{profile.tile_saturation:.0f}",
        "empty_brightness": f"{profile.empty_brightness:.0f}",
        "empty_saturation": f"{profile.empty_saturation:.0f}",
        "brightness_gap": f"{profile.brightness_gap:.0f}",
    }

    return profile


def _adaptive_check(cell_image: np.ndarray, profile: BoardProfile) -> tuple[bool, float]:
    """Secondary brightness-based check using learned board profile."""
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    if profile.brightness_gap < 5:
        return False, 0.5  # can't distinguish, abstain

    b_dist_to_tile = abs(brightness - profile.tile_brightness)
    b_dist_to_empty = abs(brightness - profile.empty_brightness)

    if b_dist_to_tile < b_dist_to_empty:
        ratio = min(1.0, b_dist_to_empty / max(1, profile.brightness_gap))
        return True, 0.5 + ratio * 0.4
    else:
        ratio = min(1.0, b_dist_to_tile / max(1, profile.brightness_gap))
        return False, 0.5 + ratio * 0.4


# ── Primary Detection (high precision) ──────────────────────────

def _detect_bonus_square(hsv: np.ndarray, square_type: str) -> tuple[bool, float]:
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_hue = float(np.mean(hsv[:, :, 0]))

    hue_ranges = EMPTY_HUE_RANGES.get(square_type)
    sat_min = EMPTY_SAT_MIN.get(square_type, 30)

    if hue_ranges is None:
        return False, 0.5

    hue_match = _hue_in_ranges(mean_hue, hue_ranges)
    sat_match = mean_sat >= sat_min

    if hue_match and sat_match:
        confidence = min(1.0, 0.5 + (mean_sat - sat_min) / 100)
        return False, confidence
    elif sat_match and not hue_match:
        return False, 0.4
    else:
        confidence = min(1.0, 0.5 + (sat_min - mean_sat) / 80)
        return True, confidence


def _detect_normal_square(gray: np.ndarray, hsv: np.ndarray) -> tuple[bool, float]:
    h, w = gray.shape[:2]

    mean_sat = float(np.mean(hsv[:, :, 1]))
    if mean_sat > 70:
        return False, 0.6

    margin = max(3, min(h, w) // 5)
    center = gray[margin:-margin, margin:-margin]
    if center.size == 0:
        return False, 0.3

    center_std = float(np.std(center))

    block_size = max(11, (min(h, w) // 3)) | 1
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )

    center_thresh = thresh[margin:-margin, margin:-margin]
    dark_ratio = np.count_nonzero(center_thresh) / center_thresh.size

    if dark_ratio > 0.06 and center_std > 20:
        confidence = min(1.0, 0.5 + dark_ratio * 2)
        return True, confidence
    elif dark_ratio < 0.03 and center_std < 15:
        return False, 0.8
    elif dark_ratio < 0.03:
        return False, 0.6
    else:
        has_tile = dark_ratio > 0.05
        confidence = 0.45
        return has_tile, confidence


def _primary_detect(cell_image: np.ndarray, row: int, col: int) -> tuple[bool, float]:
    """High-precision primary detection using per-square-type HSV checks."""
    if cell_image.size == 0 or cell_image.shape[0] < 4 or cell_image.shape[1] < 4:
        return False, 0.0

    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    square_type = get_square_type(row, col) if row >= 0 and col >= 0 else "normal"

    if square_type != "normal":
        return _detect_bonus_square(hsv, square_type)
    else:
        return _detect_normal_square(gray, hsv)


# ── Combined Detection ──────────────────────────────────────────

HIGH_CONFIDENCE = 0.7

def detect_tile_presence(cell_image: np.ndarray, row: int = -1, col: int = -1,
                          profile: BoardProfile = None) -> tuple[bool, float]:
    """
    Two-pass detection combining primary (high precision) with
    adaptive brightness calibration.

    Tile if: both agree, or one is very confident (>0.7).
    Empty if: either says empty with reasonable confidence.
    """
    primary_tile, primary_conf = _primary_detect(cell_image, row, col)

    # Without calibration, use primary only
    if profile is None or not profile.calibrated:
        return primary_tile, primary_conf

    adaptive_tile, adaptive_conf = _adaptive_check(cell_image, profile)

    # Both agree → use that answer with combined confidence
    if primary_tile and adaptive_tile:
        return True, max(primary_conf, adaptive_conf)
    if not primary_tile and not adaptive_tile:
        return False, max(primary_conf, adaptive_conf)

    # Disagreement → conservative resolution
    # High-confidence override: if either is very sure, trust it
    if primary_tile and primary_conf >= HIGH_CONFIDENCE:
        return True, primary_conf
    if adaptive_tile and adaptive_conf >= HIGH_CONFIDENCE:
        return True, adaptive_conf

    # Otherwise: lean toward empty (conservative — fewer false positives)
    # Use the empty verdict's confidence
    if not primary_tile:
        return False, primary_conf
    else:
        return False, adaptive_conf


# ── Batch detection ──────────────────────────────────────────────

def detect_tiles_batch(cell_images: list[tuple],
                        profile: BoardProfile = None) -> list[tuple]:
    results = []
    for row, col, img in cell_images:
        has_tile, conf = detect_tile_presence(img, row, col, profile)
        results.append((row, col, img, has_tile, conf))
    return results


def debug_tile_detection(cell_images: list[tuple],
                          board_image: np.ndarray,
                          grid_cells: list[tuple],
                          output_path: str = "debug_tiles.jpg",
                          profile: BoardProfile = None):
    """Save debug image with detection results and calibration info."""
    debug = board_image.copy()

    if profile is None:
        profile = calibrate(cell_images)

    detections = detect_tiles_batch(cell_images, profile)

    tile_count = 0
    for det, gc in zip(detections, grid_cells):
        row, col, img, has_tile, conf = det
        _, _, x1, y1, x2, y2 = gc

        color = (0, 255, 0) if has_tile else (0, 0, 255)
        thickness = 2 if has_tile else 1
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)

        if is_bonus_square(row, col):
            sq_type = get_square_type(row, col)
            cv2.putText(debug, sq_type, (x1 + 2, y1 + 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 100, 0), 1)

        cv2.putText(debug, f"{conf:.0%}", (x1 + 2, y2 - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        if has_tile:
            tile_count += 1

    # Mark calibration sample positions
    cell_map = {(r, c): (x1, y1, x2, y2)
                for (r, c, x1, y1, x2, y2) in grid_cells}
    for r, c in LIKELY_TILE_POSITIONS:
        if (r, c) in cell_map:
            x1, y1, _, _ = cell_map[(r, c)]
            cv2.circle(debug, (x1 + 3, y1 + 3), 3, (255, 255, 0), -1)
    for r, c in LIKELY_EMPTY_POSITIONS:
        if (r, c) in cell_map:
            x1, y1, _, _ = cell_map[(r, c)]
            cv2.circle(debug, (x1 + 3, y1 + 3), 3, (255, 0, 255), -1)

    # Calibration info overlay
    if profile.calibrated:
        d = profile.debug_info
        info = (f"Cal: tile_B={d['tile_brightness']} "
                f"empty_B={d['empty_brightness']} "
                f"gap={d['brightness_gap']}")
        y_pos = board_image.shape[0] - 15
        cv2.putText(debug, info, (5, y_pos),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imwrite(output_path, debug)
    print(f"Tile debug: {output_path}")
    print(f"  Tiles: {tile_count}, Empty: {len(detections) - tile_count}")
    if profile.calibrated:
        d = profile.debug_info
        print(f"  Calibration: tile(B={d['tile_brightness']} "
              f"S={d['tile_saturation']}) empty(B={d['empty_brightness']} "
              f"S={d['empty_saturation']}) gap={d['brightness_gap']}")
