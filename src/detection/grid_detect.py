"""
Board corner detection and perspective correction.

Auto-detects board corners using contour and line-based methods,
then lets the user fine-tune via draggable corner handles with
a live 15x15 grid overlay.
"""

import cv2
import numpy as np
from dataclasses import dataclass


GRID_SIZE = 15


@dataclass
class GridCells:
    """Result of grid detection."""
    cells: list              # (row, col, x1, y1, x2, y2) tuples
    board_image: np.ndarray  # perspective-corrected square board image
    corners: np.ndarray      # 4 corners used (original image coords)
    method: str              # 'manual', 'auto', or 'equal'


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def perspective_correct(image: np.ndarray, corners: np.ndarray,
                        output_size: int = 900) -> np.ndarray:
    """Warp the board to a perfect square given 4 corner points."""
    ordered = order_corners(corners)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (output_size, output_size))


def equal_grid(img_size: int, padding_pct: float = 0.05) -> list[tuple]:
    """Divide a square image into 15x15 equal cells with padding to trim grid line edges."""
    cell_size = img_size / GRID_SIZE
    pad = int(cell_size * padding_pct)

    cells = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1 = int(row * cell_size) + pad
            y2 = int((row + 1) * cell_size) - pad
            x1 = int(col * cell_size) + pad
            x2 = int((col + 1) * cell_size) - pad
            cells.append((row, col, x1, y1, x2, y2))

    return cells


# ── Auto-detection ──────────────────────────────────────────────


def _validate_corners(corners: np.ndarray, image_shape: tuple,
                      max_area_frac: float = 0.95) -> bool:
    """Check that detected corners form a reasonable board quadrilateral."""
    h, w = image_shape[:2]
    image_area = h * w

    ordered = order_corners(corners)

    margin = -20
    if np.any(ordered[:, 0] < margin) or np.any(ordered[:, 0] > w - margin):
        return False
    if np.any(ordered[:, 1] < margin) or np.any(ordered[:, 1] > h - margin):
        return False

    area = cv2.contourArea(ordered)
    if area < 0.15 * image_area:
        return False
    if area > max_area_frac * image_area:
        return False

    rect = cv2.minAreaRect(ordered)
    (rw, rh) = rect[1]
    if rw == 0 or rh == 0:
        return False
    ratio = max(rw, rh) / min(rw, rh)
    if ratio > 1.55:
        return False

    if not cv2.isContourConvex(ordered.astype(np.int32)):
        return False

    return True


def _downscale(image: np.ndarray, target: int = 500) -> tuple:
    """Downscale image so longest side = target. Returns (small, scale)."""
    h, w = image.shape[:2]
    scale = target / max(h, w)
    if scale >= 1.0:
        return image, 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def _find_quad_in_contours(contours, scale: float,
                           image_shape: tuple) -> np.ndarray | None:
    """Search contours for a valid quadrilateral, return corners or None."""
    for cnt in contours[:10]:
        for eps in np.arange(0.01, 0.06, 0.005):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                corners = order_corners(approx.reshape(4, 2)) / scale
                if _validate_corners(corners, image_shape):
                    return corners

    for cnt in contours[:5]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        corners = order_corners(box) / scale
        if _validate_corners(corners, image_shape):
            return corners

    return None


def _detect_by_contours(image: np.ndarray) -> np.ndarray | None:
    """Find board corners via downscale + blur + Canny + contour.

    Tries multiple blur/threshold combos from light to heavy so we catch
    both subtle edges (board on gray surface) and high-contrast ones.
    """
    small, scale = _downscale(image, 500)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    params = [
        # (blur_k, canny_lo, canny_hi, close_k, close_iter)
        (9,  20, 80,  5, 2),   # light blur — catches subtle edges
        (15, 25, 90,  7, 2),   # medium
        (21, 30, 100, 7, 2),   # heavy — original, good for high-contrast
    ]

    for blur_k, clo, chi, ck, ci in params:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, clo, chi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=ci)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        result = _find_quad_in_contours(contours, scale, image.shape)
        if result is not None:
            return result

    return None


def _seg_angle(seg):
    """Angle of a HoughLinesP segment in radians, normalized to (-pi/2, pi/2]."""
    x1, y1, x2, y2 = seg
    angle = np.arctan2(y2 - y1, x2 - x1)
    # Normalize so that lines in opposite directions map to same angle
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle <= -np.pi / 2:
        angle += np.pi
    return angle


def _seg_perp_dist(seg, axis_angle):
    """Perpendicular distance from origin along axis normal."""
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    normal = axis_angle + np.pi / 2
    return mx * np.cos(normal) + my * np.sin(normal)


def _cluster_by_distance(values, gap):
    """Cluster sorted scalar values by proximity gap.

    Returns list of clusters, each a list of original indices.
    """
    if len(values) == 0:
        return []
    order = np.argsort(values)
    sorted_vals = values[order]
    clusters = [[order[0]]]
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[i - 1] > gap:
            clusters.append([])
        clusters[-1].append(order[i])
    return clusters


def _grid_inlier_score(positions, pitch, offset):
    """Count how many positions align to a grid defined by pitch + offset."""
    if pitch <= 0:
        return 0
    residuals = np.mod(positions - offset, pitch)
    # Distance to nearest grid line
    dist_to_grid = np.minimum(residuals, pitch - residuals)
    tol = pitch * 0.15
    return int(np.sum(dist_to_grid < tol))


def _find_cell_pitch(positions, min_dim):
    """Find dominant cell pitch from line positions.

    Tests candidate pitches and picks the one that aligns the most
    observed positions to a regular grid. Candidates come from the
    expected range for a 15-cell board.
    """
    positions = np.sort(positions)
    if len(positions) < 3:
        return None

    span = positions[-1] - positions[0]
    if span < 1:
        return None

    # Expected pitch: board is 15 cells, spanning 60-100% of image dimension
    lo = min_dim / 22
    hi = min_dim / 10
    candidates = np.arange(lo, hi, 0.5)

    best_pitch = None
    best_score = 0

    for p in candidates:
        # Try several offsets for this pitch
        for pos in positions[:min(8, len(positions))]:
            offset = np.mod(pos, p)
            score = _grid_inlier_score(positions, p, offset)
            if score > best_score:
                best_score = score
                best_pitch = p

    if best_pitch is None or best_score < 4:
        return None

    # Refine: median of gaps that are close to the best pitch
    all_gaps = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = abs(positions[j] - positions[i])
            n = round(d / best_pitch)
            if n >= 1 and abs(d / n - best_pitch) < best_pitch * 0.15:
                all_gaps.append(d / n)

    return float(np.median(all_gaps)) if all_gaps else float(best_pitch)


def _fit_grid_lines(positions, pitch, n_lines=16, bounds=None):
    """Fit n_lines evenly-spaced lines to observed positions.

    Always produces exactly n_lines spanning (n_lines-1)*pitch.
    Finds the phase offset that aligns best with observations,
    then centers the grid. If bounds=(lo, hi) is given, constrains
    the grid to stay within that range (with half-cell margin).
    """
    if pitch <= 0:
        return None

    # Find best offset by testing each observed position as anchor
    best_offset = 0
    best_score = 0
    for pos in positions:
        offset = np.mod(pos, pitch)
        score = _grid_inlier_score(positions, pitch, offset)
        if score > best_score:
            best_score = score
            best_offset = offset

    # Center the grid on the observed positions
    obs_center = (positions.min() + positions.max()) / 2
    center_idx = round((obs_center - best_offset) / pitch)
    first_line = best_offset + (center_idx - (n_lines - 1) / 2) * pitch

    fitted = np.array([first_line + i * pitch for i in range(n_lines)])

    # Constrain to bounds if provided (no margin — grid must be inside image)
    if bounds is not None:
        lo, hi = bounds
        if fitted[-1] > hi:
            fitted -= (fitted[-1] - hi)
        if fitted[0] < lo:
            fitted += (lo - fitted[0])

    return fitted


def _detect_by_grid_lines(image: np.ndarray) -> np.ndarray | None:
    """Detect board corners by finding the internal 15x15 grid lines.

    Uses HoughLinesP to find long line segments, clusters them into
    horizontal and vertical families, determines cell pitch from spacing,
    fits a regular 16-line grid in each direction, and returns the
    outermost intersections as corners.
    """
    small, scale = _downscale(image, 800)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = small.shape[:2]
    min_dim = min(sh, sw)

    # Try multiple Canny threshold ranges
    median_val = float(np.median(gray))
    canny_params = [
        (int(max(0, 0.66 * median_val)), int(min(255, 1.33 * median_val))),
        (30, 100),   # low — catches subtle grid lines
        (50, 150),   # medium
    ]

    for canny_lo, canny_hi in canny_params:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_lo, canny_hi)

        min_len = int(0.15 * min_dim)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=min_len, maxLineGap=20)
        if lines is None:
            continue

        segments = lines.reshape(-1, 4)
        if len(segments) < 8:
            continue

        result = _fit_grid_from_segments(segments, sh, sw, scale, image.shape)
        if result is not None:
            return result

    return None


def _fit_grid_from_segments(segments, sh, sw, scale, image_shape):
    """Given HoughLinesP segments, try to fit a 15x15 grid and return corners."""
    min_dim = min(sh, sw)
    angles = np.array([_seg_angle(s) for s in segments])

    h_mask = np.abs(angles) < np.radians(15)
    v_mask = np.abs(angles - np.pi / 2) < np.radians(15)
    # Also catch negative-side vertical (angles near -90° mapped to +90° by normalize)
    # _seg_angle already normalizes to (-pi/2, pi/2] so vertical = near pi/2

    h_segs = segments[h_mask]
    v_segs = segments[v_mask]

    if len(h_segs) < 3 or len(v_segs) < 3:
        return None

    h_angle = float(np.median(angles[h_mask]))
    v_angle = float(np.median(angles[v_mask]))

    h_dists = np.array([_seg_perp_dist(s, h_angle) for s in h_segs])
    v_dists = np.array([_seg_perp_dist(s, v_angle) for s in v_segs])

    cluster_gap = min_dim * 0.02
    h_clusters = _cluster_by_distance(h_dists, cluster_gap)
    v_clusters = _cluster_by_distance(v_dists, cluster_gap)

    if len(h_clusters) < 3 or len(v_clusters) < 3:
        return None

    h_positions = np.array([np.median(h_dists[c]) for c in h_clusters])
    v_positions = np.array([np.median(v_dists[c]) for c in v_clusters])

    h_pitch = _find_cell_pitch(h_positions, min_dim)
    v_pitch = _find_cell_pitch(v_positions, min_dim)
    if h_pitch is None or v_pitch is None:
        return None

    # Sanity: h and v pitch should be similar (square cells)
    if max(h_pitch, v_pitch) / min(h_pitch, v_pitch) > 1.4:
        return None

    pitch = (h_pitch + v_pitch) / 2

    # Compute perp-distance bounds from image corners
    img_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    h_corner_dists = [_seg_perp_dist([c[0], c[1], c[0], c[1]], h_angle) for c in img_corners]
    v_corner_dists = [_seg_perp_dist([c[0], c[1], c[0], c[1]], v_angle) for c in img_corners]
    h_bounds = (min(h_corner_dists), max(h_corner_dists))
    v_bounds = (min(v_corner_dists), max(v_corner_dists))

    h_fitted = _fit_grid_lines(h_positions, pitch, n_lines=16, bounds=h_bounds)
    v_fitted = _fit_grid_lines(v_positions, pitch, n_lines=16, bounds=v_bounds)
    if h_fitted is None or v_fitted is None:
        return None

    h_normal = h_angle + np.pi / 2
    v_normal = v_angle + np.pi / 2

    def intersect_hv(d_h, d_v):
        hx, hy = d_h * np.cos(h_normal), d_h * np.sin(h_normal)
        vx, vy = d_v * np.cos(v_normal), d_v * np.sin(v_normal)
        hd = np.array([np.cos(h_angle), np.sin(h_angle)])
        vd = np.array([np.cos(v_angle), np.sin(v_angle)])
        det = hd[0] * (-vd[1]) - (-vd[0]) * hd[1]
        if abs(det) < 1e-6:
            return None
        t = ((-vd[1]) * (vx - hx) - (-vd[0]) * (vy - hy)) / det
        return np.array([hx + t * hd[0], hy + t * hd[1]], dtype=np.float32)

    tl = intersect_hv(h_fitted[0], v_fitted[0])
    tr = intersect_hv(h_fitted[0], v_fitted[-1])
    br = intersect_hv(h_fitted[-1], v_fitted[-1])
    bl = intersect_hv(h_fitted[-1], v_fitted[0])

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32) / scale
    if _validate_corners(corners, image_shape, max_area_frac=0.98):
        return corners

    return None


def auto_detect_corners(image: np.ndarray) -> np.ndarray | None:
    """Try grid-line detection first, fall back to contour method."""
    corners = _detect_by_grid_lines(image)
    if corners is not None:
        return corners

    corners = _detect_by_contours(image)
    if corners is not None:
        return corners

    return None


# ── Grid overlay computation ────────────────────────────────────


def _grid_overlay_lines(corners: np.ndarray, output_size: int = 900) -> list:
    """Compute 15x15 grid lines in original image coordinates."""
    ordered = order_corners(corners)
    src = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, ordered)
    cell = output_size / GRID_SIZE
    lines = []

    for i in range(GRID_SIZE + 1):
        pos = i * cell
        # Horizontal line
        pts = np.array([[[0, pos], [output_size - 1, pos]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, M)[0]
        lines.append((tuple(transformed[0].astype(int)),
                       tuple(transformed[1].astype(int))))
        # Vertical line
        pts = np.array([[[pos, 0], [pos, output_size - 1]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, M)[0]
        lines.append((tuple(transformed[0].astype(int)),
                       tuple(transformed[1].astype(int))))

    return lines


# ── Interactive grid-fit UI ─────────────────────────────────────


class GridFitUI:
    """
    Interactive grid fitting: shows auto-detected grid overlay with
    draggable corners. Press Enter to confirm, R to reset, Q to quit.
    """

    GRAB_RADIUS = 20
    CORNER_LABELS = ["TL", "TR", "BR", "BL"]

    def __init__(self, image: np.ndarray, initial_corners: np.ndarray = None):
        self.original = image
        self.scale = 1.0
        self.display_base = self._compute_display()
        self.dragging = -1

        if initial_corners is not None:
            self.corners = order_corners(initial_corners).copy()
        else:
            h, w = image.shape[:2]
            self.corners = np.array([
                [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
            ], dtype=np.float32)

        self.auto_corners = self.corners.copy()
        self.window_name = "Grid Fit — drag corners, Enter=confirm, R=reset, Q=quit"

    def _compute_display(self) -> np.ndarray:
        h, w = self.original.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            self.scale = max_dim / max(h, w)
            return cv2.resize(self.original,
                              (int(w * self.scale), int(h * self.scale)))
        self.scale = 1.0
        return self.original.copy()

    def _scaled_corners(self) -> np.ndarray:
        return (self.corners * self.scale).astype(np.int32)

    def _redraw(self):
        display = self.display_base.copy()
        scaled = self._scaled_corners()

        # Grid lines
        overlay_corners = self.corners.copy()
        lines = _grid_overlay_lines(overlay_corners)
        for (p1, p2) in lines:
            sp1 = (int(p1[0] * self.scale), int(p1[1] * self.scale))
            sp2 = (int(p2[0] * self.scale), int(p2[1] * self.scale))
            cv2.line(display, sp1, sp2, (0, 255, 0), 1)

        # Corner handles
        for i, (cx, cy) in enumerate(scaled):
            color = (0, 255, 255) if i == self.dragging else (0, 255, 0)
            cv2.circle(display, (cx, cy), 10, color, -1)
            cv2.circle(display, (cx, cy), 10, (0, 0, 0), 2)
            cv2.putText(display, self.CORNER_LABELS[i], (cx + 14, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status
        cv2.putText(display, "Drag corners | Enter=confirm | R=reset | Q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.window_name, display)

    def _find_nearest_corner(self, x: int, y: int) -> int:
        scaled = self._scaled_corners()
        dists = np.sqrt(np.sum((scaled - np.array([x, y])) ** 2, axis=1))
        nearest = int(np.argmin(dists))
        if dists[nearest] <= self.GRAB_RADIUS:
            return nearest
        return -1

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_nearest_corner(x, y)
            if idx >= 0:
                self.dragging = idx

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging >= 0:
            self.corners[self.dragging] = np.array(
                [x / self.scale, y / self.scale], dtype=np.float32
            )
            self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = -1
            self._redraw()

    def run(self) -> np.ndarray | None:
        """Open window, return confirmed corners (4x2 float32) or None."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()

        print("Grid overlay shown. Drag corners to adjust.")
        print("  Enter = confirm | R = reset | Q = quit")

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return None

            if key == ord('r'):
                self.corners = self.auto_corners.copy()
                self._redraw()
                print("Reset to auto-detected corners.")

            if key in (13, 10):
                cv2.destroyAllWindows()
                return self.corners.copy()


# ── Main pipeline entry ─────────────────────────────────────────


def detect_grid(image: np.ndarray, corners: np.ndarray = None,
                interactive: bool = False,
                output_size: int = 900) -> GridCells:
    """
    Full grid detection pipeline.
    Priority: corners arg > interactive (auto + UI) > auto-detect > equal fallback.
    """
    method = "equal"
    used_corners = None

    if corners is not None:
        board = perspective_correct(image, corners, output_size)
        used_corners = corners
        method = "manual"

    elif interactive:
        auto = auto_detect_corners(image)
        ui = GridFitUI(image, initial_corners=auto)
        clicked = ui.run()
        if clicked is not None:
            board = perspective_correct(image, clicked, output_size)
            used_corners = clicked
            method = "auto" if auto is not None else "manual"
        else:
            h, w = image.shape[:2]
            size = min(h, w)
            board = cv2.resize(image[:size, :size],
                               (output_size, output_size))
            used_corners = np.array([[0, 0], [size, 0],
                                      [size, size], [0, size]],
                                    dtype=np.float32)
    else:
        auto = auto_detect_corners(image)
        if auto is not None:
            board = perspective_correct(image, auto, output_size)
            used_corners = auto
            method = "auto"
        else:
            h, w = image.shape[:2]
            size = min(h, w)
            board = cv2.resize(image[:size, :size],
                               (output_size, output_size))
            used_corners = np.array([[0, 0], [size, 0],
                                      [size, size], [0, size]],
                                    dtype=np.float32)

    cells = equal_grid(output_size)
    return GridCells(cells=cells, board_image=board,
                     corners=used_corners, method=method)


def extract_cell_images(grid: GridCells, buffer_pct: float = 0.05) -> list[tuple]:
    """Extract cell images with a small buffer to catch off-center tiles."""
    board = grid.board_image
    h, w = board.shape[:2]
    cell_size = w / GRID_SIZE
    buf = int(cell_size * buffer_pct)
    results = []
    for (row, col, x1, y1, x2, y2) in grid.cells:
        bx1 = max(0, x1 - buf)
        by1 = max(0, y1 - buf)
        bx2 = min(w, x2 + buf)
        by2 = min(h, y2 + buf)
        cell_img = board[by1:by2, bx1:bx2]
        if cell_img.size > 0:
            results.append((row, col, cell_img))
    return results


def debug_grid(image: np.ndarray, grid: GridCells,
               output_path: str = "debug_grid.jpg"):
    """Save debug image with grid overlay."""
    debug = grid.board_image.copy()
    for (row, col, x1, y1, x2, y2) in grid.cells:
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(debug, f"{row},{col}", (x1 + 2, y1 + 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(output_path, debug)
    print(f"Grid debug: {output_path} (method: {grid.method})")


def save_corners(corners: np.ndarray, path: str):
    """Save corners for reuse."""
    np.save(path, corners)
    print(f"Corners saved: {path}")


def load_corners(path: str) -> np.ndarray:
    """Load previously saved corners."""
    return np.load(path)
