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


def _validate_corners(corners: np.ndarray, image_shape: tuple) -> bool:
    """Check that detected corners form a reasonable board quadrilateral."""
    h, w = image_shape[:2]
    image_area = h * w

    ordered = order_corners(corners)

    # All corners must be within image bounds (with small margin)
    margin = -20
    if np.any(ordered[:, 0] < margin) or np.any(ordered[:, 0] > w - margin):
        return False
    if np.any(ordered[:, 1] < margin) or np.any(ordered[:, 1] > h - margin):
        return False

    # Area must be at least 15% of image
    area = cv2.contourArea(ordered)
    if area < 0.15 * image_area:
        return False

    # Aspect ratio roughly square
    rect = cv2.minAreaRect(ordered)
    (rw, rh) = rect[1]
    if rw == 0 or rh == 0:
        return False
    ratio = max(rw, rh) / min(rw, rh)
    if ratio > 1.55:
        return False

    # Must be convex
    if not cv2.isContourConvex(ordered.astype(np.int32)):
        return False

    return True


def _detect_by_contours(image: np.ndarray) -> np.ndarray | None:
    """Find board corners via adaptive thresholding and contour approximation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 2
    )
    thresh = cv2.bitwise_not(thresh)

    kernel_size = max(15, min(image.shape[:2]) // 60)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        for eps in np.arange(0.01, 0.06, 0.005):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                corners = order_corners(approx.reshape(4, 2))
                if _validate_corners(corners, image.shape):
                    return corners

    # Fallback: minAreaRect on largest contour
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    corners = order_corners(box)
    if _validate_corners(corners, image.shape):
        return corners

    return None


def _line_intersection(line1, line2):
    """Compute intersection of two lines given as (rho, theta) pairs."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b2 * rho1 - b1 * rho2) / det
    y = (a1 * rho2 - a2 * rho1) / det
    return np.array([x, y], dtype=np.float32)


def _detect_by_hough(image: np.ndarray) -> np.ndarray | None:
    """Find board corners via Hough line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None or len(lines) < 4:
        return None

    horizontal = []
    vertical = []
    for line in lines:
        rho, theta = line[0]
        if abs(theta - np.pi / 2) < np.pi / 6:
            horizontal.append((rho, theta))
        elif abs(theta) < np.pi / 6 or abs(theta - np.pi) < np.pi / 6:
            vertical.append((rho, theta))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # Sort by rho to find outermost lines
    horizontal.sort(key=lambda l: l[0])
    vertical.sort(key=lambda l: l[0])

    top = horizontal[0]
    bottom = horizontal[-1]
    left = vertical[0]
    right = vertical[-1]

    tl = _line_intersection(top, left)
    tr = _line_intersection(top, right)
    br = _line_intersection(bottom, right)
    bl = _line_intersection(bottom, left)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    if _validate_corners(corners, image.shape):
        return corners

    return None


def auto_detect_corners(image: np.ndarray) -> np.ndarray | None:
    """Try multiple strategies to find the board's 4 corners."""
    corners = _detect_by_contours(image)
    if corners is not None:
        return corners

    corners = _detect_by_hough(image)
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
