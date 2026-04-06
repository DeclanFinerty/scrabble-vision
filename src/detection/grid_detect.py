"""
Board corner detection and perspective correction.

Two modes:
    1. Interactive: User clicks the four corners of the board
    2. Automatic: Contour-based detection (best-effort, may need fallback)

Once corners are known, the board is warped to a perfect square and
divided into a 15x15 equal grid. This works because the physical
board has perfectly equal cell spacing — no grid line detection needed.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


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
    """
    Divide a square image into 15x15 equal cells with padding to
    trim grid line edges. Correct because the board is manufactured
    with perfectly equal cell spacing.
    """
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


class CornerSelector:
    """
    Interactive tool: user clicks the 4 corners of the board.
    Click order: top-left, top-right, bottom-right, bottom-left.
    Press 'r' to reset, Enter to confirm, 'q' to quit.
    """

    def __init__(self, image: np.ndarray):
        self.original = image
        self.corners = []
        self.scale = 1.0
        self.window_name = "Click 4 corners: TL -> TR -> BR -> BL"

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append([x, y])
            self._redraw()

    def _redraw(self):
        display = self._get_display().copy()

        labels = ["TL", "TR", "BR", "BL"]
        for i, (cx, cy) in enumerate(self.corners):
            cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if i > 0:
                cv2.line(display, tuple(self.corners[i - 1]),
                         tuple(self.corners[i]), (0, 255, 0), 2)
        if len(self.corners) == 4:
            cv2.line(display, tuple(self.corners[3]),
                     tuple(self.corners[0]), (0, 255, 0), 2)
            cv2.putText(display, "Press Enter to confirm, R to reset",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        cv2.imshow(self.window_name, display)

    def _get_display(self) -> np.ndarray:
        h, w = self.original.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            self.scale = max_dim / max(h, w)
            return cv2.resize(self.original,
                              (int(w * self.scale), int(h * self.scale)))
        self.scale = 1.0
        return self.original.copy()

    def run(self) -> np.ndarray | None:
        """Open window, let user click 4 corners. Returns 4x2 array or None."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self._get_display())

        print("Click the 4 corners: TL, TR, BR, BL")
        print("  r = reset | Enter = confirm | q = quit")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return None

            if key == ord('r'):
                self.corners = []
                cv2.imshow(self.window_name, self._get_display())
                print("Reset.")

            if key in (13, 10) and len(self.corners) == 4:
                cv2.destroyAllWindows()
                corners = np.array(self.corners, dtype=np.float32)
                if self.scale != 1.0:
                    corners = corners / self.scale
                return corners


def auto_detect_corners(image: np.ndarray) -> np.ndarray | None:
    """Try to automatically find the board's 4 corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))

    return None


def detect_grid(image: np.ndarray, corners: np.ndarray = None,
                interactive: bool = False,
                output_size: int = 900) -> GridCells:
    """
    Full grid detection pipeline.

    Priority: corners arg > interactive > auto-detect > equal fallback.
    """
    method = "equal"
    used_corners = None

    if corners is not None:
        board = perspective_correct(image, corners, output_size)
        used_corners = corners
        method = "manual"

    elif interactive:
        selector = CornerSelector(image)
        clicked = selector.run()
        if clicked is not None:
            board = perspective_correct(image, clicked, output_size)
            used_corners = clicked
            method = "manual"
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


def extract_cell_images(grid: GridCells) -> list[tuple]:
    """Extract cell images. Returns list of (row, col, cell_image)."""
    board = grid.board_image
    results = []
    for (row, col, x1, y1, x2, y2) in grid.cells:
        cell_img = board[y1:y2, x1:x2]
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
    """Save corners for reuse (so you don't re-click every time)."""
    np.save(path, corners)
    print(f"Corners saved: {path}")


def load_corners(path: str) -> np.ndarray:
    """Load previously saved corners."""
    return np.load(path)