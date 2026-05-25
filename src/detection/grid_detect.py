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


def _detect_by_flood(image: np.ndarray) -> np.ndarray | None:
    """Find board by flood-filling from image borders to isolate the board."""
    small, scale = _downscale(image, 500)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    sh, sw = small.shape[:2]

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure borders are background (white) for flood fill
    # Try both polarities: board lighter or darker than background
    for invert in [False, True]:
        mask = cv2.bitwise_not(binary) if invert else binary.copy()

        # Flood fill from all border pixels
        fill_mask = np.zeros((sh + 2, sw + 2), dtype=np.uint8)
        border_pts = (
            [(0, y) for y in range(sh)] +
            [(sw - 1, y) for y in range(sh)] +
            [(x, 0) for x in range(sw)] +
            [(x, sh - 1) for x in range(sw)]
        )
        for x, y in border_pts:
            if mask[y, x] == 255:
                cv2.floodFill(mask, fill_mask, (x, y), 0)

        # Whatever remains is the board interior
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        result = _find_quad_in_contours(contours, scale, image.shape)
        if result is not None:
            return result

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


def _cluster_lines(lines: list, gap: float = 20.0) -> list:
    """Cluster lines by rho proximity, return cluster representatives (median)."""
    if not lines:
        return []
    lines_sorted = sorted(lines, key=lambda l: l[0])
    clusters = [[lines_sorted[0]]]
    for line in lines_sorted[1:]:
        if abs(line[0] - clusters[-1][-1][0]) < gap:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return [min(c, key=lambda l: abs(l[0] - np.median([x[0] for x in c])))
            for c in clusters]


def _detect_by_hough(image: np.ndarray) -> np.ndarray | None:
    """Find board corners via Hough lines on downscaled image."""
    small, scale = _downscale(image, 500)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 30, 100)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
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

    h_clusters = _cluster_lines(horizontal, gap=15)
    v_clusters = _cluster_lines(vertical, gap=15)

    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None

    top = h_clusters[0]
    bottom = h_clusters[-1]
    left = v_clusters[0]
    right = v_clusters[-1]

    tl = _line_intersection(top, left)
    tr = _line_intersection(top, right)
    br = _line_intersection(bottom, right)
    bl = _line_intersection(bottom, left)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32) / scale
    if _validate_corners(corners, image.shape):
        return corners

    return None


def auto_detect_corners(image: np.ndarray) -> np.ndarray | None:
    """Try multiple strategies to find the board's 4 corners."""
    corners = _detect_by_contours(image)
    if corners is not None:
        return corners

    corners = _detect_by_flood(image)
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
    Interactive grid fitting. Two driving modes:
      - run() — keyboard-driven (Enter/R/Q). Used by scan.py --interactive.
      - run_until_action(buttons=...) — click-driven button strip rendered
        below the image. Used by verify_corners.py because cv2 keyboard
        input is unreliable on Linux/Wayland.

    The mouse callback never re-enters cv2.waitKey itself; it just
    updates self.corners and sets a dirty flag. The owning loop reads
    that flag and repaints. This avoids a Qt event-loop reentrancy
    storm during fast drags that was leaving stdin in a broken state.
    """

    GRAB_RADIUS = 20
    CORNER_LABELS = ["TL", "TR", "BR", "BL"]
    BUTTON_HEIGHT = 48
    DEFAULT_MAX_DISPLAY = 800
    DEFAULT_VERIFY_BUTTONS = (
        ("Verify", "verify", (60, 160, 60)),
        ("Rotate", "rotate", (40, 180, 220)),
        ("FlipH",  "flip_h", (180, 100, 180)),
        ("FlipV",  "flip_v", (100, 180, 180)),
        ("Reset",  "reset",  (40, 140, 200)),
        ("Skip",   "skip",   (140, 140, 140)),
        ("Reject", "reject", (50, 50, 200)),
        ("Quit",   "quit",   (40, 40, 100)),
    )

    def __init__(self, image: np.ndarray,
                 initial_corners: np.ndarray = None,
                 max_display: int = DEFAULT_MAX_DISPLAY):
        self._input_image = image     # never modified -- used by Reset
        self.original = image         # working image; Flip H/V mutates this
        self.max_display = max_display
        self.scale = 1.0
        self.display_base = self._compute_display()
        self.dragging = -1
        self._opened = False
        self._dirty = True
        self._buttons: tuple = ()  # (label, action, bgr) tuples
        self._title_text = ""
        self._action: str | None = None
        # Flip parity vs the un-modified input image. Two flips of
        # the same axis cancel; the caller saves only when at least
        # one is True at verify time.
        self._h_flipped = False
        self._v_flipped = False

        if initial_corners is not None:
            # Trust the caller's order. Re-running order_corners()
            # here would discard any rotation the user previously
            # applied (and saved to corners.json) by re-sorting the
            # array back into image-position order. Callers that
            # need normalisation (auto_detect_corners output, COCO
            # bbox import) should run order_corners themselves
            # before passing in.
            self.corners = np.asarray(initial_corners,
                                      dtype=np.float32).copy()
        else:
            h, w = image.shape[:2]
            self.corners = np.array([
                [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
            ], dtype=np.float32)

        # Clip seed corners to image bounds so handles can never start
        # off-screen (some COCO bbox seeds land just outside).
        h, w = image.shape[:2]
        self.corners[:, 0] = np.clip(self.corners[:, 0], 0, w - 1)
        self.corners[:, 1] = np.clip(self.corners[:, 1], 0, h - 1)

        self.auto_corners = self.corners.copy()
        # ASCII only: an em-dash in the Qt window name makes namedWindow
        # silently fail to register on Linux/XCB, breaking setMouseCallback.
        self.window_name = "Grid Fit - drag corners"

    def _compute_display(self) -> np.ndarray:
        h, w = self.original.shape[:2]
        if max(h, w) > self.max_display:
            self.scale = self.max_display / max(h, w)
            return cv2.resize(
                self.original,
                (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        self.scale = 1.0
        return self.original.copy()

    def _scaled_corners(self) -> np.ndarray:
        return (self.corners * self.scale).astype(np.int32)

    def _render_canvas(self) -> np.ndarray:
        display = self.display_base.copy()
        scaled = self._scaled_corners()

        for (p1, p2) in _grid_overlay_lines(self.corners):
            sp1 = (int(p1[0] * self.scale), int(p1[1] * self.scale))
            sp2 = (int(p2[0] * self.scale), int(p2[1] * self.scale))
            cv2.line(display, sp1, sp2, (0, 255, 0), 1)

        for i, (cx, cy) in enumerate(scaled):
            color = (0, 255, 255) if i == self.dragging else (0, 255, 0)
            cv2.circle(display, (cx, cy), 10, color, -1)
            cv2.circle(display, (cx, cy), 10, (0, 0, 0), 2)
            cv2.putText(display, self.CORNER_LABELS[i], (cx + 14, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        title = self._title_text or "Drag corners"
        cv2.putText(display, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(display, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if not self._buttons:
            return display

        img_h, img_w = display.shape[:2]
        strip = np.full((self.BUTTON_HEIGHT, img_w, 3), 30, dtype=np.uint8)
        btn_w = img_w // len(self._buttons)
        for i, (label, _action, color) in enumerate(self._buttons):
            x1 = i * btn_w
            x2 = (i + 1) * btn_w if i < len(self._buttons) - 1 else img_w
            cv2.rectangle(strip, (x1 + 2, 4),
                          (x2 - 2, self.BUTTON_HEIGHT - 4), color, -1)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = x1 + (btn_w - text_size[0]) // 2
            ty = (self.BUTTON_HEIGHT + text_size[1]) // 2
            cv2.putText(strip, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return np.vstack([display, strip])

    def _button_at(self, x: int, y: int) -> str | None:
        if not self._buttons:
            return None
        img_h, img_w = self.display_base.shape[:2]
        if y < img_h or y >= img_h + self.BUTTON_HEIGHT:
            return None
        btn_w = img_w // len(self._buttons)
        idx = min(x // btn_w, len(self._buttons) - 1)
        if idx < 0:
            return None
        return self._buttons[idx][1]

    def _find_nearest_corner(self, x: int, y: int) -> int:
        scaled = self._scaled_corners()
        dists = np.sqrt(np.sum((scaled - np.array([x, y])) ** 2, axis=1))
        nearest = int(np.argmin(dists))
        if dists[nearest] <= self.GRAB_RADIUS:
            return nearest
        return -1

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            action = self._button_at(x, y)
            if action is not None:
                self._action = action
                return
            idx = self._find_nearest_corner(x, y)
            if idx >= 0:
                self.dragging = idx

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging >= 0:
            # Clamp mouse coords to the image region so a corner can
            # never be dragged off the visible area (would otherwise be
            # ungrabbable on the next pass).
            img_h, img_w = self.display_base.shape[:2]
            cx = max(0, min(img_w - 1, x))
            cy = max(0, min(img_h - 1, y))
            self.corners[self.dragging] = np.array(
                [cx / self.scale, cy / self.scale], dtype=np.float32
            )
            self._dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = -1
            self._dirty = True

    # Enter spans ASCII CR/LF and Qt::Key_Return / Qt::Key_Enter, which
    # OpenCV's Qt backend on Linux sometimes returns as the raw Qt code
    # (where & 0xFF would mask down to 4 or 5 and miss the comparison).
    ENTER_KEYS = (13, 10, 0x01000004, 0x01000005)

    def _repaint_if_dirty(self) -> None:
        if self._dirty:
            cv2.imshow(self.window_name, self._render_canvas())
            self._dirty = False

    def open(self) -> None:
        if self._opened:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._dirty = True
        self._repaint_if_dirty()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        # Qt can have already torn the window down (e.g. user closed the
        # X window manually, or a previous error path ran). Don't let
        # that mask the real exception we're cleaning up from.
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        finally:
            self._opened = False
            self._buttons = ()
            self._title_text = ""

    def pump_events(self, timeout_ms: int = 30) -> int:
        self._repaint_if_dirty()
        return cv2.waitKey(timeout_ms)

    def reset_to_auto(self) -> None:
        self.corners = self.auto_corners.copy()
        self._dirty = True

    def current_corners(self) -> np.ndarray:
        return self.corners.copy()

    # ── Click-driven driver (used by verify_corners.py) ──

    def run_until_action(self, buttons=None, title: str = "") -> tuple[str, np.ndarray]:
        """Show the window with a clickable button strip below the image.
        Returns (action, corners) when the user clicks any button other
        than Reset. Reset rebases corners to the seed in-place and keeps
        the window open."""
        self._buttons = tuple(buttons) if buttons is not None \
            else self.DEFAULT_VERIFY_BUTTONS
        self._title_text = title
        self._action = None
        self._dirty = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._opened = True

        try:
            while True:
                self._repaint_if_dirty()
                cv2.waitKey(30)
                if self._action == "reset":
                    # Restore everything to the unmodified input.
                    self.original = self._input_image
                    self._h_flipped = False
                    self._v_flipped = False
                    self.display_base = self._compute_display()
                    self.corners = self.auto_corners.copy()
                    self._dirty = True
                    self._action = None
                    continue
                if self._action == "rotate":
                    # Cycle labels one slot clockwise: the point that
                    # had label TL is relabeled TR, the BL point
                    # rotates in to become the new TL, etc. Repeat
                    # 1-3 times to reach any of the 4 orientations.
                    self.corners = self.corners[[3, 0, 1, 2]].copy()
                    self._dirty = True
                    self._action = None
                    continue
                if self._action == "flip_h":
                    self._apply_flip_h()
                    self._action = None
                    continue
                if self._action == "flip_v":
                    self._apply_flip_v()
                    self._action = None
                    continue
                if self._action is not None:
                    return self._action, self.corners.copy()
        finally:
            self.close()

    # ── Flip helpers ──

    def _apply_flip_h(self) -> None:
        """Mirror image left/right; swap horizontal label pairs so the
        array still reads [TL, TR, BR, BL] for the flipped image."""
        h, w = self.original.shape[:2]
        self.original = self.original[:, ::-1].copy()
        new_corners = self.corners.copy()
        new_corners[:, 0] = (w - 1) - new_corners[:, 0]
        # In a horizontally-flipped image, the old TL pixel sits at
        # the new TR position; same for BR <-> BL.
        self.corners = new_corners[[1, 0, 3, 2]].copy()
        self._h_flipped = not self._h_flipped
        self.display_base = self._compute_display()
        self._dirty = True

    def _apply_flip_v(self) -> None:
        h, w = self.original.shape[:2]
        self.original = self.original[::-1, :].copy()
        new_corners = self.corners.copy()
        new_corners[:, 1] = (h - 1) - new_corners[:, 1]
        # Vertical flip swaps TL<->BL and TR<->BR.
        self.corners = new_corners[[3, 2, 1, 0]].copy()
        self._v_flipped = not self._v_flipped
        self.display_base = self._compute_display()
        self._dirty = True

    @property
    def image_modified(self) -> bool:
        return self._h_flipped or self._v_flipped

    @property
    def current_image(self) -> np.ndarray:
        return self.original

    # ── Key-driven driver (used by scan.py --interactive) ──

    def run(self) -> np.ndarray | None:
        """Open window, return confirmed corners (4x2 float32) or None."""
        # namedWindow MUST precede setMouseCallback — Qt registers the
        # window handle here and setMouseCallback looks it up by name.
        self._buttons = ()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._dirty = True
        self._opened = True

        print("Grid overlay shown. Drag corners to adjust.")
        print("  Enter (or Y) = confirm | R = reset | Q = quit")

        try:
            while True:
                self._repaint_if_dirty()
                key = cv2.waitKey(30)
                if key == -1:
                    continue
                masked = key & 0xFF
                if masked == ord('q'):
                    return None
                if masked == ord('r'):
                    self.corners = self.auto_corners.copy()
                    self._dirty = True
                    print("Reset to auto-detected corners.")
                    continue
                if key in self.ENTER_KEYS or masked in (13, 10) or masked == ord('y'):
                    return self.corners.copy()
                if 0 <= masked < 256:
                    print(f"  (ignored key: raw={key}, masked={masked})")
        finally:
            self.close()


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
