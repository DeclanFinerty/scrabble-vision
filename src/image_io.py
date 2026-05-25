"""
Image loading helpers for board photos.

Centralises HEIC handling so the rest of the codebase can treat every
supported board image the same way. `cv2.imread` cannot read HEIC, so
we go through Pillow + pillow-heif for those and convert to BGR
in-memory; no on-disk conversion happens.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

BOARD_IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".heic", ".heif")


def is_board_image(path: Path) -> bool:
    return path.suffix.lower() in BOARD_IMAGE_EXTS


def load_board_image(path: Path) -> np.ndarray | None:
    """Read a board image as a BGR uint8 ndarray, or None on failure.

    Falls back to Pillow (which handles HEIC via pillow-heif) when
    OpenCV can't decode the file -- HEIC being the obvious case.
    """
    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        return _load_via_pillow(path)

    img = cv2.imread(str(path))
    if img is not None:
        return img
    return _load_via_pillow(path)


def _load_via_pillow(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            rgb = np.array(pil_img)
    except Exception:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_board_image(path: Path, image_bgr: np.ndarray) -> None:
    """Write a BGR uint8 image back to disk, preserving the file's
    extension. JPEG/PNG go through cv2.imwrite; HEIC goes through
    Pillow + pillow-heif (which registers a heif writer alongside
    the opener)."""
    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(path))
        return
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")
