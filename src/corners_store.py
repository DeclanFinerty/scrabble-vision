"""
Read/write helper for data/annotations/corners.json.

Each entry is keyed by image filename (basename only — names are
unique across data/raw/boards/) and holds:
  - corners: 4 (x, y) points ordered TL/TR/BR/BL, or None
  - corners_verified: True once a human has confirmed the corners
  - corners_rejected: True if a human gave up on these corners
                      (board won't appear in either labeling stage)
  - tile_label_skipped: True if a human chose not to label tiles
                        for this board (corners can still be useful)

Older entries in corners.json may be bare arrays (or null) from before
the verification flags existed. `load()` migrates them in memory.
"""

import json
from pathlib import Path
from typing import TypedDict


CORNERS_FILE = Path("data/annotations/corners.json")


class CornerEntry(TypedDict):
    corners: list[list[float]] | None
    corners_verified: bool
    corners_rejected: bool
    tile_label_skipped: bool


def empty_entry() -> CornerEntry:
    return {
        "corners": None,
        "corners_verified": False,
        "corners_rejected": False,
        "tile_label_skipped": False,
    }


def _normalize(value) -> CornerEntry:
    if value is None or isinstance(value, list):
        return {
            "corners": value,
            "corners_verified": False,
            "corners_rejected": False,
            "tile_label_skipped": False,
        }
    return {
        "corners": value.get("corners"),
        "corners_verified": bool(value.get("corners_verified", False)),
        "corners_rejected": bool(value.get("corners_rejected", False)),
        "tile_label_skipped": bool(value.get("tile_label_skipped", False)),
    }


def load() -> dict[str, CornerEntry]:
    if not CORNERS_FILE.exists():
        return {}
    raw = json.loads(CORNERS_FILE.read_text())
    return {k: _normalize(v) for k, v in raw.items()}


def save(data: dict[str, CornerEntry]) -> None:
    CORNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CORNERS_FILE.write_text(json.dumps(data, indent=2, sort_keys=True))
