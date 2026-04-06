"""
FastAPI backend for the Scrabble board scanner.

Run with: uv run uvicorn server:app --host 0.0.0.0 --port 8000
Open http://<your-ip>:8000 on your phone.
"""

import cv2
import numpy as np
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

from src.detection.grid_detect import (
    detect_grid, extract_cell_images, auto_detect_corners, order_corners,
    GRID_SIZE
)
from src.detection.tile_detect import detect_tile_presence
from src.classification.model import load_model, predict_tiles

app = FastAPI()

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model("models/tile_classifier.pt", device="cpu")
    return _model


def _decode_image(contents: bytes) -> np.ndarray | None:
    arr = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def scan_image(image: np.ndarray, corners: np.ndarray = None,
               confidence_threshold: float = 0.7) -> dict:
    """Run the full scan pipeline on an in-memory image."""
    timings = {}

    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=False)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    cell_images = extract_cell_images(grid)

    t0 = time.perf_counter()
    tile_cells = []
    empty_cells = []
    for row, col, cell_img in cell_images:
        has_tile, _ = detect_tile_presence(cell_img, row, col)
        if has_tile:
            tile_cells.append((row, col, cell_img))
        else:
            empty_cells.append((row, col))
    timings["tiles_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    uncertain = []

    if tile_cells:
        model = get_model()
        tile_imgs = [img for _, _, img in tile_cells]
        predictions = predict_tiles(model, tile_imgs, device="cpu")

        for (row, col, _), (label, conf) in zip(tile_cells, predictions):
            if label == "EMPTY":
                board[row][col] = "?"
                confidence[row][col] = conf
                uncertain.append([row, col])
            elif label == "BLANK":
                board[row][col] = "_"
                confidence[row][col] = conf
            else:
                board[row][col] = label
                confidence[row][col] = conf
                if conf < confidence_threshold:
                    uncertain.append([row, col])

    timings["classify_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "board": board,
        "confidence": confidence,
        "uncertain": uncertain,
        "tile_count": len(tile_cells),
        "empty_count": len(empty_cells),
        "grid_method": grid.method,
        "timings": timings,
    }


@app.post("/api/detect-corners")
async def detect_corners(file: UploadFile = File(...)):
    """Auto-detect board corners, return as [[x,y], ...] in image coordinates."""
    image = _decode_image(await file.read())
    if image is None:
        return {"error": "Could not decode image"}

    h, w = image.shape[:2]
    corners = auto_detect_corners(image)

    if corners is not None:
        ordered = order_corners(corners)
        return {
            "corners": ordered.tolist(),
            "image_width": w,
            "image_height": h,
            "detected": True,
        }

    # Fallback: full image bounds
    return {
        "corners": [[0, 0], [w, 0], [w, h], [0, h]],
        "image_width": w,
        "image_height": h,
        "detected": False,
    }


@app.post("/api/scan")
async def scan(file: UploadFile = File(...), corners: str = Form(None)):
    image = _decode_image(await file.read())
    if image is None:
        return {"error": "Could not decode image"}

    corner_arr = None
    if corners:
        corner_arr = np.array(json.loads(corners), dtype=np.float32)

    result = scan_image(image, corners=corner_arr)
    return result


@app.get("/")
async def index():
    return FileResponse("web/index.html")


app.mount("/web", StaticFiles(directory="web"), name="web")
