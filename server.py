"""
FastAPI backend for the Scrabble board scanner.

Run with: uv run uvicorn server:app --host 0.0.0.0 --port 8000
Open http://<your-ip>:8000 on your phone.
"""

import cv2
import numpy as np
import time
import uuid
import os
import shutil
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import json

from src.detection.grid_detect import (
    detect_grid, extract_cell_images, auto_detect_corners, order_corners,
    GRID_SIZE
)
from src.classification.model import (
    load_model, predict_tiles, classify_result
)

app = FastAPI()

SAVED_BOARDS_DIR = os.environ.get("SAVED_BOARDS_DIR", "./saved_boards")
os.makedirs(SAVED_BOARDS_DIR, exist_ok=True)

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
    timings = {}

    t0 = time.perf_counter()
    grid = detect_grid(image, corners=corners, interactive=False)
    timings["grid_ms"] = (time.perf_counter() - t0) * 1000

    cell_images = extract_cell_images(grid)

    t0 = time.perf_counter()
    board = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    confidence = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    uncertain = []
    tile_count = 0

    model = get_model()
    all_imgs = [img for _, _, img in cell_images]
    predictions = predict_tiles(model, all_imgs, device="cpu")

    for (row, col, _), (label, conf) in zip(cell_images, predictions):
        mapped = classify_result(label)
        board[row][col] = mapped
        confidence[row][col] = conf
        if mapped != ".":
            tile_count += 1
            if conf < confidence_threshold:
                uncertain.append([row, col])

    timings["classify_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    weighted_sum = 0.0
    weight_total = 0.0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            w = 3.0 if board[r][c] != "." else 1.0
            weighted_sum += w * confidence[r][c]
            weight_total += w
    readiness_score = min(1.0, max(0.0, weighted_sum / weight_total)) if weight_total > 0 else 0.0

    return {
        "board": board,
        "confidence": confidence,
        "uncertain": uncertain,
        "tile_count": tile_count,
        "empty_count": GRID_SIZE * GRID_SIZE - tile_count,
        "grid_method": grid.method,
        "timings": timings,
        "readiness_score": round(readiness_score, 3),
    }


@app.post("/api/detect-corners")
async def detect_corners(file: UploadFile = File(...)):
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


@app.post("/api/boards/save")
async def save_board(
    file: UploadFile = File(...),
    board: str = Form(...),
    blanks: str = Form("[]"),
    confidence: str = Form("[]"),
    name: str = Form(""),
    auto_detected_corners: str = Form("null"),
    final_corners: str = Form("null"),
    corners_were_adjusted: str = Form("false"),
):
    board_id = str(uuid.uuid4())[:8]
    board_dir = os.path.join(SAVED_BOARDS_DIR, board_id)
    os.makedirs(board_dir, exist_ok=True)

    img_bytes = await file.read()
    with open(os.path.join(board_dir, "image.jpg"), "wb") as f:
        f.write(img_bytes)

    meta = {
        "id": board_id,
        "board": json.loads(board),
        "blanks": json.loads(blanks),
        "confidence": json.loads(confidence),
        "name": name or f"Board {board_id}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auto_detected_corners": json.loads(auto_detected_corners),
        "final_corners": json.loads(final_corners),
        "corners_were_adjusted": json.loads(corners_were_adjusted),
    }
    with open(os.path.join(board_dir, "board.json"), "w") as f:
        json.dump(meta, f)

    return {"id": board_id, "name": meta["name"], "timestamp": meta["timestamp"]}


@app.get("/api/boards")
async def list_boards():
    boards = []
    if not os.path.isdir(SAVED_BOARDS_DIR):
        return {"boards": []}
    for name in os.listdir(SAVED_BOARDS_DIR):
        meta_path = os.path.join(SAVED_BOARDS_DIR, name, "board.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            boards.append({
                "id": meta["id"],
                "name": meta.get("name", name),
                "timestamp": meta.get("timestamp", ""),
            })
    boards.sort(key=lambda b: b["timestamp"], reverse=True)
    return {"boards": boards}


@app.get("/api/boards/{board_id}")
async def get_board(board_id: str):
    meta_path = os.path.join(SAVED_BOARDS_DIR, board_id, "board.json")
    if not os.path.isfile(meta_path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    with open(meta_path) as f:
        return json.load(f)


@app.get("/api/boards/{board_id}/image")
async def get_board_image(board_id: str):
    img_path = os.path.join(SAVED_BOARDS_DIR, board_id, "image.jpg")
    if not os.path.isfile(img_path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(img_path, media_type="image/jpeg")


@app.delete("/api/boards/{board_id}")
async def delete_board(board_id: str):
    board_dir = os.path.join(SAVED_BOARDS_DIR, board_id)
    if not os.path.isdir(board_dir):
        return JSONResponse({"error": "Not found"}, status_code=404)
    shutil.rmtree(board_dir)
    return {"deleted": board_id}


@app.api_route("/api/boards/{board_id}/corners", methods=["PATCH"])
async def update_board_corners(board_id: str, request: Request):
    meta_path = os.path.join(SAVED_BOARDS_DIR, board_id, "board.json")
    if not os.path.isfile(meta_path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    body = await request.json()
    with open(meta_path) as f:
        meta = json.load(f)
    meta["final_corners"] = body["final_corners"]
    meta["corners_were_adjusted"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return {"ok": True}


@app.get("/boards/{board_id}/edit-corners")
async def edit_corners_page(board_id: str):
    return FileResponse("web/edit-corners.html")


@app.get("/")
async def index():
    return FileResponse("web/index.html")


app.mount("/web", StaticFiles(directory="web"), name="web")
