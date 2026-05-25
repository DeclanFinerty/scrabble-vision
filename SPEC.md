# Scrabble Vision -- Spec

## Purpose

Offline computer-vision pipeline that scans a photograph of a physical
Scrabble board and produces a 15x15 letter grid. Runs locally on CPU
(no external API calls), targets <500ms per board.

Two surfaces consume the pipeline:
- `scan.py` -- CLI for a single image (optionally interactive corner
  picking).
- `server.py` + `web/index.html` -- mobile-first web UI. Users open the
  page on their phone, point the camera at a board, and get a
  predicted grid they can edit, name, and save. The HTML side is
  purely for editing/correcting the model's output, not for running
  the model.

## Architecture

```
photo -> [grid detection] -> warped 900x900 board
       -> [equal-grid slice] -> 225 cell images (32x32 grayscale)
       -> [TileClassifier CNN] -> 225 (label, confidence) pairs
       -> board[15][15] of {letter | '.'}
```

### Grid detection (`src/detection/grid_detect.py`)
- `auto_detect_corners` tries three strategies in order until one
  produces a quadrilateral that passes `_validate_corners`
  (area fraction, aspect ratio, convexity):
  1. Contour-based: downscale -> blur -> Canny -> contour ->
     `approxPolyDP`/`minAreaRect` (tried with light/medium/heavy
     blur params).
  2. Flood-fill from image borders (handles both polarities).
  3. Hough lines: detect grid lines, cluster, take outermost
     horizontal+vertical pairs, intersect.
- `GridFitUI` -- OpenCV window with draggable corner handles and a
  live 15x15 overlay; used by `scan.py --interactive` to confirm or
  fine-tune auto-detected corners.
- `perspective_correct` warps the quadrilateral to a 900x900 square.
  Cells are extracted with `equal_grid` (5% padding to trim grid
  lines) plus a small buffer to catch off-center tiles.

### Classification (`src/classification/model.py`)
- `TileClassifier`: small CNN (~200K params).
  - 3 conv blocks (32 -> 64 -> 128 channels) with BN + ReLU + maxpool/
    adaptive-avgpool to 4x4, then 128-unit FC head with dropout
    (0.3/0.2).
  - Input: `(B, 1, 32, 32)` float32 in [0, 1], grayscale.
  - Output: 28-class logits.
- Classes: `A`-`Z`, `BONUS`, `EMPTY` (28 total). Ordering is the
  alphabetical sort of `CLASSES` -- must match
  `torchvision.datasets.ImageFolder`'s folder sort.
- `classify_result` maps `BONUS`/`EMPTY` to `.` for board output. Two
  empty-ish classes are kept separate during training to let the
  model learn the visual distinction, but they collapse at inference.
- Model weights ship at `models/tile_classifier.pt`; an ONNX export
  lives alongside at `models/tile_classifier.onnx`.

### Eval baseline (held-out board)
- Cell accuracy: 223/225 (99.1%)
- Letter accuracy: 92/94 (97.9%)
- Empty accuracy: 131/131 (100%)

## Corner Detection Model

A learned regressor replaces the heuristic `auto_detect_corners` for
production use. The heuristic remains as a fallback for when the
model is unavailable or produces low-confidence output.

### Architecture (`CornerRegressor` in `scripts/train_corners.py`)

- Backbone: MobileNetV2 pretrained on ImageNet (ResNet18 also
  supported via the `backbone` config field). `freeze_backbone` is
  superseded by `unfreeze_layers`: `-1` for fully trainable, `0`
  for fully frozen, `N > 0` to unfreeze just the last N children of
  `features`. Frozen BatchNorm modules are forced to eval mode so
  their running stats don't drift when only the head trains.
- Regression head: `AdaptiveAvgPool2d((P, P))` → `Flatten` →
  `Linear(feat_dim * P*P, 256)` → ReLU → Dropout → `Linear(256, 8)`.
  `P` (`pool_size` in the config) defaults to 7 to match MobileNetV2's
  native 7x7 final feature map (no spatial pooling loss); was 4 in
  earlier runs and 1 (GAP) destroyed positional info so the model
  could only learn average corner positions.
- Output: 8 floats, interpreted as 4 (x, y) pairs in TL/TR/BR/BL
  order, normalised to `[0, 1]` of the letterboxed 224x224 input.
  Values can fall outside `[0, 1]` (the loss accepts this) when
  augmentation crops a corner out of frame.

### Input + eval preprocessing

- Each image is **letterboxed** to 224x224 (aspect-preserving resize
  + black-bar padding), then ImageNet-normalised. Letterbox params
  (`scale`, `pad_x`, `pad_y`) round-trip exactly: any normalised
  prediction can be denormalised back to original-image pixel coords.
- Val images go through the same letterbox -- no augmentation -- so
  train and val see the same aspect-ratio convention. Train images
  come pre-augmented from `data/augmented/corner_train/`.

### Loss + metrics

- `SmoothL1Loss` on normalised coordinates. Unchanged across runs.
- Eval reports pixel error AND `% of image diagonal` per image, then
  averaged. Diagonal-normalised error is resolution-fair across val
  images of different sizes (a 50 px miss on 4032 px vs on 1300 px
  reads very differently in raw pixels). See `evaluate()` in
  `scripts/train_corners.py`.
- Best result so far: 2.7% mean val error (run_015 -- see
  `training/corner/CHANGELOG.md` for the per-run history). Target
  is sub-1%.

### Training pipeline

- Config-driven (`training/corner/corner_config.yaml`); pass a
  different YAML with `--config <path>`. Each `train_corners.py`
  invocation auto-creates `training/corner/runs/run_NNN/` with a
  config snapshot, `best_model.pt`, `metrics.json` (best epoch +
  full per-epoch history, both px and % units), `loss_curve.png`,
  and `log.txt`.
- Optimiser: Adam. Scheduler: `ReduceLROnPlateau` on val loss with
  `factor=0.5`, `patience=lr_patience`. Early-stop on val loss with
  `patience=early_stop_patience`.
- Splits are by **physical board** via `data/annotations/board_groups.json`
  (see `scripts/label_boards.py` + `scripts/make_splits.py`). Multiple
  photos of the same board must all land in the same split so the
  model can't memorise board-specific cues.
- Augmentation is **per-image**, controlled by the `augment_level`
  field in `corners.json`:
  - `full` (default): rotation (0-360 deg) + perspective warp
    (~5% corner perturbation) + random square crop (>=60% board
    area, >=2 corners in-frame) + color jitter, composed as a single
    homography with a round-trip math check per sample.
  - `color_only`: just brightness/contrast/saturation/hue on the
    letterboxed source, for boards where rotation augmentation would
    produce nonsense (e.g. badges, oriented text). Set via
    `scripts/set_augment_level.py`.

### Verification UI (`scripts/verify_corners.py`)

- Standard single-image + `--batch` flows iterate one image at a
  time through `GridFitUI`.
- `--group-mode` iterates by board group. For multi-image groups it
  verifies the first unlabeled image then prompts `y/n/r` to apply
  the same corners + image flips to the rest of the group.
- The UI button strip includes Verify / Rotate / FlipH / FlipV /
  Reset / Skip / Reject / Quit. Rotate cycles the corner array one
  slot CW (relabel when the keypoint exporter put corners in the
  wrong rotation). FlipH / FlipV mirror the image **and** remap
  corners; on Verify the flipped image is persisted to disk and the
  un-flipped original is preserved at `<dataset>/original/<name>`.

See `scripts/README.md` for the full pipeline order from new
Roboflow dataset to a trained run.

## Data pipeline (rebuild in progress)

The data layout has been reset. See `data/README.md` for the current
target structure:

```
data/
  raw/boards/      # board photos
  raw/racks/       # rack photos (new)
  annotations/     # corner + label files
  extracted/tiles/ # cropped + warped tile crops
  splits/          # train/val/test manifests
```

Three stages:
1. **Corner annotation** -- pick the 4 board-corner points on each
   raw photo.
2. **Ground-truth labeling** -- record the 15x15 letter grid for
   each board (uppercase = printed face, lowercase = the letter a
   blank is playing as).
3. **Tile extraction** -- using corners + ground truth, slice each
   board into 225 tile images grouped by class.

The new pipeline replaces the previous `data/raw_boards/`,
`data/train/`, `data/real_tiles/`, and `data/eval/` directories.
`extract_tiles.py`, `augment_tiles.py`, `train.py`, `evaluate.py`,
and `retrain.bat` still reference the **old** paths and need to be
updated to read from `raw/boards/`, `annotations/`, `extracted/tiles/`,
and `splits/` (and to drop the held-out eval split out of `splits/`
rather than a separate `data/eval/`).

Ground-truth file format (`annotations/<stem>.txt`):
- 15 lines of 15 characters each.
- Uppercase A-Z = printed letter.
- Lowercase a-z = blank tile played as that letter (excluded from
  training; treated as empty for eval).
- `.` = empty/bonus square.

## Modules

| Path | Purpose |
|---|---|
| `src/detection/grid_detect.py` | Corner detection, perspective warp, grid overlay, interactive `GridFitUI`. |
| `src/classification/model.py` | `TileClassifier`, preprocessing, batched inference, model load/save, ONNX export. |
| `scan.py` | CLI: image -> 15x15 grid. Supports `--interactive`, `--corners`, `--debug`, `--words`. |
| `server.py` | FastAPI backend. Endpoints: `/api/scan`, `/api/scan/quick` (corner-quality readiness pre-check), `/api/detect-corners`, `/api/boards/*` (CRUD on saved boards). |
| `web/index.html` | Single-page mobile UI: camera capture, polling auto-capture, draggable corner overlay, editable 15x15 grid with confidence colouring, save/load. |
| `generate_tiles.py` | Render synthetic A-Z/BONUS/EMPTY tile images with augmentation. |
| `extract_tiles.py` | Slice real boards into per-class tile crops using saved corners + ground truth. **References old `data/raw_boards/` layout -- needs updating.** |
| `augment_tiles.py` | Class-balancing augmentation of real tile crops. **References old `data/real_tiles/` layout -- needs updating.** |
| `train.py` | Train `TileClassifier` on synthetic + real, with `WeightedRandomSampler` oversampling real, board-grouped val split. Exports ONNX after training. **References old `data/train/` + `data/real_tiles/` -- needs updating.** |
| `evaluate.py` | Batch eval against ground-truth boards in `data/eval/`. Also has `--label-corners` and `--classifier-accuracy` (compare original scan vs user-corrected `saved_boards/`). **References old `data/eval/` -- needs updating.** |
| `retrain.bat` | Windows-only full pipeline: extract -> augment -> generate -> train -> evaluate. |
| `start_mobile.bat` | Windows-only: launches uvicorn + ngrok in two terminals. |

## Running end-to-end

### Scan a single image
```bash
uv run scan.py --image path/to/board.jpg --interactive --debug
```
- `--interactive` opens the `GridFitUI` window. Drag corners,
  Enter to confirm, R to reset to auto, Q to abort.
- `--debug` writes `debug_grid.jpg` and saves used corners to
  `corners.npy`.
- `--corners corners.npy` skips detection and uses the file directly.
- `--words` additionally prints every length>=2 word found
  horizontally and vertically.

### Web scanner (mobile)
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8001
```
Then open `http://<your-lan-ip>:8001` on a phone on the same WiFi.
On Windows, `start_mobile.bat` also launches an `ngrok` HTTPS tunnel
(needed because some mobile browsers refuse camera access on
plain-HTTP non-localhost origins).

### Retrain (after data pipeline rebuild lands)
The intended end-to-end retrain is:
```
extract real tiles -> augment to balance classes
                   -> generate synthetic supplements
                   -> train (synthetic + real, oversample real 3x)
                   -> evaluate on held-out split
```
Currently driven by `retrain.bat`; all four data-touching scripts
need their input/output paths updated for the new `data/` layout.

## Known limitations / planned improvements

- **Corner detection training is the active focus.** Labeling
  (`scripts/verify_corners.py` -- supports group-mode propagation,
  flip H/V with backups to `<dataset>/original/`, and label rotation)
  and the training loop driven by `scripts/train_corners.py` are
  the day-to-day work. Best val error to date is 2.7% of image
  diagonal (run_015); target is sub-1%. See
  `training/corner/CHANGELOG.md` for per-run notes.
- **Tile classification rebuild deferred.** The existing
  `models/tile_classifier.pt` continues to ship; the
  `extract_tiles.py` / `augment_tiles.py` / `train.py` /
  `evaluate.py` / `server.py` / `web/index.html` entry points have
  been moved into `ss/` (superseded) pending the corner work landing
  first. References to them in the "Modules" + "Running end-to-end"
  sections below document the legacy layout, not the current root.
- **Corner detection failure mode.** Heuristic auto-detect can fail
  on busy backgrounds or low-contrast surfaces; the trained
  regressor is the planned replacement, with the heuristic retained
  as a fallback. Until the regressor ships, users fall back to
  manual drag in the (legacy) web UI or `--interactive` CLI.
- **Blank tiles.** They have no printed face, so they're excluded
  from training and counted as empty during eval. The web UI lets
  the user mark blanks post-scan; the model can't infer them.
- **EMPTY vs BONUS class split** is held in training but collapsed at
  inference. This helps the model see bonus squares (which often
  carry coloured printed text) as a distinct visual category but
  produces the same `.` on the board.
- **Windows-only batch drivers.** `retrain.bat` and
  `start_mobile.bat` are `.bat` files. The underlying `uv run ...`
  commands work on any OS -- a shell equivalent would be cheap to add.
- **No tests.** No automated test suite exists. End-to-end validation
  is currently `evaluate.py` against held-out boards.
