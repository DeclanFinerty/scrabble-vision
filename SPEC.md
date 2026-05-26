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

The shipping model is the **dense regressor** (`training/corner_dense/`,
described below). The win is real and came *with no additional
labeling* -- the 256 dense targets are derived analytically from
the same 4 corner labels via perspective interpolation. The
comparisons:

- *Like-for-like on the original 10-image val*: dense run_001
  reached **1.26%** mean corner error vs the best 4-corner run
  (`training/corner/runs/run_015`) at **2.72%** -- a ~54% relative
  reduction with the only change being the supervision density.
- *Shipping (run_005) on the current val*: **1.70%** on a
  deliberately enlarged 54-image val (5.4x bigger than the original
  10-image set, refreshed in dense run_002 to include
  hard-background and visually-tricky boards). The harder val is
  the apples-to-apples baseline for any future work, not the old
  one; the 1.7% figure absorbs that difficulty bump and is still
  comfortably below the old 4-corner best.

The 4-corner regressor below is retained for ablation only.

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
- Best 4-corner result: 2.72% mean val error (run_015, on the
  original 10-image val -- see `training/corner/CHANGELOG.md` for
  the per-run history). Superseded by the dense regressor below,
  which on that same 10-image val reached 1.26% (`corner_dense/run_001`)
  and on the current 54-image val ships at 1.70%
  (`corner_dense/run_005`).

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

### Dense regressor (`scripts/train_corner_dense.py`)

The dense regressor replaces the 4-corner head with one that predicts
**all 256 intersections of the 15x15 board grid** (a 16x16 lattice).
It's the shipping corner detector.

- **Targets**: the 256 dense points are computed *analytically* from
  the existing 4 corner labels via a perspective transform from the
  unit grid to image space (`compute_dense_targets` in
  `scripts/train_corner_dense.py`). The augmentation pipeline is
  shared with the 4-corner regressor unchanged -- it writes a
  4-corner manifest; the dense targets are derived on the fly at
  `__getitem__` time. No new labeling is required.
- **Ordering**: row-major (`index = row * 16 + col`, row 0 at the
  top). The 4 board corners land at indices `(0, 15, 255, 240)` for
  TL/TR/BR/BL, so per-corner error is directly comparable against
  the 4-corner runs.
- **Backbone**: same MobileNetV2 trunk as the 4-corner regressor,
  with `unfreeze_layers` honoured the same way (frozen BN forced to
  eval mode while only later blocks train).
- **Head**: fully-convolutional with optional spatial pooling at the
  end --

  ```
  features (B, 1280, 7, 7)
    -> Conv2d(1280, 512, 1x1) -> ReLU
    -> Conv2d(512,  256, 1x1) -> ReLU
    -> AdaptiveAvgPool2d(pool_size)
    -> Flatten
    -> Dropout(dropout)
    -> Linear(256 * pool_size**2, 512)
  ```

  The 1x1 convs do per-spatial-cell channel mixing on the native
  7x7 grid before pooling. `pool_size: 1` (default; GAP) gives the
  smallest FC; raising it preserves more positional information at
  the cost of head size.
- **Loss** = `SmoothL1` on all 512 normalized outputs
            + `grid_reg_weight` * spacing-variance penalty
            + `collinearity_weight` * perpendicular-distance penalty.
  Both penalties are computed on the predicted 16x16 lattice:
  - *Spacing variance* (`grid_regularity_penalty`): for each row /
    column, variance of the 15 consecutive-point distances. Disabled
    by default (`grid_reg_weight: 0.0`) because perspective
    augmentation produces labels with unequal row/column spacing, so
    penalizing variance fights the real labels.
  - *Collinearity* (`grid_collinearity_penalty`): for each row /
    column, mean squared perpendicular distance from each
    intermediate point to the line through the two endpoints.
    True under any perspective, so always safe to apply.
  Setting a weight to 0 zeros the gradient contribution but still
  logs the unweighted penalty each epoch as a diagnostic; weights
  are kept in the config (rather than removing keys) so older runs'
  configs remain reproducible.
- **Eval**: reports per-corner pixel + %-of-diagonal error at the 4
  board-corner indices (for direct comparison with the 4-corner
  runs) *and* mean error across all 256 lattice points. Both go into
  `metrics.json`; the per-epoch console line carries both.
- **Best result (shipping)**: run_005 -- 1.70% mean corner error /
  1.08% all-256 error on the current 54-image val; 5.3% / 3.2% on
  the held-out 22-image test set. The cleanest dense-vs-4-corner
  comparison is run_001, which on the original 10-image val reached
  1.26% versus the 4-corner best of 2.72% -- a ~54% relative
  reduction with no change other than supervision density. run_005
  ships rather than run_001 because the val set was deliberately
  enlarged 5.4x in run_002 (refreshed to include hard-background
  boards), and run_005 is the best result on that harder val. See
  `training/corner_dense/CHANGELOG.md` for per-run notes.
- **Run dir layout** mirrors the 4-corner pipeline:
  `training/corner_dense/runs/run_NNN/` contains `config.yaml`,
  `best_model.pt`, `metrics.json` (best epoch + full history),
  `loss_curve.png` (total + base/reg/coll components),
  `log.txt`, an `augmented/` subdir (gitignored, auto-deleted on
  success unless `--keep-augmented`), a `predictions/` subdir from
  `visualize_predictions_dense.py`, and one or more auto-numbered
  `predictions_test_NN/` subdirs from `predict_test_dense.py`.

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

- **Corner detection is feature-complete.** The dense regressor
  (`scripts/train_corner_dense.py`, run_005 ships) is the production
  corner detector at 1.70% mean val corner error on the current
  54-image val. The cleanest comparison to the prior 4-corner stack
  is dense run_001 (1.26%) vs 4-corner run_015 (2.72%) on the
  original 10-image val -- the dense supervision is the biggest
  single improvement to the corner stack. The 4-corner regressor
  (`scripts/train_corners.py`) is retained for ablation only. See
  `training/corner_dense/CHANGELOG.md` for run notes and
  `data/annotations/corners.json` for the labeled set. Follow-ups
  (from-scratch CNN comparison, ONNX export, heuristic-vs-trained
  eval harness) live in `TODO.md`.
- **Tile classification rebuild is the active focus.** The existing
  `models/tile_classifier.pt` continues to ship while the rebuild
  is in flight. `extract_tiles.py` / `augment_tiles.py` / `train.py` /
  `evaluate.py` / `server.py` / `web/index.html` have been moved
  into `ss/` (superseded) pending the rebuild; references to them
  in the "Modules" + "Running end-to-end" sections document the
  legacy layout, not the current root. The rebuild reads from the
  new `data/raw/boards/` + `data/annotations/` + `data/extracted/tiles/`
  layout and uses the dense regressor's perspective warp as the
  upstream input.
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
