# CLAUDE.md -- Project conventions

Read `SPEC.md` first for the full project picture. This file covers
day-to-day conventions for contributing.

## Stack
- Python 3.12 (`.python-version`)
- Package + env: `uv` (`pyproject.toml` + `uv.lock`). Always
  `uv sync` to install; always `uv run <cmd>` to execute.
- ML: PyTorch + torchvision. CPU-only by default; `--device cuda` is
  supported in `train.py` but the shipped model is trained for CPU
  inference.
- CV: OpenCV (`opencv-python`).
- Web: FastAPI + uvicorn, single static `web/index.html` (no build
  step, no JS framework).
- Deployment artifact: `models/tile_classifier.pt` (PyTorch) and a
  matching `.onnx` export for non-Python consumers.

## Running things

```bash
uv sync                                                # install deps
```

Corner detection is feature-complete; the tile classification
rebuild is the current focus. The corner pipeline lives under
`scripts/`; see `scripts/README.md` for the full per-stage tour
(intake -> labeling -> training-set build -> train -> evaluate).
The legacy tile-classification + web-UI entry points (`scan.py`,
`server.py`, `evaluate.py`, `train.py`, `extract_tiles.py`,
`web/index.html`, `retrain.bat`, `start_mobile.bat`) have been
moved to `ss/` (superseded); the shipped `models/tile_classifier.pt`
is still used by anything that pulls the tile classifier directly.

## Pipeline scripts

`scripts/README.md` is the canonical reference for the corner-
detection pipeline -- what each script does, what it reads/writes,
and the order to run them in for new data. The high-level flow:

```
import_coco_dataset.py / rename_coco_files.py   # intake
  -> batch_detect_corners.py                    # auto-seed (optional)
  -> verify_corners.py                          # human review
  -> label_boards.py                            # group by physical board
  -> set_augment_level.py                       # full vs color_only
  -> make_splits.py                             # train/val/test
  -> augment_corners.py                         # build augmented set
  -> train_corners.py    | train_corner_dense.py
                                                # train -> runs/run_NNN
  -> visualize_predictions.py |
     visualize_predictions_dense.py             # eyeball val predictions
  -> predict_test.py | predict_test_dense.py    # held-out test eval
```

The `_dense` variants are the **shipping** corner regressor
(predicts all 256 grid intersections, not just 4 corners). The
4-corner scripts (`train_corners.py`, `visualize_predictions.py`,
`predict_test.py`) are kept for ablations and share all upstream
labeling, board grouping, augmentation, and split tooling with the
dense scripts -- both pipelines read `data/annotations/corners.json`
+ `data/splits/`. The dense run dir is `training/corner_dense/runs/`;
its layout (`config.yaml`, `best_model.pt`, `metrics.json`,
`loss_curve.png`, `log.txt`, `augmented/`, `predictions/`,
`predictions_test_NN/`) mirrors `training/corner/runs/`.

When you add a script to `scripts/`, update `scripts/README.md` at
the same commit so this reference doesn't rot.

## Tests
There is no automated test suite. Validation is manual:
- For dense regressor changes, run `scripts/train_corner_dense.py`
  (a short epoch budget is fine for smoke-testing the pipeline)
  and then `scripts/visualize_predictions_dense.py
  training/corner_dense/runs/run_NNN` to eyeball whether the
  predicted 256-point lattice lands on the real grid lines across
  the val set. Use `scripts/predict_test_dense.py` against the test
  split only once a candidate looks good on val -- the test set is
  meant to be peeked at sparingly.
- For 4-corner regressor changes (`scripts/train_corners.py`):
  same flow with `scripts/visualize_predictions.py` and
  `scripts/predict_test.py`.
- For augmentation changes, `scripts/sanity_check_corners.py`
  draws the manifest's normalized coords onto a sample of augmented
  images and reports min/max coord ranges; catches labeling math
  bugs before they silently train a bad model.
- Legacy: tile-classifier validation was via `ss/evaluate.py`
  against held-out boards; revisit when the tile pipeline rebuild
  resumes.

When you add code that has natural unit boundaries (e.g. pure
functions in `grid_detect.py`), tests are welcome but not required.

## Conventions

### Code style
- Module docstring at the top of every script explaining purpose +
  usage example. Match the existing tone -- short, terse, real
  command examples in the docstring.
- Type hints on function signatures. Return-type annotations on
  anything non-trivial.
- No comments restating what the code does. Comments are reserved
  for *why* -- a constraint, an invariant, a non-obvious choice.
  Existing files follow this; please keep it that way.
- `from x import y` over `import x.y as ...` when reasonable.
- Constants in `UPPER_CASE` at module top
  (e.g. `GRID_SIZE`, `CLASSES`, `TILE_INPUT_SIZE`).

### Filesystem layout
- Vision/ML code lives under `src/{detection,classification}/`.
- Top-level scripts (`scan.py`, `train.py`, `extract_tiles.py`,
  `evaluate.py`, ...) are entry points; they import from `src/` and
  expose `argparse` CLIs.
- Generated data is git-ignored under `data/`. Only `data/README.md`
  is tracked. `saved_boards/` (user-saved boards from the web UI)
  is also git-ignored.

### Class labels
`CLASSES` is defined once, in `src/classification/model.py`. Every
script that touches training data, inference output, or folder names
imports from there -- do not redefine the list. Folder names under
`extracted/tiles/<CLASS>/` must match `CLASSES` exactly so
`ImageFolder`'s alphabetical sort lines up.

### Ground truth format
- 15 lines, 15 chars per line.
- Uppercase A-Z = printed tile face.
- Lowercase a-z = blank tile playing as that letter. Blanks have no
  printed face: `extract_tiles.py` skips them, and `evaluate.py`
  treats them as empty so the model isn't penalised for predicting
  `.`.
- `.` = empty or bonus square.

### Web UI <-> server contract
- All API endpoints are JSON over POST/GET under `/api/`. See
  `server.py` for the canonical list.
- The web client sends camera frames as JPEG `multipart/form-data`
  uploads. Corners (when adjusted) are sent as a JSON-encoded
  4x2 array in a form field, not a separate JSON body.
- Saved boards live under `./saved_boards/<id>/` with `image.jpg` +
  `board.json`. The `board.json` schema is whatever `server.py`'s
  `/api/boards/save` writes -- extend it carefully and update the
  `is_update` branch to keep immutable fields stable.
- `readiness_score` (0-1) gates auto-capture in the polling loop --
  the recent default threshold is 0.85 (see recent commit
  `bd18f3a`).

### Performance budget
- Full scan budget: <500ms on CPU for a typical phone photo.
- Classifier alone: <50ms for all 225 cells batched.
- If you add work to the scan path (`scan_image` in `server.py` or
  `scan_board` in `scan.py`), time it -- the existing code reports
  per-stage timings and the UI exposes them.

## When changing the data pipeline

The data layout is mid-rebuild. If you're updating the training
scripts to point at the new `data/raw/boards/`,
`data/annotations/`, `data/extracted/tiles/`, `data/splits/` layout:
- Keep `CLASSES` as the source of truth for folder names.
- The held-out evaluation board should move into `data/splits/`
  rather than a separate `data/eval/` -- `splits/` is the right
  place for any "this set of files is the test set" manifest.
- Update `SPEC.md` (the "Data pipeline" and "Modules" sections) at
  the same time you update the scripts -- the spec currently flags
  these scripts as out-of-date.
- Update `retrain.bat` accordingly.

## Corner detection training

Two training surfaces share the same upstream data pipeline:

```
training/corner/                          training/corner_dense/
  corner_config.yaml                        dense_config.yaml
  CHANGELOG.md                              CHANGELOG.md
  runs/run_NNN/  (gitignored)               runs/run_NNN/  (gitignored)
```

- `training/corner_dense/` -- **shipping** model. Dense supervision
  on all 256 grid intersections, fully-conv head. Driven by
  `scripts/train_corner_dense.py`. See `SPEC.md`'s "Dense regressor"
  section for the architecture and loss.
- `training/corner/` -- 4-corner regressor; kept for ablations.
  Driven by `scripts/train_corners.py`.

Running:

```bash
# Shipping pipeline:
uv run scripts/train_corner_dense.py                                       # default config
uv run scripts/train_corner_dense.py --config path/to/some.yaml            # override
uv run scripts/visualize_predictions_dense.py training/corner_dense/runs/run_NNN
uv run scripts/predict_test_dense.py training/corner_dense/runs/run_NNN

# 4-corner ablation pipeline:
uv run scripts/train_corners.py
uv run scripts/visualize_predictions.py training/corner/runs/run_NNN
uv run scripts/predict_test.py training/corner/runs/run_NNN
```

Each `train_*` invocation creates a fresh `runs/run_NNN/`
(auto-incremented) and copies the resolved config into it. Don't
mutate an existing run directory -- start a new one. Don't peek at
the test set: only run `predict_test*` once a candidate looks good
on val.

Config conventions:
- `corner_config.yaml` / `dense_config.yaml` hold the canonical
  defaults; keep them minimal and human-readable.
- Experiment variants are separate YAML files (anywhere on disk),
  passed with `--config <path>`.
- Anything that affects the trained weights (augmentation params,
  optimizer, split seed, epochs, pool_size, dropout,
  unfreeze_layers, grid_reg_weight, collinearity_weight) must come
  from the config -- no hidden defaults baked into the code.
- After every meaningful run, append a `## run_NNN -- <title>` entry
  to the relevant `CHANGELOG.md` (`training/corner_dense/` or
  `training/corner/`) explaining what changed and what the result
  was. Newest entries at the top.
- Setting `grid_reg_weight` or `collinearity_weight` to 0 disables
  that term's gradient contribution but keeps it logged each epoch
  as a diagnostic; prefer zeroing the weight over deleting the key
  so historical configs remain reproducible.

### Board-level split rule

**Never split images of the same physical board across train and
val/test.** A single board photographed from multiple angles is
*one* unit for the purposes of splitting -- all its photos must land
in the same split. Otherwise the model can memorize board-specific
cues (the corkboard behind it, a chipped tile, the rack on the side)
and the val/test numbers will overestimate generalization.

The split is by *board id*, not by file. The manifest under
`data/splits/` is authoritative; build new splits by grouping at the
board level first, then partitioning groups.

### HEIC images

Phone photos in `data/raw/boards/` may be `.heic` / `.HEIC`. Always
load board images through `src.image_io.load_board_image` (or check
`is_board_image` for filtering) -- it handles HEIC via `pillow-heif`
and returns OpenCV-format `np.ndarray`. Do not call `cv2.imread`
directly on raw board paths.

## Don'ts
- Don't recreate `data/raw_boards/` or `data/eval/` -- those are
  retired. New code reads from `data/raw/boards/` and
  `data/splits/`.
- Don't change the `CLASSES` list without retraining -- the model
  weights bake in the class ordering.
- Don't add a build step to the web UI. `web/index.html` is a
  single self-contained file by design.
- Don't commit anything under `data/` other than `data/README.md`,
  or anything under `saved_boards/`.
