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
uv run scan.py --image board.jpg --interactive         # CLI scan
uv run uvicorn server:app --host 0.0.0.0 --port 8001   # web UI
uv run evaluate.py                                     # batch eval
uv run train.py                                        # retrain
```

The Windows batch files (`retrain.bat`, `start_mobile.bat`) are
convenience wrappers; the `uv run ...` lines inside them work on any
OS.

## Tests
There is no automated test suite. Validation is manual:
- Run `evaluate.py` after any change that touches detection or the
  model -- the held-out board should stay at 99.1% cell accuracy.
- For web/server changes, drive `web/index.html` against a phone (or
  desktop browser with a fake camera) and exercise the
  scan -> corner-adjust -> edit -> save flow end to end. Type checks
  alone do not verify feature correctness.

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
