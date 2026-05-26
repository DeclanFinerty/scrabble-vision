# Scripts

Operational entry points for the corner-detection data pipeline and
training loop. Everything here is an `argparse` CLI; run any with
`-h` for full flag docs.

The pipeline assumes the layout in `SPEC.md` / `data/README.md`:
raw photos under `data/raw/boards/`, annotations under
`data/annotations/`, splits under `data/splits/`, and training runs
under `training/corner/runs/`. Augmented training data lives next to
the run that uses it (`training/corner/runs/run_NNN/augmented/`),
generated automatically by `train_corners.py`. The legacy
`data/augmented/` location is still the default when running
`augment_corners.py` standalone (e.g. for ad-hoc inspection).

## Pipeline order

```
                            +--------------------------------+
                            |  new images dropped in         |
                            |  data/raw/boards/<dataset>/    |
                            +--------------------------------+
                                         |
              import_coco_dataset.py     |     rename_coco_files.py
              (fresh COCO dataset,       |     (you labeled the long
               no labels yet)            |      filenames first; this
                                         |      remaps without touching
                                         |      label content)
                                         v
                            +--------------------------------+
                            |  batch_detect_corners.py       |
                            |  optional: auto-detect seeds   |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  verify_corners.py             |
                            |  REQUIRED: human review        |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  label_boards.py               |
                            |  assign physical-board groups  |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  set_augment_level.py          |
                            |  full vs color_only            |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  make_splits.py                |
                            |  train/val/test manifests      |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  augment_corners.py            |
                            |  build 224x224 training set.   |
                            |  optional: train_corners.py    |
                            |  calls this internally and     |
                            |  writes into the run dir.      |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  preview_augmentations.py      |
                            |  sanity_check_corners.py       |
                            |  optional eyeball checks       |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  train_corners.py              |
                            |  augments into                 |
                            |    runs/run_N/augmented/ then  |
                            |    trains -> runs/run_N/       |
                            +--------------------------------+
                                         v
                            +--------------------------------+
                            |  visualize_predictions.py      |
                            |  per-val-image predictions     |
                            +--------------------------------+
```

## Stage 1 — Data intake

Get new images on disk with clean filenames and a baseline corner
seed (or empty entry) in `corners.json`.

### `import_coco_dataset.py`
Renames Roboflow COCO images to their short `extra.name`, rewrites
the COCO json's `file_name`, and seeds `corners.json` with the 4
board corners (re-ordered TL/TR/BR/BL). Auto-detects two Roboflow
export shapes:
  - **object detection task** -- `CORNER` category with 4 bboxes
    per image (one per corner); bbox centers are taken.
  - **keypoint detection task** -- one annotation per image whose
    `keypoints` array holds 4 `(x, y, visibility)` triples; visible
    keypoints are taken directly.
Used the **first** time a COCO dataset is added, **before** any
labeling. Never overwrites verified entries; `--force-corners` is
required to overwrite unverified entries that already have corners.

Reads: `data/raw/boards/<dataset>/_annotations.coco.json`
Writes: renamed image files in place, updated COCO json,
  `data/annotations/corners.json`

### `rename_coco_files.py`
The "I already labeled the long filenames, don't lose those labels"
variant. Renames files + remaps existing `corners.json` and
`board_groups.json` entries to the new short keys. Never imports
COCO bbox annotations, never modifies entry content (verification
flags, corners, augment_level all untouched). Defaults to dry-run;
`--apply` to commit.

Reads: same as above
Writes: image renames, COCO json `file_name` updates, key remaps in
  `corners.json` + `board_groups.json`

### `batch_detect_corners.py`
Runs the heuristic `auto_detect_corners` on every raw image and
writes the result into `corners.json` for any entry that doesn't
already have corners. Optional — speeds up verification by giving
human reviewers a starting quad to nudge. Never touches verified or
rejected entries.

Reads: `data/raw/boards/**/*`
Writes: `data/annotations/corners.json`

## Stage 2 — Human labeling

Three small interactive tools. All save after every choice so quit
(or Ctrl+C) is safe.

### `verify_corners.py`
The gating step. Walks images without a `corners_verified` flag,
shows each in an OpenCV window with the GridFitUI drag handles,
and prompts the user to verify / reset / reject / skip / quit
via the on-image button strip (mouse-only — no terminal input,
because Qt event-loop reentrancy on Linux broke it).

Reads: `data/raw/boards/**/*`, `data/annotations/corners.json`
Writes: `corners_verified` / `corners_rejected` flags in
  `corners.json`

### `label_boards.py`
Assigns each verified image to a "physical board group" so the
splits step can keep all photos of the same board in one split.
Two flavours of group key:
- Named groups `A`, `B`, ..., `Z`, `AA`, ... for boards with
  multiple photos.
- Unique groups `U01`, `U02`, ... for one-off boards.

Reads: `data/raw/boards/**/*`, `data/annotations/board_groups.json`
Writes: `data/annotations/board_groups.json`

### `set_augment_level.py`
Pick per-image augmentation profile:
- `full` (default): rotation + perspective + crop + color
- `color_only`: just brightness/contrast/saturation/hue, on top of a
  letterboxed 224x224. Use this for boards where rotation
  augmentation would produce nonsense.

Reads: `data/raw/boards/**/*`, `data/annotations/corners.json`
Writes: `augment_level` field in `corners.json`

## Stage 3 — Build the training set

### `make_splits.py`
Reads board-group assignments + verified corners and writes the
three split manifests. Splits are by group, never by image.
Defaults: `test=D`, `val=A + N unique boards`, `train=rest`.
Override via `--test-groups`, `--val-groups`, `--val-unique-count`,
`--seed`.

Reads: `data/annotations/board_groups.json`,
  `data/annotations/corners.json`
Writes: `data/splits/corner_{train,val,test}.txt`

### `augment_corners.py`
Reads `corner_train.txt` + `corners.json` and produces N
augmentations per image (default `aug_count` from config, fallback
30) into `--out-dir` (default `data/augmented/`), branching on each
image's `augment_level`. Round-trip checks the coordinate math per
sample (T_inv(T(corners)) must equal the source corners to
floating-point tolerance) so any silent label drift aborts the run.

`train_corners.py` calls this programmatically (via `run_augment()`)
and points `--out-dir` at the new run's `augmented/` subdir, so two
trainings can run in parallel with different augmentation params
without trampling each other's data. Run it standalone only for
ad-hoc inspection.

Reads: `data/splits/corner_train.txt`, `data/annotations/corners.json`,
  `data/raw/boards/**/*`, `training/corner/corner_config.yaml`
Writes: `<out_dir>/corner_train/*.jpg`,
  `<out_dir>/corner_manifest.csv`

### `preview_augmentations.py`
Spot-check tool. Shows 20 augmentations of one source image in a
4×5 grid with corner overlays drawn on top. `--save out.png` writes
the grid to disk; otherwise a cv2 window pops up.

Reads: one source image + its corners
Writes: nothing by default, or `out.png` with `--save`

### `sanity_check_corners.py`
Picks N random rows from a `corner_manifest.csv` (default
`data/augmented/corner_manifest.csv`; pass `--manifest
training/corner/runs/run_NNN/augmented/corner_manifest.csv` for a
per-run check), draws the manifest's coords onto the 224x224 image,
saves to `training/corner/sanity_check/`. Also reports min/max ranges
of all manifest coords (expect mostly `[0, 1]` with a tail of
negatives and values >1 from crops that excluded corners). Catches
augmentation labeling bugs *before* they silently train a bad model.

Reads: `--manifest` (default `data/augmented/corner_manifest.csv`)
Writes: `training/corner/sanity_check/<sample>.jpg`

## Stage 4 — Train

### `train_corners.py`
Reads `training/corner/corner_config.yaml` (or `--config <path>`),
builds a MobileNetV2 or ResNet18 + regression-head model. Before
training it generates a fresh augmented set into the new run's
`augmented/` subdir (via `augment_corners.run_augment()`) so the
training data, the params that produced it, and the trained weights
all live next to each other. Pass `--reuse-augmented <manifest_path>`
to skip augmentation and reuse a sibling run's data instead.

Evaluates per-epoch against `corner_val.txt` (letterboxed, no
augmentation), and writes everything to an auto-incrementing
`training/corner/runs/run_NNN/` directory:
- `config.yaml` (snapshot of config used)
- `augmented/corner_train/*.jpg` + `augmented/corner_manifest.csv`
  (per-run training data)
- `best_model.pt` (state_dict at best val epoch)
- `metrics.json` (best-epoch + full per-epoch history, both px and
  % of image diagonal)
- `loss_curve.png`
- `log.txt` (stdout duplicate)

Loss is `SmoothL1` on normalized coords. Pixel + diagonal-% errors
are reported alongside in eval.

Reads: `training/corner/corner_config.yaml`,
  `data/splits/corner_{train,val}.txt`, `data/annotations/corners.json`,
  `data/raw/boards/**/*`
Writes: `training/corner/runs/run_NNN/` (including its `augmented/`
  subdir, unless `--reuse-augmented` is given)

## Stage 5 — Evaluate

### `visualize_predictions.py`
Loads a run directory's `config.yaml` + `best_model.pt`, runs
inference on every val image, and writes annotated full-resolution
JPEGs (predicted in red, ground truth in green) plus a summary
grid to `<run_dir>/predictions/`. Per-image table prints both
pixel error and % of diagonal. `--trace` prints the full letterbox
+ denormalize math for the first val image so you can verify
nothing drifted end-to-end.

Reads: `<run_dir>/config.yaml`, `<run_dir>/best_model.pt`,
  `data/splits/corner_val.txt`, `data/annotations/corners.json`
Writes: `<run_dir>/predictions/<stem>.jpg`,
  `<run_dir>/predictions/summary.jpg`

## Quick-reference: order of script invocation for new data

After dropping a new Roboflow dataset into `data/raw/boards/<name>/`:

```bash
# 1. Get short filenames (pick ONE of these depending on labeling state):
uv run scripts/import_coco_dataset.py --coco data/raw/boards/<name>/_annotations.coco.json
# OR (if you already labeled the long filenames):
uv run scripts/rename_coco_files.py --coco data/raw/boards/<name>/_annotations.coco.json --apply

# 2. Optional: seed corners with the heuristic
uv run scripts/batch_detect_corners.py

# 3. Human labeling
uv run scripts/verify_corners.py --batch
uv run scripts/label_boards.py
uv run scripts/set_augment_level.py

# 4. Build training set
uv run scripts/make_splits.py --seed 42 --val-unique-count 3
uv run scripts/augment_corners.py --count 30 --seed 42
uv run scripts/sanity_check_corners.py --seed 42   # optional eyeball

# 5. Train + visualize
uv run scripts/train_corners.py
uv run scripts/visualize_predictions.py training/corner/runs/run_NNN
```

If you only need to re-train (config tweaks, no new data) start at
step 4 (or step 5 if the augmented manifest is still current).
