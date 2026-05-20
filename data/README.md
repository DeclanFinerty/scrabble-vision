# Data Pipeline

Training data for the Scrabble Vision corner-detection and tile-extraction models.
Everything under `data/` is git-ignored except this README.

## Directory structure

```
data/
  raw/
    boards/      # Raw board photos (JPEG/PNG)
    racks/       # Raw rack photos (JPEG/PNG)
  annotations/   # Corner and label annotations (JSON/CSV)
  extracted/
    tiles/       # Cropped & warped 15x15 tile images
  splits/        # train/val/test split manifests
```

## Three-stage pipeline

### 1. Corner annotation
Raw board images in `raw/boards/` are annotated with four corner points
that define the board quadrilateral. Annotations are stored in
`annotations/`.

### 2. Ground-truth labeling
Each annotated board gets a 15x15 ground-truth grid that records the
letter (or blank) on every square. These labels are also kept in
`annotations/`.

### 3. Tile extraction
Using the corner coordinates, the board image is perspective-warped to a
square and sliced into 225 individual tile images saved to
`extracted/tiles/`. The `splits/` directory holds train/val/test
manifests that reference these tiles and their labels.
