# TODO -- deferred / future work

Items here are explicitly **not** in scope for the current corner
detection training push. They're parked so they don't get lost.

## Tile classification (deferred)

- Ground-truth labeling for the remaining boards (stage 2 of the
  labeling pipeline -- acts on `corners_verified` boards only).
- Tile extraction pipeline rewrite against the new
  `data/raw/boards/` + `data/annotations/` + `data/extracted/tiles/`
  layout.
- Retrain the tile CNN with the expanded dataset once extraction is
  back online.

## Corner detection -- follow-ups

- Train a from-scratch CNN for corner detection and compare against
  MobileNetV2 transfer learning (does the pretraining actually help
  at our dataset size?).
- Experiment with model compression: pruning, quantization,
  knowledge distillation. Target is faster CPU inference without
  giving up the 0.5% accuracy budget.
- Investigate grokking and minimum-training-set-size experiments
  (how few labeled boards can we get away with?).
- ONNX export for the trained corner detection model (mirror the
  tile classifier's `.pt` + `.onnx` pairing).
- Rust inference server for corner detection (latency / deploy
  target separate from the Python FastAPI server).
- Automated evaluation comparing heuristic `auto_detect_corners`
  vs the trained model on the same test set, so we can track the
  delta over time rather than eyeballing it.
