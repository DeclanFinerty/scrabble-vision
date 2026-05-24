# Corner regressor -- run changelog

What changed between training runs of `scripts/train_corners.py`,
newest first. Each `run_NNN/` directory under `runs/` is a frozen
snapshot of one experiment (config, weights, metrics, log).

## run_003 -- architectural fixes
- Fixed val letterboxing (was squishing non-square images, causing train/val distribution mismatch)
- Changed head from AdaptiveAvgPool2d(1) to AdaptiveAvgPool2d((4,4)) -- global pooling destroyed spatial info needed for coordinate regression
- Unfroze backbone (freeze_backbone: false)
- LR: 1e-4

## run_002 -- lower LR
- Dropped LR from 3e-4 to 1e-4, same architecture as run_001
- Marginal improvement (556px -> 502px), plateau remained

## run_001 -- baseline
- Frozen MobileNetV2, LR 3e-4, global avg pool
- 556px mean error, model learned average corner positions but couldn't refine spatially
