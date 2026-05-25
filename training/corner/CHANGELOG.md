# Corner regressor -- run changelog

What changed between training runs of `scripts/train_corners.py`,
newest first. Each `run_NNN/` directory under `runs/` is a frozen
snapshot of one experiment (config, weights, metrics, log).

# run_016 -- testing batch sizes
- run batch size 4 with 40 augementation each first
- error trajectory was more noisy

## run_015 -- doubled augmentation to 40 each
- best accuracy on validation was 2.7%

## run_014 -- updated data
- added two large datasets
- changed augmentation to 20 each, got below 5% error

## run_013 -- trimmed validation and poolsize
- removed unique boards from validation, only grammy's board shots
- fixed poolsize to 7 because of mobilenet 7x7

## run_012 -- added new dataset
- added scrabble_dataset2
- same config as 011 and 010, slightly better but validation unique values had changed

## run_011 -- remove some more boards and added color-only
- after removing the worst boards (warped, lifesize and other) the loss improved on validation set
- ran same set as 010

## run_010 -- added color-only augmentation
- labeled certain data that was already warped to be color only
    - may be more stringent with this afterwards, right now only 5 are color-only
- reran with PS -1, LR1e-4, ES 15

## run_009 -- testing LR
- 1e-5 and 25 LR and ES
- worse than 008 but better tha 007

## run_008 -- all unfrozen
- much better validation loss than with 3 unfrozen

## run_007 -- pool size 8
- increased pool size dim to 8
- reverted back to 15 early stopping, 1e-4 LR, and 3 unfrozen

## run_006 -- kept long early stopping, went back to all unfrozen
- lasting much longer.  Mean Error px is still high (650-750) and looks to be overfitting in the training loss
- validation loss not changing much

## run_005 -- longer early stopping
- kept 3 unfrozen, lowered learning rate and increased early stopping
- still ended very early on epoch 27

## run_004 -- unfreeze layers
- tried unfreezing only the last 3 layers.  Seemed to be worse learning and ended after 19

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
