# Corner regressor -- run changelog

What changed between training runs of `scripts/train_corners.py`,
newest first. Each `run_NNN/` directory under `runs/` is a frozen
snapshot of one experiment (config, weights, metrics, log).

## run_034 -- -- tested unfreezing layers again
- tested layer unfreezing to see if it maintained earlier worse results
- 

## run_033 -- tested seed 
- ran run_026 parameters, but with seed 123 (not aug_seed) to test replicability
- 5% error

## run_032 -- tested aug size
- redued aug to 20 from 40
- 100% prob, 3 corners, replicate boundary
- 3.2%

## run_031 -- tested constant fill
- tested constant fill for 3 corners with 40, 100% prob
- 5.4%

## run_030  -- redo earlier runs with constant
- this should be comapared to 25
- 40 aug, 100% and 2 corners
- changed replicate borders to black box constant
- 4.42

## run_029 -- lowered aug
- aug 20
- crop prob 50, min corners 2
- 3.8%

## run_028 -- keep low corner prob, increase corners
- crop prob at 50%, minimum corners at 3.  Aug 40
- 2.68%

## run_027 -- change crop prob
- used minimum corners=2, changed crop prob to 50% from 100%
- 40 aug
- 4.53 error

## run_026 -- chnage minimum corners
- changed minimum corners to 3 from 2
- 40 aug, 100% prob
- 2.5%

## run_025 -- rerun with 40aug
- 40 aug
- have been using 30 count from the default, this changed once we started loading the augmentations inside the training script.  Rerun same config with 40 instead to gauge difference
- 5.21% error

## run_024 -- replicate bounds 
- replicate bounds added so there aren't black boxes
- still 100% cropping
- likely uses 30 instead of the 40, realized train_corners was using a default of 30 because it is called inside
- 2.8%

## run_023 -- original validation with new data
- went back to grammy's boards as validation
- makes sense because likely board pictures

## run_022 -- different validation
- ranwith red boards as validation
- worse as well

## run_021 -- add data
- we added the final solid online dataset (includes shadows)
- also changed the group in the validation set.  Grammy's board group had large outlier that would be better for training, and my board group has more images
- poor validation loss

## run_020 -- dropout increase
- increase dropout to 0.5
- hurt training, switching back to 0.3

## run_019 -- reverted aug. magnitude and increase volume
- increased to 80 augmented images
-- slighlty worse than 40

## run _018 -- tweaked magnitude of cropping augmentation 
- added augmentation params to config
- ran with less cropping, config was saved
- less cropping produced slightly worse results

## run_017 -- testing larger batch size
- run batch size 16

## run_016 -- testing batch sizes
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
