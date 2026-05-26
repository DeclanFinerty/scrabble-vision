# Dense corner regressor -- run changelog

What changed between training runs of `scripts/train_corner_dense.py`,
newest first. Each `run_NNN/` directory under `runs/` is a frozen
snapshot of one experiment (config, weights, metrics, log).

The dense regressor predicts all 256 intersections of the 15x15
Scrabble grid (16x16 lattice). Per-corner error at indices
[0, 15, 240, 255] is reported for comparison against the 4-corner
runs in `training/corner/CHANGELOG.md`.

# run_005 -- double aug
- 80 augmentations
- 1.7% validation. 
- few outliers in test >10% error, but majority are around 1-2% error
    - continuous scanning will fix this

## run_004 -- remove grid regularity
- grid_reg loss has a technical problem.  If an image is warped and the points perfectly match up, there will be error because the distances are not equal.  
- rerun without grid reg, and upped the collinearlity to 0.2
- 2.17% corner error

## run_003 -- add dropout
- add dropout 0.3
- 2.1% corner

## run_002 -- changed test split and added loss regularizer
- decided to change test split to swap out dark bacground board for common boards with very specific backgrounds
    - felt like trying to classify a class never seen before, with basic background.  This approach is better.
    - uv run scripts/make_splits.py --seed 42 --test-groups F,AH,G --val-groups E,H --val-unique-count 0 
- added collinearity regularization.  Adds penalty for straight lines
- same params as before
- 1.45% corner error, and gernalizes to test better.  Won't check test again until totally happy with validation

## run_001 -- initial run
- removed adaptiveavgpool of 7, just scaling down to a vector now.  
- can re-evaluate whether adding this back in helps
- removed the item from the config
- low error in validation (1.26%), higher in test
    - checked test as a mistake