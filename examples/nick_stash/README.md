# Nick's Stash

## Scripts

This folder contains a bunch of scripts and work from my MEng work. A lot of it uses older versions of models and may not work directly with the latest infrastructure versions.

The `old_scripts` folder contains code used to interact with the SPAD for the display case demo and other preliminary explorations. This code generally uses the example driver code (before the SPAD standardized infrastructure was created).

The `newer_scripts` folder contains code used to interact with the SPAD for various other demos and generally uses the newer SPAD control interfaces included in the `cc-hardware` repo. This folder includes a few versions of data capture pipelines, training notebooks, demos, and other miscellaneous content. Backprojection work is accessible in `newer_scripts/backprojection.ipynb`.

## Datasets

A small selection of captured datasets are included in `datasets`. Dataset collection data is as follows:

### Arrow demo data

This data was used to train the arrow demo model in `cc-hardware/examples/localization_demo`. This data was also used for backprojection work.

`datasets/demo_arrow/`: VL53L8CH 8x8 capture, sensor.integration_time_ms=100 sensor.cnh_num_bins=16 sensor.cnh_subsample=2

`datasets/demo_arrow/first_bouncezero.pkl`: sensor.cnh_start_bin=0, 10 samples x 100 locations (total 1000 samples)
`datasets/demo_arrow/zero.pkl`: sensor.cnh_start_bin=24, 5 samples x 100 locations (NO OBJECT) (total 500 samples)
`datasets/demo_arrow/capture_retro_mini_1.pkl`: 10 samples x 100 locations grid (total 1000 samples)
`datasets/demo_arrow/capture_retro_1.pkl`: 100 samples x 100 locations grid (total 10000 samples)
`datasets/demo_arrow/debug.pkl`: 4x4 capture, moving first-bounce retro-reflective patch from top left to top right to bottom right to bottom left to top left (used for debugging SPAD orientation)

### Classification demo data

This is newer data captures (not used in the original development of the display box demo, but recaptured after infrastructure upgrades). This data was used to train the classification demo model included in `cc-hardware/examples/classification_demo`.

`datasets/classification-demo/`: new captures for window demo: VL53L8CH 4x4 capture, sensor.integration_time_ms=100 sensor.cnh_subsample=3 sensor.cnh_start_bin=16 sensor.cnh_num_bins=16

`datasets/classification-demo/training_2.pkl` 1000 samples each: zero, zone_1, zone_2, zone_3 (total 4000 samples)
`datasets/classification-demo/mini_2`: 200 samples each: zero_mini, zone_1_mini, zone_2_mini, zone_3_mini (total 800 samples)
