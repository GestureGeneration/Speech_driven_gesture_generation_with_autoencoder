# How to use the helper scripts

This directory provides data handling scripts for our gesture generation framework. It provides the following functionality:
- Velocity remover for predicted gestures

(The neural network outputs coordinates and velocities to regularize training and we remove velocities as postprocessing)
- Original gesture converter to create the ground truth

(Converting the original motion for joint angles space in .bvh format to 3d coordinates in txt coordinates)
- Temporal filters for motion smoothing

(Can be applied as postprocessing. Were not used in the experiments from the paper)

## Data preparation 
  1. Run `../predict.py` to predict gestures from speech audio as described in the root folder.
  2. Put the predicted gestures (e.g. `predict_1094.txt, ...`) into a directory, say, `your_prediction_dir/`.

### Velocity remover

`remove_velocity.py` removes velocities from raw predicted gestures. This produces gesture files containing `(x, y, z) x 64 joints = 192` white space separated data for each line. 
**You have to remove the velocities before using the evaluation scripts or the animation server.**

```sh
# Remove velocities
python remove_velocity.py -g your_prediction_dir
```
The resulting files will be stored in the subfolder: `your_prediction_dir/no_vel`

### Original gesture converter

`convert_original.py` converts `.bvh` files in the test set to ground truth body keypoint positions. **You need the ground truth for the quantitative evaluation.**

```sh
# Convert test bvh to ground truth
python convert_original.py
```

Note: `convert_original.py` assumes that the `.bvh` files are stored in `../data/test/labels/` by default. You can use `--data` or `-d` option to specify a different directory. You can specify the output directory by `--out` or `-o` option (default: `../evaluation/data/original/`).

### Temporal filters

We support two types of temporal filters, 1€ filter and Simple Moving Average (SMA) filter, to smooth gesture motion.

To apply filters, you can use `apply_filters.py`.
You can change the averaging window size for SMA filter by `--window` or `-w` option (default: 5).

```sh
# Apply temporal filters
python apply_filters.py -g your_prediction_dir -w 5
```

Note: `apply_filters.py` produces three types of smoothed gestures (1€, SMA, and 1€ + SMA). The smoothed gestures will be stored in `euro/`, `sma/`, and `euro_sma/` subfolders of `your_prediction_dir/`.
