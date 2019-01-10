# How to use the evaluation script

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We support the following measures:
- Average Position Error (APE)
- Mean Absolute Error (MAE)
- Average Jerk (AJ)
- Average Acceleration (AA)
- Histogram of Moving Distance (HMD, for velocity/acceleration)

## Data preparation 
  1. Use `../helpers/remove_velocity.py` to delete velocities from predicted data.
  2. Use `../helpers/convert_original.py` to create original data.

This produces gesture files containing `(x, y, z) x 64 joints = 192` white space separated data for each line.

  3. (optional) Use `../helpers/apply_filters.py` to smooth predicted data.

## Directory organization

We assume original/predicted gesture data are stored as follows:

```
-- evaluation/
      |-- calc_distance.py
      |-- calc_errors.py
      |-- calc_jerk.py
      |-- joints.txt
      |-- data/
           |-- original/
                  |-- gesture1093.txt, gesture1095.txt, ...
           |-- predicted/
                  |-- your_prediction_dir/
                        |-- gesture1093.txt, gesture1095.txt, ...
```

**Important Note: You have to store the gesture files of the same indices in `original` and `predicted` directories.
If you have gestures 1093, 1095, ... in the `original` directory, but gestures 1094, 1096, ... in the `predicted' - you will get wrong results**

## Run

`calc_errors.py`, `calc_jerk.py`, and `calc_distance.py` support different quantitative measures, described below.

`--gesture` or `-g` option specifies the predicted directory under `data/predicted`. If you store the predicted gesture files in `data/predicted/your_prediction_dir/`, use `-g your_prediction_dir`.

### APE/MAE

Average Position Error (APE) and Mean Absolute Error (MAE) indicate the prediction errors against the original gestures.

To calculate APE/MAE, you can use `calc_errors.py`.
You can select the metric to compute by `--metric` or `-m` option (default: ape).

```sh
# Compute APE
python calc_errors.py -g your_prediction_dir -m ape

# Compute MAE
python calc_errors.py -g your_prediction_dir -m mae
```

### AJ/AA

Average Jerk (AJ) and Average Acceleration (AA) represent the characteristics of gesture motion.

To calculate AJ/AA, you can use `calc_jerk.py`.
You can select the measure to compute by `--measure` or `-m` option (default: jerk).

```sh
# Compute AJ
python calc_jerk.py -g your_prediction_dir -m jerk

# Compute AA
python calc_jerks.py -g your_prediction_dir -m acceleration
```

Note: `calc_jerk.py` computes AJ/AA for both original and predicted gestures. The AJ/AA of the original gestures will be stored in `result/original` by default. The AJ/AA of the predicted gestures will be stored in `result/your_prediction_dir`.

### HMD

Histogram of Moving Distance (HMD) shows the velocity/acceleration distribution of gesture motion.

To calculate HMD, you can use `calc_distance.py`.
You can select the measure to compute by `--measure` or `-m` option (default: velocity).  
In addition, this script supports histogram visualization. To enable visualization, use `--visualize` or `-v` option.

```sh
# Compute velocity histogram
python calc_distance.py -g your_prediction_dir -m velocity -w 0.05  # You can change the bin width of the histogram

# Compute acceleration histogram
python calc_distance.py -g your_prediction_dir -m acceleration -w 0.05
```

Note: `calc_distance.py` computes HMD for both original and predicted gestures. The HMD of the original gestures will be stored in `result/original` by default.

### Calculate evaluation measures for specific joints  
You can use `-s` option for all evaluation scripts to select specific joints, e.g. `-s Head LeftLeg RightLeg`  
Here is a table for the joint names:

| Joint to Calculate | Corresponding Name |
| --- | --- |
| Head | Head |
| Neck | Neck |
| Left Shoulder | LeftArm |
| Left Elobow | LeftForeArm |
| Left Wrist | LeftHand |
| Right Shoulder | RightArm |
| Right Elobow | RightForeArm |
| Right Wrist | RightHand |
| Left Hip | LeftUpLeg |
| Left Knee | LeftLeg |
| Left Ankle | LeftFoot |
| Right Hip | RightUpLeg |
| Right Knee | RightLeg |
| Right Ankle | RightFoot |

When you calculate the velocity histogram for both elbows, use
```sh
python calc_distance.py -g your_prediction_dir -m velocity -w 0.05 -s LeftForeArm RightForeArm
```
