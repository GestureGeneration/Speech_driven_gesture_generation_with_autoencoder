# -*- coding: utf-8 -*-
"""
Remove velocity from the network output
(it produces both coordinates and velocities while we need only velocities)

@author: kaneko.naoshi
"""

import argparse
import glob
import os

import numpy as np


def save_positions(out_dir, gesture_name, positions):
    """Save body keypoint positions into file

      Args:
          out_dir:      output directory
          gesture_name: basename of the output file
          positions:    keypoint positions to save
    """

    filename = os.path.join(out_dir, gesture_name + '.txt')
    np.savetxt(filename, positions, fmt='%s')


def remove_velocity(data, dim=3):
    """Remove velocity values from raw prediction data

      Args:
          data:         array containing both position and velocity values
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   array containing only position values
    """

    starts = np.arange(0, data.shape[1], dim * 2)
    stops = np.arange(dim, data.shape[1], dim * 2)
    return np.hstack([data[:, i:j] for i, j in zip(starts, stops)])


def main():
    parser = argparse.ArgumentParser(
        description='Remove velocity values from the raw generated gestures')
    parser.add_argument('--gesture', '-g', required=True,
                        help='Path to the raw gesture directory')
    args = parser.parse_args()

    print('Remove velocities from the '
          'gestures in "{}"'.format(args.gesture))
    print('')

    # List of gesture files
    txt_paths = sorted(glob.glob(os.path.join(args.gesture, '*.txt')))

    # Check file existence
    if not txt_paths:
        raise ValueError('Could not find the gesture files in "{}". '
                         'Please specify correct folder as --gesture flag.'
                         .format(args.gesture))

    # Make output directory
    out_dir = os.path.join(args.gesture, 'no_vel')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for txt_path in txt_paths:
        print('Process "{}"'.format(txt_path))

        pos_vel = np.loadtxt(txt_path)

        # Remove velocity values
        only_pos = remove_velocity(pos_vel)

        gesture_name, _ = os.path.splitext(os.path.basename(txt_path))
        save_positions(out_dir, gesture_name, only_pos)

    print('')
    print('Results were written in "{}"'.format(out_dir))
    print('')


if __name__ == '__main__':
    main()
