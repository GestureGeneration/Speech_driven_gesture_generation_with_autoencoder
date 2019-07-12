# -*- coding: utf-8 -*-
"""
Apply smoothing filters as postprocessing

@author: kaneko.naoshi
"""


import argparse
import glob
import os

import numpy as np

from filters.ma_filter import simple_moving_average
from filters.one_euro_filter import apply_one_euro


def save_positions(out_dir, gesture_name, positions):
    """Save body keypoint positions into file

      Args:
          out_dir:      output directory
          gesture_name: basename of the output file
          positions:    keypoint positions to save
    """

    filename = os.path.join(out_dir, gesture_name + '.txt')
    np.savetxt(filename, positions, fmt='%s')


def main():
    parser = argparse.ArgumentParser(
        description='Apply filters to the generated gestures')
    parser.add_argument('--gesture', '-g', required=True,
                        help='Path to the gesture directory to filter')
    parser.add_argument('--window', '-w', type=int, default=5,
                        help='Windows size for moving average (must be odd)')
    args = parser.parse_args()

    print('Apply temporal filters to the '
          'gestures in "{}"'.format(args.gesture))
    print('')

    # List of gesture files
    txt_paths = sorted(glob.glob(os.path.join(args.gesture, '*.txt')))

    # Check file existence
    if not txt_paths:
        raise ValueError('Could not find the gesture files in "{}". '
                         'Please specify correct folder as --gesture flag.'
                         .format(args.gesture))

    # Filter types
    types = {
        'euro': 'euro',
        'sma': 'sma{}'.format(args.window),
        'euro_sma': 'euro_sma{}'.format(args.window)}

    # Make output directories
    euro_dir = os.path.join(args.gesture, types['euro'])
    sma_dir = os.path.join(args.gesture, types['sma'])
    euro_sma_dir = os.path.join(args.gesture, types['euro_sma'])
    for d in [euro_dir, sma_dir, euro_sma_dir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    for txt_path in txt_paths:
        print('Process "{}"'.format(txt_path))

        raw_pos = np.loadtxt(txt_path)

        # One Euro filter
        euro_pos = apply_one_euro(raw_pos)

        # Moving average filter
        sma_pos = simple_moving_average(raw_pos, args.window)

        # Combined
        euro_sma_pos = simple_moving_average(euro_pos, args.window)

        gesture_name, _ = os.path.splitext(os.path.basename(txt_path))
        save_positions(euro_dir, gesture_name, euro_pos)
        save_positions(sma_dir, gesture_name, sma_pos)
        save_positions(euro_sma_dir, gesture_name, euro_sma_pos)

    print('')
    print('Results were written under "{}"'.format(args.gesture))
    print('')


if __name__ == '__main__':
    main()
