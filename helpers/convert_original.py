# -*- coding: utf-8 -*-
"""
Convert ground truth gestures from joint angles in bvh format to the 3d coordinates in text format

@author: kaneko.naoshi
"""


import argparse
import glob
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../data_processing'))

import numpy as np
from bvh_read.BVH_io import bvh2npy

def main():
    parser = argparse.ArgumentParser(
        description='Convert original motion data into joint positions')
    parser.add_argument('--data', '-d', default='../data/test/labels',
                        help='Path to the original test motion data directory')
    parser.add_argument('--out', '-o', default='../evaluation/data/original',
                        help='Directory to store the resultant position files')
    args = parser.parse_args()

    print('Convert original gestures to the ground truth')
    if args.data != parser.get_default('data'):
        print('Warning: non-default original gesture directory is given: '
              + args.data)
    print('')

    # List of bvh files
    bvh_paths = sorted(glob.glob(os.path.join(args.data, '*.bvh')))

    # Check file existence
    if not bvh_paths:
        raise ValueError(
            'Could not find the ground truth bvh files in "{}". '
            'Please specify correct folder as --data flag.'.format(args.data))


    # Make output directories
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    main_joints = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',  # Head and spine
                   'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1',
                   'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
                   'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing1', 'RightHandRing2',
                   'RightHandRing3', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3',  # Right hand
                   'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1',
                   'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
                   'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing1', 'LeftHandRing2',
                   'LeftHandRing3', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',  # left hand
                   ]

    for bvh_path in bvh_paths:
        print('Process "{}"'.format(bvh_path))
        out_data = bvh2npy(bvh_path, main_joints, hips_centering=True)
        gesture_name, _ = os.path.splitext(os.path.basename(bvh_path))
        out_path = os.path.join(args.out, 'gesture'+gesture_name[-1] + '.txt')
        np.savetxt(out_path, out_data, fmt='%s')

    print('')
    print('Results were written in "{}"'.format(args.out))
    print('')


if __name__ == '__main__':
    main()