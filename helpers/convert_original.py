# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../data_processing'))

import numpy as np
import pyquaternion as pyq
from bvh_read.BVH import load
from bvh_read.Animation import Animation
from bvh_read.BVH_io import bvh2npy


def rotation_to_position(frames, nodes):
    """Convert bvh frames to body keypoint positions

      Args:
          frames:       bvh frames
          nodes:        bvh hierarchy nodes

      Returns:
          out_data:     array containing body keypoint positions
    """

    output_lines = []

    for frame in frames:
        node_idx = 0
        for i in range(51):
            stepi = i * 3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi + 1])
            y_deg = float(frame[stepi + 2])

            if nodes[node_idx]['name'] == 'End Site':
                node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            if start_node['children'] is not None:
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])

                    qz = pyq.Quaternion(axis=[0, 0, 1],
                                        degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(axis=[1, 0, 0],
                                        degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(axis=[0, 1, 0],
                                        degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos'] = start_node['abs_qt'].rotate(
                        offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:
                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)

    output_array = np.asarray(output_lines)
    out_data = np.empty([len(output_array), 192])
    for idx, line in enumerate(output_array):
        out_data[idx] = line.flatten()

    return out_data


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