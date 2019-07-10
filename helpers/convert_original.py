# -*- coding: utf-8 -*-
"""
Convert ground truth gestures from joint angles in bvh format to the 3d coordinates in text format

@author: kaneko.naoshi
"""


import argparse
import glob
import os

import numpy as np
import pyquaternion as pyq


def create_hierarchy_nodes(filename):
    """Load bvh hierarchy nodes

      Args:
          filename:     name of the hierarchy file

      Returns:
          nodes:        bvh hierarchy nodes
    """

    # Read BVH hierarchy
    with open(filename, 'r') as f:
        hierarchy = f.readlines()

    joint_offsets = []
    joint_names = []

    for idx, line in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()

        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == 'OFFSET':
                offset = np.array([float(hierarchy[idx][1]),
                                   float(hierarchy[idx][2]),
                                   float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(hierarchy[idx][1])
            elif line_type == 'End':
                joint_names.append('End Site')

    nodes = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            parent = None
        elif idx in [6, 30]:  # spine1->shoulders
            parent = 2
        elif idx in [14, 18, 22, 26]:  # lefthand->leftfingers
            parent = 9
        elif idx in [38, 42, 46, 50]:  # righthand->rightfingers
            parent = 33
        elif idx in [54, 59]:  # hip->legs
            parent = 0
        else:
            parent = idx - 1

        if name == 'End Site':
            children = None
        elif idx == 0:  # hips
            children = [1, 54, 59]
        elif idx == 2:  # spine1
            children = [3, 6, 30]
        elif idx == 9:  # lefthand
            children = [10, 14, 18, 22, 26]
        elif idx == 33:  # righthand
            children = [34, 38, 42, 46, 50]
        else:
            children = [idx + 1]

        node = dict([('name', name), ('parent', parent),
                     ('children', children), ('offset', joint_offsets[idx]),
                     ('rel_degs', None), ('abs_qt', None),
                     ('rel_pos', None), ('abs_pos', None)])
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes


def load_bvh(filename):
    """Load bvh motion frames

      Args:
          filename:     bvh filename

      Returns:
          frames:       list of bvh frames
    """

    with open(filename, 'r') as f:
        frames = f.readlines()
        frametime = frames[310].split()[2]

    del frames[0:311]
    bvh_len = len(frames)

    for idx, line in enumerate(frames):
        frames[idx] = [float(x) for x in line.split()]

    for i in range(0, bvh_len):
        for j in range(0, 306 // 3):
            st = j * 3
            del frames[i][st:st + 3]

    # If data is approx 24fps, cut it to 20 fps (del every sixth line)
    # If data is 100fps, cut it to 20 fps (take every fifth line)
    if float(frametime) == 0.0416667:
        del frames[::6]
    elif float(frametime) == 0.010000:
        frames = frames[::5]
    else:
        print('Unsupported fps {} in {}'.format(frametime, filename))

    return frames


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

    # Read bvh hierarchy
    nodes = create_hierarchy_nodes('../hierarchy.txt')

    # Make output directories
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    for bvh_path in bvh_paths:
        print('Process "{}"'.format(bvh_path))
        frames = load_bvh(bvh_path)

        out_data = rotation_to_position(frames, nodes)
        gesture_name, _ = os.path.splitext(os.path.basename(bvh_path))
        out_path = os.path.join(args.out, gesture_name + '.txt')
        np.savetxt(out_path, out_data, fmt='%s')

    print('')
    print('Results were written in "{}"'.format(args.out))
    print('')


if __name__ == '__main__':
    main()
