# -*- coding: utf-8 -*-
"""
Calculating average jerk over the produced and ground truth gestures

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import warnings

import numpy as np


def read_joint_names(filename):
    """Read motion capture's body joint names from file

      Args:
          filename:     file name to read

      Returns:
          joint_names:  list of joint names
    """

    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names


def compute_jerks(data, dim=3):
    """Compute jerk between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   jerks of each joint averaged over all frames
    """

    # Third derivative of position is jerk
    jerks = np.diff(data, n=3, axis=0)

    num_jerks = jerks.shape[0]
    num_joints = jerks.shape[1] // dim

    jerk_norms = np.zeros((num_jerks, num_joints))

    for i in range(num_jerks):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            jerk_norms[i, j] = np.linalg.norm(jerks[i, x1:x2])

    return np.mean(jerk_norms, axis=0)


def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   accelerations of each joint averaged over all frames
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    return np.mean(acc_norms, axis=0)


def save_result(lines, out_dir, measure):
    """Write computed measure to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          measure:      used measure
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if measure == "jerk":
        outname = os.path.join(out_dir, 'aj.csv')
    elif measure == "acceleration":
        outname = os.path.join(out_dir, 'aa.csv')

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)

    print('More detailed result was writen to the file: ' + outname)
    print('')


def main():
    measures = {
        'jerk': compute_jerks,
        'acceleration': compute_acceleration,
    }

    parser = argparse.ArgumentParser(
        description='Calculate prediction errors')
    parser.add_argument('--original', '-o', default='data/original',
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default='data/predicted',
                        help='Predicted gesture directory')
    parser.add_argument('--joints', '-j', default='joints.txt',
                        help='Joint name file')
    parser.add_argument('--gesture', '-g', required=True,
                        help='Directory storing predicted txt files')
    parser.add_argument('--measure', '-m', default='jerk',
                        help='Measure to calculate (jerk or acceleration)')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    predicted_dir = os.path.join(args.predicted, args.gesture)

    original_files = sorted(glob.glob(os.path.join(args.original, '*.txt')))
    predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.txt')))

    # Check number of files
    if len(original_files) != len(predicted_files):
        warnings.warn('Inconsistent number of files : {} vs {}'
                      ''.format(len(original_files), len(predicted_files)),
                      RuntimeWarning)

    # Check if error measure was correct
    if args.measure not in measures:
        raise ValueError('Unknown measure: \'{}\'. Choose from {}'
                         ''.format(args.measure, list(measures.keys())))

    joint_names = read_joint_names(args.joints)

    if args.select is not None:
        selected_joints = []
        for s in args.select:
            try:
                index = joint_names.index(s)
            except ValueError:
                print('Ignore invalid joint: {}'.format(s))
            else:
                selected_joints.append(index)
        selected_joints.sort()

        if len(selected_joints) == 0:
            selected_joints = range(len(joint_names))
            print('No valid joints are selected. Use all joints')
    else:
        # Use all joints
        selected_joints = range(len(joint_names))

    joint_names = [joint_names[s] for s in selected_joints]
    original_out_lines = [','.join(['file'] + joint_names) + '\n']
    predicted_out_lines = [','.join(['file'] + joint_names) + '\n']

    original_values = []
    predicted_values = []
    for original_file, predicted_file in zip(original_files, predicted_files):
        original = np.loadtxt(original_file)
        predicted = np.loadtxt(predicted_file)

        if original.shape[0] != predicted.shape[0]:
            # Cut them to the same length
            length = min(original.shape[0], predicted.shape[0])
            original = original[:length]
            predicted = predicted[:length]

        original_value = measures[args.measure](original)[selected_joints]
        predicted_value = measures[args.measure](predicted)[selected_joints]

        original_values.append(original_value)
        predicted_values.append(predicted_value)

        basename = os.path.basename(original_file)
        original_line = basename
        predicted_line = basename
        for ov, pv in zip(original_value, predicted_value):
            original_line += ',' + str(ov)
            predicted_line += ',' + str(pv)
        original_line += '\n'
        predicted_line += '\n'

        original_out_lines.append(original_line)
        predicted_out_lines.append(predicted_line)

    original_average_line = 'Average'
    predicted_average_line = 'Average'
    original_avgs = np.mean(original_values, axis=0)
    predicted_avgs = np.mean(predicted_values, axis=0)
    for oa, pa in zip(original_avgs, predicted_avgs):
        original_average_line += ',' + str(oa)
        predicted_average_line += ',' + str(pa)

    original_out_lines.append(original_average_line)
    predicted_out_lines.append(predicted_average_line)

    original_out_dir = os.path.join(args.out, 'original')
    predicted_out_dir = os.path.join(args.out, args.gesture)

    save_result(original_out_lines, original_out_dir, args.measure)
    save_result(predicted_out_lines, predicted_out_dir, args.measure)

    if args.measure == 'jerk':
        print('AJ:')
    elif args.measure == 'acceleration':
        print('AA:')
    print('original: {:.2f}'.format(np.mean(original_values)))
    print('predicted: {:.2f}'.format(np.mean(predicted_values)))


if __name__ == '__main__':
    main()
