# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:58:35 2020

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


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


def compute_speed(data, dim=3):
    """Compute speed between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          speeds:       velocities of each joint between each adjacent frame
    """

    # First derivative of position is velocity
    vels = np.diff(data, n=1, axis=0)

    num_vels = vels.shape[0]
    num_joints = vels.shape[1] // dim

    speeds = np.zeros((num_vels, num_joints))

    for i in range(num_vels):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            speeds[i, j] = np.linalg.norm(vels[i, x1:x2])

    return speeds


def normalize(hist):
    return hist / np.sum(hist)


def hellinger(hist1, hist2):
    """Compute Hellinger distance between two histograms

      Args:
          hist1:        first histogram
          hist2:        second histogram of the same size as hist1

      Returns:
          float:        Hellinger distance between hist1 and hist2
    """

    return np.sqrt(1.0 - np.sum(np.sqrt(normalize(hist1) * normalize(hist2))))


# https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort  # NOQA
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def natural_sort(l, key=natural_sort_key):
    return sorted(l, key=key)


# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list  # NOQA
def reject_outliers(data, m=5.189):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def main():
    parser = argparse.ArgumentParser(
        description='Compute Hellinger distances between predicted '
                    'and ground truth gestures in a one-to-one manner')
    parser.add_argument('--original', '-o', default='data/original',
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default='data/predicted',
                        help='Predicted gesture directory')
    parser.add_argument('--width', '-w', type=float, default=0.05,
                        help='Bin width of the histogram (default: 0.05)')
    parser.add_argument('--joints', '-j', default='joints.txt',
                        help='Joint name file')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--match_yticks', '-m', action='store_true',
                        help='Match y-ticks over all the sequences in visualization')
    parser.add_argument('--out', default='results',
                        help='Directory to output the result')
    args = parser.parse_args()

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

    def get_directories(directory):
        return sorted(filter(lambda x: os.path.isdir(x), glob.glob(directory)))

    # Define histogram bins
    bins = np.arange(0, 1 + args.width, args.width)

    # Find original gesture data
    original_files = natural_sort(
        glob.glob(os.path.join(args.original, '*.txt')))
    
    if args.match_yticks:
        max_freqs = []

    # Compute speed histogram for original gestures
    original_hists = []
    for original_file in original_files:
        original = np.loadtxt(original_file)

        # Compute speed histogram
        original_speed = compute_speed(original)[:, selected_joints]
        original_hist, _ = np.histogram(original_speed, bins=bins)

        original_hists.append(original_hist)

        if args.match_yticks:
            max_freqs.append(normalize(original_hist).max().item())

    # List of predicted gesture direcotires
    predicted_dirs = get_directories(os.path.join(args.predicted, '*'))

    if len(predicted_dirs) == 0:
        raise ValueError('No gesture directories are found in '
                         + args.predicted)

    results = {os.path.basename(d): None for d in predicted_dirs}

    assert 'original' not in results.keys()

    # Store original gesture histograms
    original_key = 'original'
    results[original_key] = dict()
    for i, original_hist in enumerate(original_hists):
        file_key = os.path.basename(original_files[i])
        results[original_key][file_key] = {'hist': original_hist}

    # Iterate over the list of direcotires
    overall_dists = dict()
    for predicted_dir in predicted_dirs:
        predicted_files = natural_sort(
            glob.glob(os.path.join(predicted_dir, '*.txt')))

        # Check if the predicted gesture files are consistent with the original files
        if [os.path.basename(p) for p in predicted_files] != [os.path.basename(o) for o in original_files]:
            raise ValueError('Gesture files located in ' + predicted_dir + ' are inconsistent with '
                             'original gesture files located in ' + args.original)

        dir_key = os.path.basename(predicted_dir)
        results[dir_key] = dict()

        # Compute speed histogram for predicted gestures
        predicted_hists = []
        for predicted_file in predicted_files:
            predicted = np.loadtxt(predicted_file)

            # Compute speed histogram
            predicted_speed = compute_speed(predicted)[:, selected_joints]
            predicted_hist, _ = np.histogram(predicted_speed, bins=bins)

            predicted_hists.append(predicted_hist)

            if args.match_yticks:
                max_freqs.append(normalize(predicted_hist).max().item())

        assert len(original_hists) == len(predicted_hists)

        # Compute Hellinger distance in a one-to-one manner
        for i, (original_hist, predicted_hist) in enumerate(zip(original_hists, predicted_hists)):
            assert len(original_hist) == len(predicted_hist)

            # Hellinger distance between two histograms
            dist = hellinger(original_hist, predicted_hist)

            # Store results
            file_key = os.path.basename(predicted_files[i])
            results[dir_key][file_key] = {'dist': dist, 'hist': predicted_hist}

        # Print the overall Hellinger distance (Note: this is not one-to-one)
        overall_dist = hellinger(np.sum(original_hists, axis=0),
                                 np.sum(predicted_hists, axis=0))
        overall_dists[dir_key] = overall_dist

    # Create a dataframe to save
    dir_keys = natural_sort(results.keys())
    dir_keys.remove('original')
    file_keys = natural_sort(results['original'].keys())

    save_dict = {d_k: [results[d_k][f_k]['dist'] for f_k in file_keys] for d_k in dir_keys}
    df = pd.DataFrame(save_dict, index=file_keys)

    # Add mean and std values
    mean = df.mean()
    std = df.std()
    df.loc['mean'] = mean
    df.loc['std'] = std

    # Make an output directory
    if selected_joints == range(len(joint_names)):
        selected_joint_names = ['Total']
    else:
        selected_joint_names = [joint_names[s] for s in selected_joints]
    out = os.path.join(args.out, os.path.basename(args.predicted),
                       '+'.join(selected_joint_names))
    if not os.path.isdir(out):
        os.makedirs(out)

    # Save the results to a CSV file
    df.to_csv(os.path.join(out, 'hellinger_distances.csv'))

    # Print and save the overall distances
    overall_str = ['Overall Hellinger distances:']
    print('Overall Hellinger distances:')
    for dir_key in dir_keys:
        overall_str.append('{}: {}'.format(dir_key, overall_dists[dir_key]))
        print('{: <20}'.format(dir_key),
              '\t{:.3f}'.format(overall_dists[dir_key]))
    print('')
    
    overall_str = '\n'.join(overall_str)

    with open(os.path.join(out, 'overall_distances.txt'), 'w') as f:
        f.write(overall_str)

    if args.visualize:
        # Set color and style
        mpl_default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        sns.set(context='poster', palette=sns.color_palette(mpl_default), font_scale=1.05)
        sns.set_style('white', {'legend.frameon': True})

        # Velocities are computed in 20fps: make them into cm/s
        plot_bins = [format(b, '.2f') for b in bins[:-1] * 20]

        # Plot speed in a range of [0, 15]
        plot_bins = plot_bins[:-4]

        # Make an output directory
        vis_out = os.path.join(out, 'histograms')
        if not os.path.isdir(vis_out):
            os.makedirs(vis_out)
        
        if args.match_yticks:
            max_percentage = int(reject_outliers(np.array(max_freqs)).max().item() * 100)

            tick_interval = 5 if max_percentage // 5 < 9 else 10  # Avoid too many ticks
            ticks = list(range(0, max_percentage, tick_interval))
        
        for file_key in file_keys:
            # Plot in a range of [0, 15]
            original_hist = results['original'][file_key]['hist'][:-4]

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)

            # Convert frequency to percentage
            gt_handle, = ax.plot(plot_bins, normalize(original_hist) * 100, color='C4')

            # Awesome way to create a tabular-style legend
            # https://stackoverflow.com/questions/25830780/tabular-legend-layout-for-matplotlib
            # Create a blank rectangle
            blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

            # Correspond to each columns of the tabular
            legend_handles = [blank, gt_handle]
            legend_names = ['Name', 'Ground Truth']
            legend_dists = ['Hell. Dist.', '0'.center(16)]

            colors = ['C1', 'C3', 'C0', 'C2'] if len(dir_keys) <= 4 else \
                     ['C1', 'C0', 'C6', 'C7', 'C8', 'C9', 'C5', 'C2', 'C3']

            assert len(dir_keys) <= len(colors)

            for color, dir_key in zip(colors, dir_keys):
                predicted_hist = results[dir_key][file_key]['hist'][:-4]
                label = dir_key.split('-')[1].replace('_smooth', '*')

                # if 'Aud2Pose' in label:
                #     label += ' [18]'

                handle, = ax.plot(plot_bins, normalize(predicted_hist) * 100, color=color)

                legend_handles.append(handle)
                legend_names.append(label)
                legend_dists.append('{:.3f}'.format(results[dir_key][file_key]['dist']).center(12))

            # Legend will have a tabular of (rows x 3)
            rows = len(legend_handles)
            empty_label = ['']

            legend_handles = legend_handles + [blank] * (rows * 2)
            legend_labels = np.concatenate([empty_label * rows, legend_names, legend_dists])

            ax.legend(legend_handles, legend_labels,
                      ncol=3, handletextpad=0.5, columnspacing=-2.15,
                      labelspacing=0.35)
            ax.set_xlabel('Speed (cm/s)')
            ax.set_ylabel('Frequency (%)')
            ax.set_xticks(np.arange(16))
            ax.tick_params(pad=6)

            if args.match_yticks:
                ax.set_ylim(0, max_percentage)
                ax.yaxis.set_major_locator(FixedLocator(ticks))
            else:
                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10], integer=True))

            plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.12)
            plt.savefig(os.path.join(vis_out, os.path.splitext(file_key)[0] + '_speed_histogram.pdf'))
            plt.show()

            plt.clf()
            plt.close()

    print('Results were writen in ' + out)
    print('')


if __name__ == '__main__':
    main()
