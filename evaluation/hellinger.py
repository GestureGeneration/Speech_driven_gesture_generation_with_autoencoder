# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:30:40 2020

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def main():
    parser = argparse.ArgumentParser(
        description='Calculate histograms of moving distances')
    parser.add_argument('--original', default='data/original',
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default='data/predicted',
                        help='Predicted gesture directory')
    parser.add_argument('--file', '-f', default='hmd_vel_0.05.csv',
                        help='File name to load')
    parser.add_argument('--joints', '-j', default='joints.txt',
                        help='Joint name file')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    args = parser.parse_args()

    joint_names = read_joint_names(args.joints)

    if args.select is not None:
        selected_joints = []
        for s in args.select:
            if not s in joint_names:
                print('Ignore invalid joint: {}'.format(s))
            else:
                selected_joints.append(s)

        if not selected_joints:
            selected_joints = ['Total']
            print('No valid joints are selected. Use all joints')
    else:
        # Use all joints
        selected_joints = ['Total']

    def get_directories(directory):
        return sorted(filter(lambda x: os.path.isdir(x), glob.glob(directory)))

    # Read original gesture's distribution
    original_file = os.path.join(args.original, args.file)
    original = pd.read_csv(original_file, index_col=0)
    original_hist = np.array(original[selected_joints]).sum(axis=1)

    # List of predicted gesture direcotires
    predicted_dirs = get_directories(os.path.join(args.predicted, '*'))

    results = {os.path.basename(d): None for d in predicted_dirs}

    # Iterate over the list of direcotires
    for predicted_dir in predicted_dirs:
        # Does this directory have a target file?
        try:
            predicted_file = os.path.join(predicted_dir, args.file)
            predicted = pd.read_csv(predicted_file, index_col=0)
        except FileNotFoundError:
            # Are there any subdirectories which have integer names?
            sub_dirs = sorted(
                filter(lambda x: os.path.basename(x).isdecimal(),
                get_directories(os.path.join(predicted_dir, '*'))))

            # If no, raise an exception
            if not sub_dirs:
                raise FileNotFoundError(
                    'There is neither ' + args.file
                    + ' nor subdirectories in ' + predicted_dir)

            predicted = None
            for sub_dir in sub_dirs:
                predicted_file = os.path.join(sub_dir, args.file)
                tmp = pd.read_csv(predicted_file, index_col=0)

                if predicted is None:
                    predicted = tmp
                else:
                    predicted = predicted + tmp
                
            predicted = predicted / float(len(sub_dirs))

        # Get histograms
        predicted_hist = np.array(predicted[selected_joints]).sum(axis=1)

        assert len(original_hist) == len(predicted_hist)

        # Hellinger distance between two histograms
        dist = hellinger(original_hist, predicted_hist)

        # Store results
        key = os.path.basename(predicted_dir)
        results[key] = {'dist': dist, 'hist': predicted_hist}

    # Print and save results
    keys = natural_sort(results.keys())

    result_str = ['Hellinger distances:']
    for key in keys:
        result_str.append('\t{}: {}'.format(key, results[key]['dist']))
    
    result_str = '\n'.join(result_str)
    
    print(result_str)
    print('')

    # Make output directory
    out = os.path.join(args.out, os.path.basename(args.predicted),
                       '+'.join(selected_joints))
    if not os.path.isdir(out):
        os.makedirs(out)
    
    with open(os.path.join(out, 'distances.txt'), 'w') as f:
        f.write(result_str)

    if args.visualize:
        # Set color and style
        mpl_default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        sns.set(context='poster', palette=sns.color_palette(mpl_default), font_scale=1.05)
        sns.set_style('white', {'legend.frameon':True})

        # Velocities are computed in 20fps: make them into cm/s
        index = original.index * 20
        bins = [format(i, '.2f') for i in list(index)]

        # Plot speed in a range of [0, 15]
        bins = bins[:-4]
        original_hist = original_hist[:-4]

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Convert frequency to percentage
        gt_handle, = ax.plot(bins, normalize(original_hist) * 100, color='C4')

        # Awesome way to create a tabular-style legend
        # https://stackoverflow.com/questions/25830780/tabular-legend-layout-for-matplotlib
        # Create a blank rectangle
        blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

        # Correspond to each columns of the tabular
        legend_handles = [blank, gt_handle]
        legend_names = ['Name', 'Ground Truth']
        legend_dists = ['Hell. Dist.', '0'.center(16)]

        colors = ['C1', 'C3', 'C0', 'C2'] if len(keys) <= 4 else \
                 ['C1', 'C0', 'C6', 'C7', 'C8', 'C9', 'C5', 'C2', 'C3']
        
        assert len(keys) <= len(colors)

        for color, key in zip(colors, keys):
            predicted_hist = results[key]['hist'][:-4]
            label = key.split('-')[1].replace('_smooth', '*')

            #if 'Aud2Pose' in label:
            #    label += ' [18]'

            handle, = ax.plot(bins, normalize(predicted_hist) * 100, color=color)

            legend_handles.append(handle)
            legend_names.append(label)
            legend_dists.append('{:.3f}'.format(results[key]['dist']).center(12))

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
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10], integer=True))

        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.12)
        plt.savefig(os.path.join(out, 'speed_histogram.pdf'))
        plt.show()
    
    print('Results were writen in ' + out)
    print('')


if __name__ == '__main__':
    main()
