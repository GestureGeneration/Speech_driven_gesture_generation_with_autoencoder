"""
This file contains a script for learning encoding-decoding network
on our dataset.

Usage example: python learn_dataset_encoding.py data_dir

Developed by Taras Kucherenko (tarask@kth.se)
"""

import sys
import numpy as np
import os

import train as tr
from utils.utils import prepare_motion_data, DataSet, DataSets, fl

def create_nn(train_data, dev_data, max_val, mean_pose, restoring):
    """
    Train or restore a neural network
    Args:
     train_data:         training dataset normalized to the values [-1,1]
     dev_data:           dev dataset normalized to the values [-1,1]
     max_val:            maximal values in the dataset
     mean_pose:          mean pose of the dataset
     restoring:          weather  we are going to just restore already trained model
    Returns:
     nn: neural network, which is ready to use
    """

    # Create DataSet object

    data = DataSets()

    data.train = DataSet(train_data, fl.FLAGS.batch_size)
    data.test = DataSet(dev_data, fl.FLAGS.batch_size)

    # Assign variance
    data.train.sigma = np.std(train_data, axis=(0, 1))

    # Create information about the dataset
    data_info = tr.DataInfo(data.train.sigma, data.train._sequences.shape,
                            data.test._sequences.shape, max_val, mean_pose)

    # Set "restore" flag
    fl.FLAGS.restore = restoring

    # Train the network
    nn = tr.learning(data, data_info, just_restore=restoring)

    return nn

def check_params():

    # Check if script get enough parameters
    if len(sys.argv)<2:
        raise ValueError('Not enough paramters! \nUsage : python '+sys.argv[0].split("/")[-1]+' DATA_DIR')

    # Check if the dataset exists
    if not os.path.exists(sys.argv[1]):
        raise ValueError('Path to the dataset ({}) does not exist!\nPlease, provide correct DATA_DIR as a script parameter'
                         ''.format(sys.argv[1]))

    # Check if the flags were set properly

    if not os.path.exists(fl.FLAGS.chkpt_dir):
        raise ValueError('Path to the checkpoints ({}) does not exit!\nChange the "chkpt_dir" flag in utils/flags.py'
                         ''.format(fl.FLAGS.chkpt_dir))

if __name__ == '__main__':

    # Check parameters
    check_params()

    # Get the data
    DATA_DIR = sys.argv[1]
    train_normalized_data, train_data, test_normalized_data, test_data, dev_normalized_data, \
    max_val, mean_pose = prepare_motion_data(DATA_DIR)

    # Train an AE network
    nn = create_nn(train_normalized_data, dev_normalized_data, max_val, mean_pose, restoring=False)