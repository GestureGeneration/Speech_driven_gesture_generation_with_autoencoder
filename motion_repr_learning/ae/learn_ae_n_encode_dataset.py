"""
This file contains a script for learning encoding-decoding network
on our dataset.

Usage example: python learn_dataset_encoding.py data_dir

Developed by Taras Kucherenko (tarask@kth.se)
"""

import sys
sys.path.append('.')
import numpy as np
import os

import train as autoencoder_training
from utils.utils import prepare_motion_data, DataSet, DataSets

from config import args

def create_nn(train_data, dev_data, max_val, mean_pose):
    """
    Train or restore a neural network
    Args:
     train_data:         training dataset normalized to the values [-1,1]
     dev_data:           dev dataset normalized to the values [-1,1]
     max_val:            maximal values in the dataset
     mean_pose:          mean pose of the dataset
    Returns:
     nn: neural network, which is ready to use
    """

    # Create DataSet object

    data = DataSets()

    data.train = DataSet(train_data, args.batch_size)
    data.test = DataSet(dev_data, args.batch_size)

    # Assign variance
    data.train.sigma = np.std(train_data, axis=(0, 1))

    # Create information about the dataset
    data_info = autoencoder_training.DataInfo(data.train.sigma, data.train._sequences.shape,
                            data.test._sequences.shape, max_val, mean_pose)

    # Train the network
    nn = autoencoder_training.learning(data, data_info, just_restore=args.load_model_from_checkpoint)

    return nn

def check_params():
    # Check if the dataset exists
    if not os.path.isdir(os.path.abspath(args.data_dir)):
        raise ValueError(f'Path to the dataset ({os.path.abspath(args.data_dir)}) does not exist!\n' + \
                          'Please provide the correct path.')

    # Check if the flags were set properly
    if not os.path.isdir(os.path.abspath(args.chkpt_dir)):
         raise ValueError(f'Path to the checkpoints ({args.chkpt_dir}) does not exist!\n' + \
                           'Please provide the correct path.')

if __name__ == '__main__':
    # Check parameters
    check_params()

    train_normalized_data, train_data, dev_normalized_data, \
    max_val, mean_pose = prepare_motion_data(args.data_dir)

    # Train or load the AE network
    nn = create_nn(train_normalized_data, dev_normalized_data, max_val, mean_pose)

    """       Create save directory for the encoded data         """
    
    save_dir = os.path.join(args.data_dir, str(args.layer1_width))
    
    if not os.path.isdir(save_dir):
        print(f"Created directory {os.path.abspath(save_dir)} for saving the encoded data.")
        os.makedirs(save_dir)


    """                  Encode the train data                   """
    # Encode it
    encoded_train_data = autoencoder_training.encode(nn, train_normalized_data)
    # And save into file
    np.save(os.path.join(save_dir, "Y_train_encoded.npy"), encoded_train_data)

    """                  Encode the dev data                     """

    # Encode it
    encoded_dev_data = autoencoder_training.encode(nn, dev_normalized_data)
    # And save into files
    np.save(os.path.join(save_dir, "Y_dev_encoded.npy"), encoded_dev_data)
