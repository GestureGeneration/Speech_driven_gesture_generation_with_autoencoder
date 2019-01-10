"""
This file contains a script for encoding motion dataset.

Usage example: python encode_dataset.py data_dir

Developed by Taras Kucherenko (tarask@kth.se)
"""

import sys
import numpy as np

import train as tr
import utils.flags as fl
from learn_dataset_encoding import create_nn, prepare_motion_data, check_params, os

if __name__ == '__main__':

    # Check the parameters
    check_params()

    DATA_DIR = sys.argv[1]

    # Additional check
    if not os.path.exists(DATA_DIR+"/"+str(fl.FLAGS.layer1_width)):
        raise ValueError(
            'Path to the dataset encoding ({}) does not exist!\nPlease, create a folder {} in the DATA_DIR directory'
            ''.format(DATA_DIR+"/"+str(fl.FLAGS.layer1_width), str(fl.FLAGS.layer1_width)))

    # Get the data
    train_normalized_data, train_data, test_normalized_data, test_data, dev_normalized_data, \
    max_val, mean_pose = prepare_motion_data(DATA_DIR)

    # Restore the network
    nn = create_nn(train_normalized_data, dev_normalized_data, max_val, mean_pose, restoring=True)

    debug = 0

    # For debug - shorten the dataset
    if debug:
        train_normalized_data = train_normalized_data[:12000]

    """                  Encode the train data                 """

    # Encode it
    encoded_train_data = tr.encode(nn, train_normalized_data)

    # And save into file
    np.save(DATA_DIR+"/"+str(fl.FLAGS.layer1_width)+"/Y_train_encoded.npy", encoded_train_data)

    if debug:
        print(train_normalized_data.shape)
        print(encoded_train_data.shape)

        # Decode train
        decoded = tr.decode(nn, encoded_train_data)
        print(decoded.shape)

        # Reshape back to the frames
        decoded = np.reshape(decoded, (-1, fl.FLAGS.frame_size))

        # And calculate an error

        size = min(train_normalized_data.shape[0], decoded.shape[0])
        error = decoded[:size] - train_data[:size]
        rmse = np.sqrt(np.mean(error**2))

        print("AE Train Error is ", rmse)

    """                  Encode the test data                 """

    # Encode it
    encoded_test_data = tr.encode(nn, test_normalized_data)

    # And save into files
    np.save(DATA_DIR+"/"+str(fl.FLAGS.layer1_width)+"/Y_test_encoded.npy", encoded_test_data)

    if debug:
        # Decode test
        decoded = tr.decode(nn, encoded_test_data)

        # Reshape back to the frames
        decoded = np.reshape(decoded, (-1, fl.FLAGS.frame_size))

        size = min(test_normalized_data.shape[0], decoded.shape[0])
        error = decoded[:size] - test_data[:size]
        rmse = np.sqrt(np.mean(error**2))

        print("AE Test Error is ", rmse)

    """                  Encode the dev data                     """

    # Encode it
    encoded_dev_data = tr.encode(nn, dev_normalized_data)

    # And save into files
    np.save(DATA_DIR+"/"+str(fl.FLAGS.layer1_width)+"/Y_dev_encoded.npy", encoded_dev_data)
