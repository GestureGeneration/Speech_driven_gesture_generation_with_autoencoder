"""
This file contains a usage script, intended to test using interface.
Developed by Taras Kucherenko (tarask@kth.se)
"""

import train as tr
import utils.data as dt
import utils.flags as fl
from learn_dataset_encoding import create_nn, prepare_motion_data

import numpy as np

import sys

DATA_DIR = sys.argv[1]
TEST_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

if __name__ == '__main__':

    # Get the data
    Y_train_normalized, Y_train, Y_test_normalized, Y_test, Y_dev_normalized, max_val, mean_pose  = prepare_motion_data(DATA_DIR)

    # Train the network
    nn = create_nn(Y_train_normalized, Y_dev_normalized, max_val, mean_pose, restoring=True)

    # Read the encoding
    encoding = np.loadtxt(TEST_FILE)

    print(encoding.shape)

    # Decode it
    decoding = tr.decode(nn, encoding)

    print(decoding.shape)

    np.savetxt(OUTPUT_FILE, decoding, delimiter = ' ')

    # Close Tf session
    nn.session.close()
