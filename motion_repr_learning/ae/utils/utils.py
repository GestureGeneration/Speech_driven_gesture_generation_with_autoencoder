"""
This file contains helping function for the training and testing of the AE
"""


import numpy as np
import tensorflow as tf


import utils.flags as fl

""" Dataset class"""

class DataSet(object):
    '''
    A class for storing a dataset and all important information,
    which might be needed during training,
    such as batch size amount of epochs completed and so on.
    '''


    def __init__(self, sequences, batch_size):
        self._batch_size = batch_size
        self._sequences = sequences  # all the sequnces in the dataset
        self._num_sequences = sequences.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    # Make interface to the protected variables
    @property
    def sequences(self):
        return self._sequences

    @property
    def num_sequences(self):
        return self._num_sequences

class DataSets(object):
    '''
      A class for storing Train and Eval datasets and all related information,
      '''
    pass

def read_test_seq_from_binary(binary_file_name):
    """ Read test sequence from the binart file
          Args:
            binary_file_name:  the name of the input binary file
          Returns:
            read_seq:          test sequence
    """
    # Read the sequence
    read_seq = np.fromfile(binary_file_name)
    # Reshape
    read_seq = read_seq.reshape(-1, fl.FLAGS.frame_size)
    amount_of_frames = int(read_seq.shape[0] / (fl.FLAGS.chunk_length))
    if amount_of_frames > 0:
        # Clip array so that it divides exactly into the inputs we want (frame_size * chunk_length)
        read_seq = read_seq[0:amount_of_frames * fl.FLAGS.chunk_length]

    # Reshape
    read_seq = read_seq.reshape(-1, fl.FLAGS.frame_size * fl.FLAGS.chunk_length) #?

    return read_seq

def add_noise(x, variance_multiplier, sigma):
    """
           Add Gaussian noise to the data
           Args:
               x                   - input vector
               variance_multiplier - coefficient to multiple variance of the noise on
               sigma               - variance of the dataset
           Returns:
               x - output vector, noisy data
    """
    eps = 1e-15
    noise = tf.random_normal(x.shape, 0.0, stddev=np.multiply(sigma, variance_multiplier) + eps)
    x = x + noise
    return x

def loss_reconstruction(output, target, max_vals, pretrain=False):
    """ Reconstruction error. Square of the RMSE

    Args:
      output:    tensor of net output
      target:    tensor of net we are trying to reconstruct
      max_vals:  array of absolute maximal values in the dataset,
                is used for scaling an error to the original space
      pretrain:  wether we are using it during the pretraining phase
    Returns:
      Scalar tensor of mean squared Eucledean distance
    """
    with tf.name_scope("reconstruction_loss"):
        net_output_tf = tf.convert_to_tensor(tf.cast(output, tf.float32), name='input')
        target_tf = tf.convert_to_tensor(tf.cast(target, tf.float32), name='target')

        # Euclidean distance between net_output_tf,target_tf
        error = tf.subtract(net_output_tf, target_tf)

        if not pretrain:
            # Convert it back from the [-1,1] to original values
            error_scaled = tf.multiply(error, max_vals[np.newaxis, :] + 1e-15)
        else:
            error_scaled = error

        squared_error = tf.reduce_mean(tf.square(error_scaled, name="square"), name="averaging")
    return squared_error

def convert_back_to_3d_coords(sequence, max_val, mean_pose):
    '''
    Convert back from the normalized values between -1 and 1 to original 3d coordinates
    and unroll them into the sequence

    Args:
        sequence: sequence of the normalized values
        max_val: maximal value in the dataset
        mean_pose: mean value in the dataset

    Return:
        3d coordinates corresponding to the batch
    '''

    # Convert it back from the [-1,1] to original values
    reconstructed = np.multiply(sequence, max_val[np.newaxis, :] + 1e-15)

    # Add the mean pose back
    reconstructed = reconstructed + mean_pose[np.newaxis, :]

    # Unroll batches into the sequence
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])

    return reconstructed

def reshape_dataset(dataset):
    """
    Changing the shape of the dataset array to correspond to the frame dimentionality

    Args:
        dataset: an array of the dataset
    Return:
        dataset_final: array of the dataset in a proper shape
    """

    amount_of_train_chunks = int(dataset.shape[0] / fl.FLAGS.chunk_length)
    dataset_shorten = dataset[:amount_of_train_chunks * fl.FLAGS.chunk_length, :fl.FLAGS.frame_size]
    dataset_chunks = np.reshape(dataset_shorten, (-1, fl.FLAGS.chunk_length * fl.FLAGS.frame_size))

    # Merge all the time-frames together
    dataset_final = np.reshape(dataset_chunks, [amount_of_train_chunks,
                                                fl.FLAGS.chunk_length * fl.FLAGS.frame_size])

    return dataset_final

def prepare_motion_data(data_dir):
    """
    Read and preprocess the motion dataset

    Args:
        data_dir:           a directory with the dataset
    Return:
        Y_train:            an array of the training dataset
        Y_train_normalized: training dataset normalized to the values [-1,1]
        Y_test:             an array of the test dataset
        Y_test_normalized:  test dataset normalized to the values [-1,1]
        Y_dev_normalized:   dev dataset normalized to the values [-1,1]
        max_val:            maximal values in the dataset
        mean_pose:          mean pose of the dataset
    """

    # Get the data

    Y_train = np.load(data_dir + '/Y_train.npy')
    Y_dev = np.load(data_dir + '/Y_dev.npy')

    # Normalize dataset
    max_val = np.amax(np.absolute(Y_train), axis=(0))
    mean_pose = Y_train.mean(axis=(0))

    Y_train_centered = Y_train - mean_pose[np.newaxis, :]
    Y_dev_centered = Y_dev - mean_pose[np.newaxis, :]

    # Scales all values in the input_data to be between -1 and 1
    eps = 1e-8
    Y_train_normalized = np.divide(Y_train_centered, max_val[np.newaxis, :] + eps)
    Y_dev_normalized = np.divide(Y_dev_centered, max_val[np.newaxis, :] + eps)

    # Reshape to accomodate multiple frames at each input

    if fl.FLAGS.chunk_length > 1:
        Y_train_normalized = reshape_dataset(Y_train_normalized)
        Y_dev_normalized = reshape_dataset(Y_dev_normalized)

    # Pad max values and the mean pose, if neeeded
    if fl.FLAGS.chunk_length > 1:
        max_val = np.tile(max_val, fl.FLAGS.chunk_length)
        mean_pose = np.tile(mean_pose, fl.FLAGS.chunk_length)


    # Some tests for flags
    if fl.FLAGS.restore and fl.FLAGS.pretrain:
        print('ERROR! You cannot restore and pretrain at the same time!'
              ' Please, chose one of these options')
        exit(1)

    if fl.FLAGS.middle_layer > fl.FLAGS.num_hidden_layers:
        print('ERROR! Middle layer cannot be more than number of hidden layers!'
              ' Please, update flags')
        exit(1)

    return Y_train_normalized, Y_train,\
           Y_dev_normalized, max_val, mean_pose
