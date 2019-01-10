"""Functions for downloading,reading and preprocessing CMU data."""

import sys
import os

#sys.path.append('/home/taras/Desktop/Work/Code/Git/MotionCleaning/BVH_format/parser')
#from reader import MyReader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from six.moves import xrange

import utils.flags as fl

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

    @property
    def sequences(self):
        return self._sequences

    @property
    def num_sequences(self):
        return self._num_sequences

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self):
        """Return the next batch of sequences from this data set."""
        batch_numb = self._index_in_epoch
        self._index_in_epoch += self._batch_size
        if self._index_in_epoch > self._num_chunks:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_sequences)
            np.random.shuffle(perm)
            self._sequences = self._sequences[perm]
            # Start next epoch
            batch_numb = 0
            self._index_in_epoch = self._batch_size
        return self._sequences[batch_numb:batch_numb + self._batch_size:1, :]


class DataSets(object):
    '''
      A class for storing Train and Eval datasets and all related information,
      '''
    pass


def read_bvh_file(fileName, test=False):
    """
       Reads a file from CMU MoCap dataset in BVH format

       Returns:
            sequence [sequence_length,frame_size] - local chanells transformed to the hips-centered coordinates
            hips [frame_size] - coordinates of the hips

    """

    # Read the data
    reader = MyReader(fileName);
    reader.read();
    sequence = np.array(reader.points)

    # Translate to the hips-center coordinate system
    hips = sequence[:,:,0]
    sequence = sequence - hips[:,:,np.newaxis]

    # This is a visualization for debug
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    treshhold = 22 # to show legs in a different color
    # use 10 to color only the spine, 16 - spine and right hand, 22 - spine and both arms, 27 - all except left leg, 32 - all
    time_step = 10
    ax.scatter(sequence[time_step ][2][0:treshhold],sequence[time_step ][0][0:treshhold], sequence[time_step ][1][0:treshhold],
               c='r', marker='o')
    ax.scatter(sequence[time_step ][2][treshhold:], sequence[time_step ][0][treshhold:], sequence[time_step ][1][treshhold:],
               c='g', marker='o')'''
    plt.show()

    # Transpose the last 2 dimensions
    sequence = np.transpose(sequence, axes = (0,2,1))

    #Flaten all the coords into one vector [T,3,m] -> [T,3m]
    return np.reshape(sequence,(sequence.shape[0],sequence.shape[1]*sequence.shape[2])),hips

def read_a_folder(curr_dir):
    chunk_length = fl.FLAGS.chunk_length
    stride = fl.FLAGS.chunking_stride

    data = np.array([])

    for filename in os.listdir(curr_dir):
            curr_sequence,_ = read_bvh_file(curr_dir + '/' + filename)

            # Split sequence into chunks
            curr_chunks = np.array([curr_sequence[i:i + chunk_length, :] for i in
                                    xrange(0, len(curr_sequence) - chunk_length, stride)])

            if curr_chunks.shape[0] > 0:
                # Concatanate curr chunks to all of them
                data = np.vstack([data, curr_chunks]) if data.size else np.array(curr_chunks)

            print(data.shape)

    data = np.array(data)

    return data

def read_unlabeled_data(train_dir, evaluate):
    """
      Reads all 3 datasets from CMU MoCap dataset in C3D format

      Args:
          train_dir - address to the train, dev and eval datasets
          evaluate - flag : weather we want to evaluate a network or we just optimize parameters
      Returns:
          datasets - object of class DataSets, containing Train and Eval datasets
          max_val - maximal value in the raw data ( for post-processing)
          mean_pose - mean pose in the raw data ( for post-processing)
    """

    data_sets = DataSets()

    # Get constants from the file
    data_dir = fl.FLAGS.data_dir
    chunk_length = fl.FLAGS.chunk_length
    stride = fl.FLAGS.chunking_stride

    if stride > chunk_length:
        print(
            'ERROR! \nYou have stride bigger than lentgh of chunks. '
            'Please, change those values at flags.py, so that you don\'t ignore the data')
        exit(0)

    # #########             Get TRAIN data                  ###########
    print('\nReading train data from the following folder ... ', data_dir + '/train/labels')

    train_data = read_a_folder(data_dir + '/train/labels')

    [amount_of_train_strings, seq_length, DoF] = train_data.shape
    print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(
        seq_length) + ' will be used for training')

    #         #########             Get TEST data                  ###########

    if evaluate:
        print('\nReading test data from the following folder : ', data_dir + '/eval/labels')
        test_data = read_a_folder(data_dir + '/eval/labels')
    else:
        print('\nReading test data from the following folder : ', data_dir + '/dev/labels')
        test_data = read_a_folder(data_dir + '/dev/labels')

    [amount_of_test_strings, seq_length, DoF] = test_data.shape
    print('\n' + str(amount_of_test_strings) + ' sequences with length '
          + str(seq_length) + ' will be used for testing')

    # Do mean normalization : substract mean pose
    mean_pose = train_data.mean(axis=(0, 1))
    train_data = train_data - mean_pose[np.newaxis, np.newaxis, :]
    test_data = test_data - mean_pose[np.newaxis, np.newaxis, :]

    # Scales all values in the input_data to be between -1 and 1
    eps = 1e-8
    max_train = np.amax(np.absolute(train_data), axis=(0, 1))
    max_test = np.amax(np.absolute(test_data), axis=(0, 1))
    max_val = np.maximum(max_train, max_test)
    train_data = np.divide(train_data, max_val[np.newaxis, np.newaxis, :] + eps)
    test_data = np.divide(test_data, max_val[np.newaxis, np.newaxis, :] + eps)

    # Check the data range
    max_ = test_data.max()
    min_ = test_data.min()

    print("MAximum value in the normalized test dataset : " + str(max_))
    print("Minimum value in the normalized test dataset : " + str(min_))

    print('\nTrain data shape: ', train_data.shape)

    data_sets.train = DataSet(train_data, fl.FLAGS.batch_size)
    data_sets.test = DataSet(test_data, fl.FLAGS.batch_size)

    # Assign variance
    data_sets.train.sigma = np.std(train_data, axis=(0, 1))

    # Check if we have enough data
    if data_sets.train._num_sequences < data_sets.train._batch_size:
        print('ERROR: We have got not enough data! '
              'Reduce batch_size or increase amount of subfolder you use.')
        exit(1)

    return data_sets, max_val, mean_pose


def read_dataset_and_write_in_binary(evaluate):
    """
              Reads 3 datasets: "Train","Dev" and "Eval" from the CMU MoCap dataset in bvh format
              And write them in the binary format.
              Will get the address of the folder with the data from flags.py
              Args:
                  evaluate - flag: weather we evaluate the system or we optimize parameters
              Returns:
                  will write binary files in the same folder as the original data
    """

    # Get the data
    data, max_val, mean_pose = read_unlabeled_data(fl.FLAGS.data_dir, False)  # read_all_the_data()

    # Write all important information into binary files

    # Datasets themselfs
    train_file = open(fl.FLAGS.data_dir + '/train.binary', 'wb')
    data.train._sequences.tofile(train_file)
    train_file.close()

    eval_file = open(fl.FLAGS.data_dir + '/eval.binary', 'wb')
    data.test._sequences.tofile(eval_file)
    eval_file.close()

    # Dataset properties

    sigma_file = open(fl.FLAGS.data_dir + '/variance.binary', 'wb')
    data.train.sigma.tofile(sigma_file)
    sigma_file.close()

    max_val_file = open(fl.FLAGS.data_dir + '/maximums.binary', 'wb')
    max_val.tofile(max_val_file)
    max_val_file.close()

    mean_file = open(fl.FLAGS.data_dir + '/mean.binary', 'wb')
    mean_pose.tofile(mean_file)
    mean_file.close()

    print('All the binary files for the dataset was saved in the folder ', fl.FLAGS.data_dir)


def read_binary_dataset(dataset_name):
    filename = fl.FLAGS.data_dir + '/' + dataset_name + '.binary'
    dataset = np.fromfile(filename)
    amount_of_frames = int(dataset.shape[0] /(fl.FLAGS.chunk_length * fl.FLAGS.frame_size))
    # Clip array so that it divides exactly into the inputs we want (frame_size *chunk_length)
    dataset = dataset[0:amount_of_frames * fl.FLAGS.chunk_length * fl.FLAGS.frame_size]
    # Reshape
    dataset = dataset.reshape(amount_of_frames, fl.FLAGS.chunk_length, fl.FLAGS.frame_size)
    return dataset


def read_3_datasets_from_binary():
    """
      Reads train and test datasets and their properties from binary file format

      Will take them from the corresponding file in the folder, which is defined by FLAGS.data_dir

      Returns:
          datasets  - object of class DataSets, containing Train and Eval datasets
          max_val   - maximal value in the raw data ( for post-processing)
          mean_pose - mean pose in the raw data ( for post-processing)

    """
    data_sets = DataSets()

    #         #########             Get TRAIN data                  ###########

    train_data = read_binary_dataset('train')
    [amount_of_train_strings, seq_length, DoF] = train_data.shape
    print('\n' + str(amount_of_train_strings) + ' sequences with length ' + str(fl.FLAGS.chunk_length)
          + ' frames in each will be used for training')

    # Merge all the time-frames together
    train_data = np.reshape(train_data, [amount_of_train_strings, seq_length * DoF])

    #         #########             Get TEST data                  ###########

    test_data = read_binary_dataset('eval')
    [amount_of_test_strings, seq_length, DoF] = test_data.shape
    print(str(amount_of_test_strings) + ' sequences will be used for testing')

    # Merge all the time-frames together
    test_data = np.reshape(test_data, [amount_of_test_strings, seq_length * DoF])

    # Shuffle the data
    perm = np.arange(amount_of_train_strings)
    np.random.shuffle(perm)
    train_data = train_data[perm]

    data_sets.train = DataSet(train_data, fl.FLAGS.batch_size)
    data_sets.test = DataSet(test_data, fl.FLAGS.batch_size)

    # Assign variance
    data_sets.train.sigma = np.std(train_data, axis=(0, 1))

    # Read maximal value and mean pose before normalizatio
    max_val = np.fromfile(fl.FLAGS.data_dir + '/maximums.binary')
    mean_pose = np.fromfile(fl.FLAGS.data_dir + '/mean.binary')

    # Check if we have enough data
    if data_sets.train._num_sequences < data_sets.train._batch_size:
        print('ERROR: We have got not enough data! '
              'Reduce batch_size or increase amount of subfolder you use.')
        exit(1)

    return data_sets, max_val, mean_pose


def write_test_seq_in_binary(input_file_name, output_file_name):
    """ Read test sequence in c3d format and
        write it into the binart file

      Args:
        input_file_name:  the name of the input file
        output_file_name: the name of the output file
      Returns:
        nothing
    """
    test_file = open(output_file_name, 'wb')
    test_seq,_ = read_bvh_file(input_file_name)
    test_seq.tofile(test_file)
    test_file.close()
    print("The test sequence was read from", input_file_name, " and written to", output_file_name)


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


def visualize(mocap_seq, test=False):
    all_3d_coords = mocap_seq.reshape(-1, 3, int(fl.FLAGS.frame_size/3))  # Concatanate all coords into one vector

    # For debug - Visualize the skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_frame = 40
    treshhold_0 = 14
    treshhold_1 = 20
    treshhold_2 = 27
    coef = 100
    for step in range(start_frame, start_frame + 30, 10):

        # Visualize a 3D point cloud
        ax.scatter3D(all_3d_coords[step][0][:treshhold_0],
                     np.add(all_3d_coords[step][1][:treshhold_0], (step - start_frame) * coef),
                     all_3d_coords[step][2][:treshhold_0], c='c', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_0:treshhold_1],
                     np.add(all_3d_coords[step][1][treshhold_0:treshhold_1],
                            (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_0:treshhold_1], c='r', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_1:treshhold_2],
                     np.add(all_3d_coords[step][1][treshhold_1:treshhold_2],
                            (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_1:treshhold_2], c='y', marker='o')
        ax.scatter3D(all_3d_coords[step][0][treshhold_2:],
                     np.add(all_3d_coords[step][1][treshhold_2:], (step - start_frame) * coef),
                     all_3d_coords[step][2][treshhold_2:], c='b', marker='o')

        # Find which points are present

        key_point_arm = []
        for point in list([0, 1, 2, 7, 8, 9]):
            if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
                if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
                    if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
                        key_point_arm.append(point)

        key_point_arm = np.array(key_point_arm)

        key_point_leg = []
        for point in list([27, 34]):  # 28, 35
            if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
                if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
                    if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
                        key_point_leg.append(point)
        key_point_leg = np.array(key_point_leg)

        # Add lines in between

        for point in key_point_arm:
            xline = all_3d_coords[step][0][point:point + 2]
            yline = np.add(all_3d_coords[step][1][point:point + 2], (step - start_frame) * coef)
            zline = all_3d_coords[step][2][point:point + 2]
            ax.plot(xline, yline, zline, c='c')
        for point in key_point_leg:
            xline = all_3d_coords[step][0][point:point + 3:2]
            yline = np.add(all_3d_coords[step][1][point:point + 3:2], (step - start_frame) * coef)
            zline = all_3d_coords[step][2][point:point + 3:2]
            ax.plot(xline, yline, zline, c='b')

    plt.show()


if __name__ == '__main__':

    # Do some testing

    Test = False

    if Test:
        input_file_name = '/home/taras/Documents/Datasets/SpeechToMotion/Japanese/TheLAtest/dataset/motion/gesture22.bvh'
        output_file_name = fl.FLAGS.data_dir + '/talking2.csv'

        test_file = open(output_file_name, 'wb')
        test_seq, _ = read_bvh_file(input_file_name)

        visualize(test_seq, test=False)

        # Save the data into a file
        with open(output_file_name, 'w') as fp:
            np.savetxt(fp, test_seq, delimiter=",")

        print("The test sequence was read from", input_file_name, " and written to", output_file_name)

        write_test_seq_in_binary('/home/taras/Documents/Datasets/SpeechToMotion/Japanese/TheLAtest/dataset/motion/gesture1093.bvh',
                                 fl.FLAGS.data_dir + '/test_1.binary')
        write_test_seq_in_binary('/home/taras/Documents/Datasets/SpeechToMotion/Japanese/TheLAtest/dataset/motion/gesture1097.bvh',
                                 fl.FLAGS.data_dir + '/test_2.binary')

    else:
        read_dataset_and_write_in_binary(True)
