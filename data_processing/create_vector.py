"""
This script does preprocessing of the dataset specified in DATA_DIR
 and stores it in the same folder as .npy files
It should be used before training, as described in the README.md
"""

import os
import os.path
import sys


module_path = os.path.abspath(os.path.join('/home/taras/Desktop/Work/Code/Git/My_Fork/Speech_driven_gesture_generation_with_autoencoder/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing.bvh_read.BVH_io import bvh2npy
from data_processing.tools import *

N_OUTPUT = 24 # Number of gesture features (position)
DATA_DIR = ''
N_CONTEXT = 0  # Number of context: Total of how many pieces are seen before and after, when it is 60, 30 before and after
WINDOW_LENGTH = 50 # in miliseconds
FEATURES = "Pros"

if FEATURES == "MFCC":
    N_INPUT = 26 # Number of MFCC features
if FEATURES == "Pros":
    N_INPUT = 4 # Number of prosodic features
if FEATURES == "MFCC+Pros":
    N_INPUT = 30 # Total number of features
if FEATURES == "Spectro":
    N_INPUT = 64 # Number of spectrogram features
if FEATURES == "Spectro+Pros":
    N_INPUT = 68  # Total number of eatures

# Set silence adress
if os.path.isfile("data_processing/silence.wav"):
    SILENCE_PATH = "data_processing/silence.wav"
elif os.path.isfile("silence.wav"):
    SILENCE_PATH = "silence.wav"
else:
    raise("Could not find a file with the silence !!! Make sure it exists and is in ' Data processing' folder")


def pad_sequence(input_vectors):
    """
    Pad array of features in order to be able to take context at each time-frame
    We pad N_CONTEXT / 2 frames before and after the signal by the features of the silence
    Args:
        input_vectors:      feature vectors for an audio

    Returns:
        new_input_vectors:  padded feature vectors
    """

    if FEATURES == "MFCC":

        # Pad sequence not with zeros but with MFCC of the silence

        silence_vectors = calculate_mfcc(SILENCE_PATH)
        mfcc_empty_vector = silence_vectors[0]

        empty_vectors = np.array([mfcc_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Pros":

        # Pad sequence with zeros

        prosodic_empty_vector =[0, 0, 0, 0]

        empty_vectors = np.array([prosodic_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "MFCC+Pros":

        silence_vectors = calculate_mfcc(SILENCE_PATH) #
        mfcc_empty_vector = silence_vectors[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro":

        silence_spectro = calculate_spectrogram(SILENCE_PATH)
        spectro_empty_vector = silence_spectro[0]

        empty_vectors = np.array([spectro_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro+Pros":

        silence_spectro = calculate_spectrogram(SILENCE_PATH)
        spectro_empty_vector = silence_spectro[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((spectro_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append N_CONTEXT/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors


def create_vectors(audio_filename, gesture_filename):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)
        gesture_filename:  file name for a motion file (.bvh)

    Returns:
        input_with_context   : speech features
        output_with_context  : motion features
    """
    # Step 1: Vactorizing speech, with features of N_INPUT dimension, time steps of 0.01s
    # and window length with 0.025s => results in an array of 100 x N_INPUT

    if FEATURES == "MFCC":

        input_vectors = calculate_mfcc(audio_filename)

    elif FEATURES == "Pros":

        input_vectors = extract_prosodic_features(audio_filename)

    elif FEATURES == "MFCC+Pros":

        mfcc_vectors = calculate_mfcc(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        mfcc_vectors, pros_vectors = shorten(mfcc_vectors, pros_vectors)

        input_vectors = np.concatenate((mfcc_vectors, pros_vectors), axis=1)

    elif FEATURES =="Spectro":

        input_vectors = calculate_spectrogram(audio_filename)

    elif FEATURES == "Spectro+Pros":
        spectr_vectors = calculate_spectrogram(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        spectr_vectors, pros_vectors = shorten(spectr_vectors, pros_vectors)

        input_vectors = np.concatenate((spectr_vectors, pros_vectors), axis=1)

    # Step 2: Vectorize BVH

    Hand_joints = ['Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftArm', 'LeftForeArm', 'LeftHand']
    output_vectors =  bvh2npy(gesture_filename, Hand_joints, hips_centering=True)

    # Step 3: Align vector length
    input_vectors, output_vectors = shorten(input_vectors, output_vectors)

    # Step 4: Retrieve N_CONTEXT each time, stride one by one
    input_with_context = np.array([])
    output_with_context = np.array([])

    strides = len(input_vectors)

    input_vectors = pad_sequence(input_vectors)

    for i in range(strides):
        stride = i + int(N_CONTEXT/2)
        if i == 0:
            input_with_context = input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT)
            output_with_context = output_vectors[i].reshape(1, N_OUTPUT)
        else:
            input_with_context = np.append(input_with_context, input_vectors[stride - int(N_CONTEXT/2) : stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT), axis=0)
            output_with_context = np.append(output_with_context, output_vectors[i].reshape(1, N_OUTPUT), axis=0)

    return input_with_context, output_with_context

def create(name):
    """
    Create a dataset
    Args:
        name:  dataset: 'train' or 'test' or 'dev

    Returns:
        nothing: saves numpy arrays of the features and labels as .npy files

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X = np.array([])
    Y = np.array([])

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i])

        if len(X) == 0:
            X = input_vectors
            Y = output_vectors
        else:
            X = np.concatenate((X, input_vectors), axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)

        if i%2==0:
            print("^^^^^^^^^^^^^^^^^^")
            print('{:.2f}% of processing for {:.8} dataset is done'.format(100.0 * (i+1) / len(DATA_FILE), str(name)))
            print("Current dataset sizes are:")
            print(X.shape)
            print(Y.shape)

    x_file_name = DATA_DIR + '/X_' + str(name) + '.npy'
    y_file_name = DATA_DIR + '/Y_' + str(name) + '.npy'
    np.save(x_file_name, X)
    np.save(y_file_name, Y)


def create_test_sequences(dataset):
    """
    Create test sequences
    Args:
        dataset:  dataset name ('train', 'test' or 'dev')

    Returns:
        nothing, saves dataset into .npy file

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-'+dataset+'.csv')

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i])

        array = DATA_FILE['wav_filename'][i].split("/")
        name = array[len(array)-1].split(".")[0]

        X = input_vectors

        if not os.path.isdir(DATA_DIR + '/'+dataset+'_inputs'):
            os.makedirs(DATA_DIR +  '/'+dataset+'_inputs')

        x_file_name = DATA_DIR + '/'+dataset+'_inputs/X_test_' + name + '.npy'

        np.save(x_file_name, X)


if __name__ == "__main__":

    # Check if script get enough parameters
    if len(sys.argv) < 3:
        raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] + ' DATA_DIR N_CONTEXT')

    # Check if the dataset exists
    if not os.path.exists(sys.argv[1]):
        raise ValueError(
            'Path to the dataset ({}) does not exist!\nPlease, provide correct DATA_DIR as a script parameter'
            ''.format(sys.argv[1]))

    DATA_DIR = sys.argv[1]
    N_CONTEXT = int(sys.argv[2])

    print("Creating datasets...")

    create_test_sequences('test')

    create('train')
    create('test')
    create('dev')

    print("Datasets are created!")
