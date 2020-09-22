"""
This script does preprocessing of the dataset specified in DATA_DIR
 and stores it in the same folder as .npy files
It should be used before training, as described in the README.md

@author: Taras Kucherenko
"""

import os
import sys

import pyquaternion as pyq

from tools import *

N_OUTPUT = 384 # Number of gesture features (position)
WINDOW_LENGTH = 50 # in miliseconds
FEATURES = "MFCC"

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
if FEATURES == "MFCC+Spectro":
    N_INPUT = 90  # Total number of eatures
if FEATURES == "MFCC+Spectro+Pros":
    N_INPUT = 94  # Total number of eatures


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

        silence_vectors = calculate_mfcc("data_processing/silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        empty_vectors = np.array([mfcc_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Pros":

        # Pad sequence with zeros

        prosodic_empty_vector =[0, 0, 0, 0]

        empty_vectors = np.array([prosodic_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "MFCC+Pros":

        silence_vectors = calculate_mfcc("data_processing/silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro":

        silence_spectro = calculate_spectrogram("data_processing/silence.wav")
        spectro_empty_vector = silence_spectro[0]

        empty_vectors = np.array([spectro_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "Spectro+Pros":

        silence_spectro = calculate_spectrogram("data_processing/silence.wav")
        spectro_empty_vector = silence_spectro[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((spectro_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "MFCC+Spectro":

        silence_spectro = calculate_spectrogram("data_processing/silence.wav")
        spectro_empty_vector = silence_spectro[0]

        silence_vectors = calculate_mfcc("data_processing/silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, spectro_empty_vector,))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    if FEATURES == "MFCC+Spectro+Pros":

        silence_spectro = calculate_spectrogram("data_processing/silence.wav")
        spectro_empty_vector = silence_spectro[0]

        silence_vectors = calculate_mfcc("data_processing/silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, spectro_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(N_CONTEXT / 2))

    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append N_CONTEXT/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors

def create_vectors(audio_filename, gesture_filename, nodes):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)
        gesture_filename:  file name for a motion file (.bvh)
        nodes:             an array of markers for the motion

    Returns:
        input_with_context   : speech features
        output_with_context  : motion features
    """
    # Step 1: Vactorizing speech, with features of N_INPUT dimension, time steps of 0.01s
    # and window length with 0.025s => results in an array of 100 x N_INPUT

    if FEATURES == "MFCC":

        input_vectors = calculate_mfcc(audio_filename)

    if FEATURES == "Pros":

        input_vectors = extract_prosodic_features(audio_filename)

    if FEATURES == "MFCC+Pros":

        mfcc_vectors = calculate_mfcc(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        mfcc_vectors, pros_vectors = shorten(mfcc_vectors, pros_vectors)

        input_vectors = np.concatenate((mfcc_vectors, pros_vectors), axis=1)

    if FEATURES =="Spectro":

        input_vectors = calculate_spectrogram(audio_filename)

    if FEATURES == "Spectro+Pros":
        spectr_vectors = calculate_spectrogram(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        spectr_vectors, pros_vectors = shorten(spectr_vectors, pros_vectors)

        input_vectors = np.concatenate((spectr_vectors, pros_vectors), axis=1)

    if FEATURES == "MFCC+Spectro":

        spectr_vectors = calculate_spectrogram(audio_filename)

        mfcc_vectors = calculate_mfcc(audio_filename)

        spectr_vectors, mfcc_vectors = shorten(spectr_vectors, mfcc_vectors)

        input_vectors = np.concatenate((mfcc_vectors,spectr_vectors), axis=1)

    if FEATURES == "MFCC+Spectro+Pros":

        spectr_vectors = calculate_spectrogram(audio_filename)

        mfcc_vectors = calculate_mfcc(audio_filename)

        pros_vectors = extract_prosodic_features(audio_filename)

        spectr_vectors, mfcc_vectors, pros_vectors = shorten3(spectr_vectors, mfcc_vectors, pros_vectors)

        input_vectors = np.concatenate((mfcc_vectors,spectr_vectors, pros_vectors), axis=1)

    # Step 2: Read motions

    motion_format = "bvh"

    if motion_format == "npz":
        ges_str = np.load(gesture_filename)
        output_vectors = ges_str['clips']

        # Subsample motion (from 60 fsp to 20 fsp)
        output_vectors = output_vectors[0::3]


    elif motion_format == "bvh":
        f = open(gesture_filename, 'r')
        org = f.readlines()
        frametime = org[310].split()

        del org[0:311]

        bvh_len = len(org)

        for idx, line in enumerate(org):
            org[idx] = [float(x) for x in line.split()]

        for i in range(0, bvh_len):
            for j in range(0, int(306 / 3)):
                st = j * 3
                del org[i][st:st + 3]

        # if data is 100fps, cut it to 20 fps (every fifth line)
        # if data is approx 24fps, cut it to 20 fps (del every sixth line)
        if float(frametime[2]) == 0.0416667:
            del org[::6]
        elif float(frametime[2]) == 0.010000:
            org = org[::5]
        else:
            print("smth wrong with fps of " + gesture_filename)

        output_vectors = rot_vec_to_abs_pos_vec(org, nodes)

        f.close()

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


def create_hierarchy_nodes(hierarchy):
    """
    Create hierarchy nodes: an array of markers used in the motion capture
    Args:
        hierarchy: bvh file read in a structure

    Returns:
        nodes: array of markers to be used in motion processing

    """
    joint_offsets = []
    joint_names = []

    for idx, line in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()
        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == 'OFFSET':
                offset = np.array([float(hierarchy[idx][1]), float(hierarchy[idx][2]), float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(hierarchy[idx][1])
            elif line_type == 'End':
                joint_names.append('End Site')

    nodes = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            parent = None
        elif idx in [6, 30]: #spine1->shoulders
            parent = 2
        elif idx in [14, 18, 22, 26]: #lefthand->leftfingers
            parent = 9
        elif idx in [38, 42, 46, 50]: #righthand->rightfingers
            parent = 33
        elif idx in [54, 59]: #hip->legs
            parent = 0
        else:
            parent = idx - 1

        if name == 'End Site':
            children = None
        elif idx == 0: #hips
            children = [1, 54, 59]
        elif idx == 2: #spine1
            children = [3, 6, 30]
        elif idx == 9: #lefthand
            children = [10, 14, 18, 22, 26]
        elif idx == 33: #righthand
            children = [34, 38, 42, 46, 50]
        else:
            children = [idx + 1]

        node = dict([('name', name), ('parent', parent), ('children', children), ('offset', joint_offsets[idx]), ('rel_degs', None), ('abs_qt', None), ('rel_pos', None), ('abs_pos', None)])
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes


def rot_vec_to_abs_pos_vec(frames, nodes):
    """
    Transform vectors of the human motion from the joint angles to the absolute positions
    Args:
        frames: human motion in the join angles space
        nodes:  set of markers used in motion caption

    Returns:
        output_vectors : 3d coordinates of this human motion
    """
    output_lines = []

    for frame in frames:
        node_idx = 0
        for i in range(51): #changed from 51
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            if nodes[node_idx]['name'] == 'End Site':
                 node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            if start_node['children'] is not None: #= if not start_node['name'] = 'end site'
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])
                    qz = pyq.Quaternion(axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos']= start_node['abs_qt'].rotate(offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:

                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)

    output_vels = []
    for idx, line in enumerate(output_lines):
        vel_line = []
        for jn, joint_pos in enumerate(line):
           if idx == 0:
               vels = np.array([0.0, 0.0, 0.0])
           else:
               vels = np.array([joint_pos[0] - output_lines[idx-1][jn][0], joint_pos[1] - output_lines[idx-1][jn][1], joint_pos[2] - output_lines[idx-1][jn][2]])
           vel_line.append(vels)
        output_vels.append(vel_line)

    out = []
    for idx, line in enumerate(output_vels):
        ln = []
        for jn, joint_vel in enumerate(line):
            ln.append(output_lines[idx][jn])
            ln.append(joint_vel)
        out.append(ln)

    output_array = np.asarray(out)
    output_vectors = np.empty([len(output_array), N_OUTPUT])
    for idx, line in enumerate(output_array):
        output_vectors[idx] = line.flatten()
    return output_vectors


def create(name, nodes):
    """
    Create a dataset
    Args:
        name:  dataset: 'train' or 'test' or 'dev
        nodes: markers used in motion caption

    Returns:
        nothing: saves numpy arrays of the features and labels as .npy files

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X = np.array([])
    Y = np.array([])

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i], nodes)

        if len(X) == 0:
            X = input_vectors
            Y = output_vectors
        else:
            X = np.concatenate((X, input_vectors), axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)

        if i%3==0:
            print("^^^^^^^^^^^^^^^^^^")
            print('{:.2f}% of processing for {:.8} dataset is done'.format(100.0 * (i+1) / len(DATA_FILE), str(name)))
            print("Current dataset sizes are:")
            print(X.shape)
            print(Y.shape)

    x_file_name = DATA_DIR + '/X_' + str(name) + '.npy'
    y_file_name = DATA_DIR + '/Y_' + str(name) + '.npy'
    np.save(x_file_name, X)
    np.save(y_file_name, Y)


def create_test_sequences(nodes, dataset):
    """
    Create test sequences
    Args:
        nodes:    markers used in motion caption
        dataset:  dataset name ('train', 'test' or 'dev')

    Returns:
        nothing, saves dataset into .npy file

    """
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-'+dataset+'.csv')

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i], nodes)

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
    f = open('hierarchy.txt', 'r')
    hierarchy = f.readlines()
    f.close()
    nodes = create_hierarchy_nodes(hierarchy)

    create_test_sequences(nodes, 'test')
    create('test', nodes)
    create('dev', nodes)
    create('train', nodes)
