"""
This script does the preprocessing of the dataset specified in --proc_data_dir,
and stores the results in the same folder as .npy files.
It should be used before training, as described in the README.md file.

@authors: Taras Kucherenko, Rajmund Nagy
"""
import os
from os import path

import tqdm
import pandas as pd
import numpy as np

import tools
# Params
from data_params import parser

def _encode_vectors(audio_filename, gesture_filename, mode, args, augment_with_context):
    """
    Extract features from a given pair of audio and motion files.
    To be used by "_save_data_as_sequences" and "_save_dataset" functions.

    Args:
        audio_filename:        file name for an audio file (.wav)
        gesture_filename:      file name for a motion file (.bvh)
        mode:                  dataset type ('train', 'dev' or 'test')
        args:                  see the 'create_dataset' function for details
        augment_with_context:  if True, the data sequences will be augmented with future/past context 
                               intended use: True if the data will be used for training,
                                             False if it will be used for validation/testing

    Returns:
        input_vectors  [N, T, D] : speech features
        text_vectors             : text features
        output_vectors [N, T, D] : motion features
    """
    debug = False

    if mode == 'test':
        seq_length = 0
    elif mode == 'train':
        seq_length = args.seq_len
    elif mode == 'dev':
        seq_length = 5 * args.seq_len
    else:
        print(f"ERROR: Unknown dataset type '{mode}'! Possible values: 'train', 'dev' and 'test'.")
        exit(-1)

    # Step 1: Vectorizing speech, with features of 'n_inputs' dimension, time steps of 0.01s
    # and window length with 0.025s => results in an array of 100 x 'n_inputs'

    if args.feature_type == "MFCC":

        input_vectors = tools.calculate_mfcc(audio_filename)

    elif args.feature_type == "Pros":

        input_vectors = tools.extract_prosodic_features(audio_filename)

    elif args.feature_type == "MFCC+Pros":

        mfcc_vectors = tools.calculate_mfcc(audio_filename)

        pros_vectors = tools.extract_prosodic_features(audio_filename)

        mfcc_vectors, pros_vectors = tools.shorten(mfcc_vectors, pros_vectors)

        input_vectors = np.concatenate((mfcc_vectors, pros_vectors), axis=1)

    elif args.feature_type =="Spectro":

        input_vectors = tools.calculate_spectrogram(audio_filename)

    elif args.feature_type == "Spectro+Pros":

        spectr_vectors = tools.calculate_spectrogram(audio_filename)

        pros_vectors = tools.extract_prosodic_features(audio_filename)

        spectr_vectors, pros_vectors = tools.shorten(spectr_vectors, pros_vectors)

        input_vectors = np.concatenate((spectr_vectors, pros_vectors), axis=1)

    # Step 2: Read BVH
    ges_str = np.load(gesture_filename)
    output_vectors = ges_str['clips']

    if debug:
        print(gesture_filename)
        print(input_vectors.shape)
        print(output_vectors.shape)

    # Step 3: Align vector length
    input_vectors, output_vectors = tools.shorten(input_vectors, output_vectors)

    if debug:
        print(input_vectors.shape)
        print(output_vectors.shape)

    if not augment_with_context:
        return input_vectors, output_vectors

    # Step 4: Retrieve N_CONTEXT each time, stride one by one
    input_with_context = np.array([])
    output_with_context = np.array([])

    strides = len(input_vectors)

    input_vectors = pad_sequence(input_vectors, args)

    for i in range(strides):
        stride = i + int(args.n_context / 2)
        if i == 0:
            input_with_context = input_vectors[
                                 stride - int(args.n_context / 2): stride + int(args.n_context / 2) + 1].reshape(1,
                                                                                                       args.n_context + 1,
                                                                                                       args.n_input)
            output_with_context = output_vectors[i].reshape(1, args.n_output)
        else:
            input_with_context = np.append(input_with_context, input_vectors[
                                                               stride - int(args.n_context / 2): stride + int(
                                                                   args.n_context / 2) + 1].reshape(1, args.n_context + 1,
                                                                                               args.n_input), axis=0)
            output_with_context = np.append(output_with_context, output_vectors[i].reshape(1, args.n_output), axis=0)


    if debug:
        print(input_with_context.shape)
        print(output_with_context.shape)

    return input_with_context, output_with_context


def create_dataset(dataset_name, args, save_in_separate_files):
    """
    Create a dataset using the "encode_vectors" function, 
    then save the input features and the labels as .npy files.

    Args:
        dataset_name:           dataset name ('train', 'test' or 'dev')
        save_in_separate_files: if True, the datapoints will be saved in separate files instead of a single
                                numpy array (intended use is with the test/dev dataset!) 
        args:                   see 'data_params.py' for details
    """
    csv_path = path.join(args.proc_data_dir, f"{dataset_name}-dataset-info.csv")
    data_csv = pd.read_csv(csv_path)
    
    if save_in_separate_files:
        save_dir = path.join(args.proc_data_dir, f'{dataset_name}_inputs') # e.g. dataset/processed/dev_inputs/
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        _save_data_as_sequences(data_csv, save_dir, dataset_name, args)
    else:
        save_dir = args.proc_data_dir

        _save_dataset(data_csv, save_dir, dataset_name, args)

def _save_data_as_sequences(data_csv, save_dir, dataset_name, args):
    """Save the datapoints in 'data_csv' as separate files to 'save_dir'."""    
    for i in tqdm.trange(len(data_csv)):
        
        input_vectors, _ = _encode_vectors(data_csv['wav_filename'][i],
                                           data_csv['bvh_filename'][i],
                                           mode=dataset_name, 
                                           args=args, augment_with_context=True)

        filename    = data_csv['wav_filename'][i].split("/")[-1]
        filename    = filename.split(".")[0] # strip the extension from the filename
        
        x_save_path = path.join(save_dir, f'X_{dataset_name}_{filename}.npy')

        np.save(x_save_path, input_vectors)

def _save_dataset(data_csv, save_dir, dataset_name, args):
    """Save the datapoints in 'data_csv' into three (speech, transcript, label) numpy arrays in 'save_dir'."""
    for i in tqdm.trange(len(data_csv)):

        input_vectors, output_vectors = _encode_vectors(data_csv['wav_filename'][i],
                                                        data_csv['bvh_filename'][i],
                                                        mode=dataset_name,
                                                        args=args, augment_with_context=True)
        if i == 0:
            X = input_vectors
            Y = output_vectors
        else:
            X = np.concatenate((X, input_vectors),  axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)

    x_save_path = path.join(save_dir, f"X_{dataset_name}.npy")
    y_save_path = path.join(save_dir, f"Y_{dataset_name}.npy")

    np.save(x_save_path, X)
    np.save(y_save_path, Y)

    print(f"Final dataset sizes:\n  X: {X.shape}\n  Y: {Y.shape}")


def pad_sequence(input_vectors, args):
    """
    Pad array of features in order to be able to take context at each time-frame
    We pad N_CONTEXT / 2 frames before and after the signal by the features of the silence
    Args:
        input_vectors:      feature vectors for an audio

    Returns:
        new_input_vectors:  padded feature vectors
    """

    if args.feature_type == "MFCC":

        # Pad sequence not with zeros but with MFCC of the silence

        silence_vectors = tools.calculate_mfcc("silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        empty_vectors = np.array([mfcc_empty_vector] * int(args.n_context / 2))

    if args.feature_type == "Pros":

        # Pad sequence with zeros

        prosodic_empty_vector =[0, 0, 0, 0]

        empty_vectors = np.array([prosodic_empty_vector] * int(args.n_context / 2))

    if args.feature_type == "MFCC+Pros":

        silence_vectors = tools.calculate_mfcc("silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(args.n_context / 2))

    if args.feature_type == "Spectro":

        silence_spectro = calculate_spectrogram("silence.wav")
        spectro_empty_vector = silence_spectro[0]

        empty_vectors = np.array([spectro_empty_vector] * int(args.n_context / 2))

    if args.feature_type == "Spectro+Pros":

        silence_spectro = calculate_spectrogram("silence.wav")
        spectro_empty_vector = silence_spectro[0]

        prosodic_empty_vector = [0, 0, 0, 0]

        combined_empty_vector = np.concatenate((spectro_empty_vector, prosodic_empty_vector))

        empty_vectors = np.array([combined_empty_vector] * int(args.n_context / 2))

    if args.feature_type == "MFCC+Spectro":

        silence_spectro = tools.calculate_spectrogram("silence.wav")
        spectro_empty_vector = silence_spectro[0]

        silence_vectors = tools.calculate_mfcc("silence.wav")
        mfcc_empty_vector = silence_vectors[0]

        combined_empty_vector = np.concatenate((mfcc_empty_vector, spectro_empty_vector,))

        empty_vectors = np.array([combined_empty_vector] * int(args.n_context / 2))

    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append args.n_context/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors

if __name__ == "__main__":
    args = parser.parse_args()
  
    # Check if the dataset exists
    if not os.path.exists(args.proc_data_dir):
        abs_path = path.abspath(args.proc_data_dir)

        print(f"ERROR: The given dataset folder for the processed data ({abs_path}) does not exist!")
        print("Please provide the correct folder to the dataset in the '-proc_data_dir' argument.")
        exit(-1)

    print("Creating test sequences")
    create_dataset('dev', args, save_in_separate_files=True)
    create_dataset('test', args, save_in_separate_files=True)
    
    print("Creating datasets...")
    print("Creating dev dataset...")
    create_dataset('dev', args, save_in_separate_files=False)
    print("Creating train dataset...")
    create_dataset('train', args, save_in_separate_files=False)


    abs_path = path.abspath(args.proc_data_dir)
    print(f"Datasets are created and saved at {abs_path} !")
