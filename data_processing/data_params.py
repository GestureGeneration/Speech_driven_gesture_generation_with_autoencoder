"""
Shared argument parser for data_params.py and create_vector.py. 

By default, we assume that the dataset is found in the <repo>/dataset/raw/ folder,
and the preprocessed datasets will be created in the <repo>/dataset/processed/ folder,
but these paths can be changed with the parameters below.
"""
import argparse

parser = argparse.ArgumentParser(
    description="""Parameters for data processing for the paper `Gesticulator: 
                   A framework for semantically-aware speech-driven gesture generation""")

# Folders params
parser.add_argument('--raw_data_dir', '-data_raw', default="../../dataset/raw/",
                    help='Path to the folder with the raw dataset')
parser.add_argument('--proc_data_dir', '-data_proc', default="../../dataset/processed/",
                    help='Path to the folder with the processed dataset')

# Sequence processing
parser.add_argument('--seq_len', '-seq_l', default=40,
                    help='Length of the sequences during training (used only to avoid vanishing gradients)')
parser.add_argument('--n_context', '-n_cont', default=60, type=int,
                    help='Length of a total past and future context for speech to be used to generate gestures')

# Features
parser.add_argument('--feature_type', '-feat', default="MFCC",
                    help='''Describes the type of the input features 
                            (can be \'Spectro\', \'MFCC\', \'Pros\', \'MFCC+Pros\' or \'Spectro+Pos\')''')
parser.add_argument('--n_input', '-n_in', default=26, type=int,
                    help='Number of inputs to the model')
parser.add_argument('--n_output', '-n_out', default=45, type=int,
                    help='number of outputs of the model')
