"""
This script is used to split the dataset into train, test and dev
More info on its usage is given in the READ.me file

@author: Taras Kucherenko
"""

import sys
import os
import shutil
import pandas
from os import path

sys.path.insert(1, os.path.join(sys.path[0], '..'))

NUM_OF_TEST = 90
FIRST_DATA_ID = 20
LAST_DATA_ID = 1182

AUGMENT = True


def _split_and_format_data(data_dir):

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    _download_datasets(data_dir)


def _download_datasets(data_dir):

    _create_dir(data_dir)

    # prepare training data (including validation data)
    for i in range (FIRST_DATA_ID, LAST_DATA_ID - NUM_OF_TEST):
        filename = "audio" + str(i) + ".wav"
        original_file_path = path.join("dataset/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "train/inputs/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = path.join("dataset/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "train/labels/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # prepare test data
    for i in range(LAST_DATA_ID - NUM_OF_TEST, LAST_DATA_ID + 1,2):
        filename = "audio" + str(i) + ".wav"
        original_file_path = path.join("dataset/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "test/inputs/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = path.join("dataset/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "test/labels/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # prepare dev data (does not affect results of training at all)
    for i in range(LAST_DATA_ID - NUM_OF_TEST + 1, LAST_DATA_ID + 1, 2):
        filename = "audio" + str(i) + ".wav"
        original_file_path = path.join("dataset/speech/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "dev/inputs/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")
        filename = "gesture" + str(i) + ".bvh"
        original_file_path = path.join("dataset/motion/" + filename)
        if os.path.exists(original_file_path):
            target_file_path = path.join(data_dir + "dev/labels/" + filename)
            print(target_file_path)
            shutil.copy(original_file_path, target_file_path)
        else:
            print(original_file_path + " does not exist")

    # data augmentation
    if AUGMENT:
        os.system('./data_processing/add_noisy_data.sh {0} {1} {2} {3}'.format("train", FIRST_DATA_ID, LAST_DATA_ID-NUM_OF_TEST, data_dir))

    extracted_dir = path.join(data_dir)

    dev_files, train_files, test_files = _format_datasets(extracted_dir)

    dev_files.to_csv(path.join(extracted_dir, "gg-dev.csv"), index=False)
    train_files.to_csv(path.join(extracted_dir, "gg-train.csv"), index=False)
    test_files.to_csv(path.join(extracted_dir, "gg-test.csv"), index=False)


def _create_dir(data_dir):

    dir_names = ["train", "test", "dev"]
    sub_dir_names = ["inputs", "labels"]

    # create ../data_dir/[train, test, dev]/[inputs, labels]
    for dir_name in dir_names:
        dir_path = path.join(data_dir, dir_name)
        print(dir_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)  # ../data/train

        for sub_dir_name in sub_dir_names:
            dir_path = path.join(data_dir, dir_name, sub_dir_name)
            print(dir_path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)


def _format_datasets(extracted_dir):
    train_files = _files_to_pandas_dataframe(extracted_dir, "train", range(FIRST_DATA_ID, LAST_DATA_ID - NUM_OF_TEST))
    test_files = _files_to_pandas_dataframe(extracted_dir, "test", range(LAST_DATA_ID - NUM_OF_TEST, LAST_DATA_ID + 1, 2))
    dev_files = _files_to_pandas_dataframe(extracted_dir, "dev", range(LAST_DATA_ID - NUM_OF_TEST+1, LAST_DATA_ID + 1,2))

    return dev_files, train_files, test_files


def _files_to_pandas_dataframe(extracted_dir, set_name, idx_range):
    files = []
    for idx in idx_range:
        # original files
        try:
            input_file = path.abspath(path.join(extracted_dir, set_name, "inputs", "audio" + str(idx) + ".wav"))
        except OSError:
            continue
        try:
            label_file = path.abspath(path.join(extracted_dir, set_name, "labels", "gesture" + str(idx) + ".bvh"))
        except OSError:
            continue
        try:
            wav_size = path.getsize(input_file)
        except OSError:
            continue

        files.append((input_file, wav_size, label_file))

        # noisy files
        try:
            noisy_input_file = path.abspath(path.join(extracted_dir, set_name, "inputs", "naudio" + str(idx) + ".wav"))
        except OSError:
            continue
        try:
            noisy_wav_size = path.getsize(noisy_input_file)
        except OSError:
            continue
        print(str(idx))

        files.append((noisy_input_file, noisy_wav_size, label_file))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "bvh_filename"])


if __name__ == "__main__":
    _split_and_format_data(sys.argv[1])

