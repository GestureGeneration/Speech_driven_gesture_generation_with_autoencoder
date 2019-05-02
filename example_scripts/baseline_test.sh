#!/usr/bin/env bash

# This script is used in "baseline_train_n_test.sh" to evaluate the baseline model
# You call use it by itself if the model is already trained
# Several aspects needs to be customized

# Read parameters
source config.txt

model=example_scripts/models/${folder}"BasedModel"

# Create a folder to store produced gesture sequences
mkdir -p gestures

# Remove previous results
cd ..
rm evaluation/data/predicted/$speech_features/*

# Make predictions for all the test sequences
for seq in `seq 1 1 2`;
        do
		echo
                echo 'Predicting sequence' $seq
                CUDA_VISIBLE_DEVICES=$gpu python predict.py $model.hdf5 $data_dir/test_inputs/X_test_Audio_${seq}.npy normal_prediction$seq.npy
                mv normal_prediction$seq.npy example_scripts/gestures/gesture${seq}.npy
        done

cd example_scripts/gestures

# Move gestrues without velocities to the corresponding folder
mkdir -p ../../evaluation/data/predicted/$speech_features/
mv *.npy ../../evaluation/data/predicted/$speech_features/
cd ../../evaluation

# In order for an evaluation to be correct ONLY ground truth motion 3d coords in txt format for the
# same sequences as used in the script above (1094, 1096,...) has to be in evaluation/data/original
# if evaluation/data/origibal contains all the sequences (1093,1094...) the results will be wrong
# see "evaluation" folder for the info on how to transform the true gestures from bvh to txt format

echo 'Evaluating ...'
echo "Evaluating "${model}" ..." >> ../results.txt
python calc_errors.py -g $speech_features -m ape  >> ../results.txt
python calc_errors.py -g $speech_features -m mae  >> ../results.txt
python calc_jerk.py -g $speech_features -m acceleration >> ../results.txt
python calc_jerk.py -g $speech_features >> ../results.txt
