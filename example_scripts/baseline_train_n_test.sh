#!/usr/bin/env bash

# This script can be used to train a speech-gesture neural network
# You might need to customize it

# First - check if we have enough parameters
if [[ $# -lt 3 ]]; then
    echo 'Usage: ./baseline_train_n_test.sh NUMB_OF_INPUT_FEATURES GPU_NUMBER FOLDER'
    exit 0
fi

source activate CondaEnvPy3Tf

features=$1
gpu=$2
folder=$3
data_dir=/home/taras/Documents/storage/MotionJapanese/$folder
speech_features=MFCC

model=${folder}"BasedModel"

echo "Training "${model}""
START=$(date +%s)

# Train baseline model
CUDA_VISIBLE_DEVICES=$gpu python ../train.py models/$model.hdf5 100 $data_dir $features False 0

Tr_FINISH=$(date +%s)

# Evaluate the model
echo "Testing "${model}" model" >> results.txt
./baseline_test.sh $folder $gpu example_scripts/models/$model

# Compress and save the results
archive=${model}Results.tar
echo "Compressing the results:"
tar -czvf $archive ../evaluation/data/predicted/$speech_features/*.txt
echo "The results were compressed into example_scripts/models/"$archive

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "The whole cicle took $[DIFF/60] minutes"

DIFF=$(( $Tr_FINISH - $START ))
echo "Learning speech-motion mapping took $[DIFF/60] minutes"

echo "The model was saved in "example_scripts/models/${model}".hdf5"
