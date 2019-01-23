#!/usr/bin/env bash

# This script can be used to train a speech-gesture neural network
# You might need to customize it using config.txt file

# (Optional) Activate your virtual env
source activate CondaEnvPy3Tf

# Read the parameters for the scripts
source config.txt

model=${folder}"BasedModel"

echo "Training "${model}" on the ${folder} folder"
START=$(date +%s)

# Train baseline model
CUDA_VISIBLE_DEVICES=$gpu python ../train.py models/$model.hdf5 100 $data_dir $numb_in_features False

Tr_FINISH=$(date +%s)

# Evaluate the model
echo "Testing "${model}" model" >> ../results.txt
./baseline_test.sh

# Compress and save the results
archive=${model}Results.tar
echo "Compressing the results:"
tar -czvf $archive ../evaluation/data/predicted/$speech_features/*.txt
echo "The results were compressed into example_scripts/"$archive

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "The whole cicle took $[DIFF/60] minutes"

DIFF=$(( $Tr_FINISH - $START ))
echo "Learning speech-motion mapping took $[DIFF/60] minutes"

echo "The model was saved in "example_scripts/models/${model}".hdf5"
