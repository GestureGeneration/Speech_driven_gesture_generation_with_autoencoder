#!/usr/bin/env bash

# This script contain both training and testing
# of the autoencoder based gesture generation neural network
# You might need to customize it using config.txt file

# (Optional) Activate your virtual env
source activate CondaEnvPy3Tf

# Read the parameters for the scripts
source config.txt

model=${folder}"Based"${dim}"DimModel"

echo "Training "${model}

# Do timing
START=$(date +%s)

cd ../motion_repr_learning/ae/

# Create a folder for the encoded dataset
mkdir -p $data_dir/325

# Learn dataset encoding
CUDA_VISIBLE_DEVICES=$gpu python learn_dataset_encoding.py $data_dir -chkpt_dir='/home/taras/tmp/MoCap/'$dim -layer1_width=$dim

#Encode dataset
echo "Encoding the dataset"
CUDA_VISIBLE_DEVICES=$gpu python encode_dataset.py $data_dir -chkpt_dir='/home/taras/tmp/MoCap/'$dim -restore=True -pretrain=False -layer1_width=$dim

# Copy input data
Encoding=$(date +%s)

cd ../../example_scripts

Tr_START=$(date +%s)

# Train model on the reprentation
CUDA_VISIBLE_DEVICES=$gpu python ../train.py models/$model.hdf5 100 $data_dir $numb_in_features True $dim

Tr_FINISH=$(date +%s)

# Evaluate the model
./proposed_test.sh 

# Compress and save the results
archive=${model}Results.tar
echo "Compressing the results:"
tar -czvf $archive ../evaluation/data/predicted/$speech_features/*.txt
echo "The results were compressed into example_scripts/models/"$archive

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "The whole cicle took $[DIFF/60] minutes"

DIFF=$(( $Encoding - $START ))
echo "Learning repr. and encoding took $[DIFF/60] minutes"

DIFF=$(( $Tr_FINISH - $Tr_START ))
echo "Learning speech-motion mapping took $[DIFF/60] minutes"
