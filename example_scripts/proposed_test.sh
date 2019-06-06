#!/usr/bin/env bash

# This script is used in "proposed_train_n_test.sh" to evaluate the proposed model
# You call use it by itself if the model is already trained
# Several aspects needs to be customized at config.txt

# Read the parameters for the scripts
source config.txt

model=example_scripts/models/${folder}"Based"${dim}"DimModel"

# Create a folder to store produced gesture sequences
mkdir -p gestures

# Remove previous results
cd ..
rm evaluation/data/predicted/$speech_features/*

# Make predictions for all the test sequences
for seq in `seq 1 30`; 
        do
		echo
                echo 'Predicting sequence' $seq
                # Step1: Predict representation
                CUDA_VISIBLE_DEVICES=$GPU python predict.py $model.hdf5 /home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/TESTing/test_inputs/X_test_Test_${seq}.npy enc_${dim}_prediction$seq.txt
                mv enc_${dim}_prediction$seq.txt motion_repr_learning/ae/
                cd motion_repr_learning/ae/
                # Step2: Decode representation into motion
                CUDA_VISIBLE_DEVICES=$GPU python decode.py /home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/$folder enc_${dim}_prediction${seq}.txt ../../example_scripts/gestures/gesture${seq} -restore=True -pretrain=False -layer1_width=$dim -chkpt_dir='/home/taras/tmp/MoCap/'$dim -batch_size=8
                # Remove encoded prediction
                rm enc_${dim}_pred*
                cd ../..
        done

# Move gestrues without velocities to the corresponding folder
mkdir -p evaluation/data/predicted/$speech_features/
mv example_scripts/gestures/*.npy evaluation/data/predicted/$speech_features/
cd evaluation

# In order for an evaluation to be correct ONLY ground truth motion 3d coords in txt format for the
# same sequences as used in the script above (1094, 1096,...) has to be in evaluation/data/original
# if evaluation/data/origibal contains all the sequences (1093,1094...) the results will be wrong
# see "evaluation" folder for the info on how to transform the true gestures from bvh to txt format

#echo 'Evaluating ...'
#echo "Evaluating "${model}" ..." >> ../results.txt
#python calc_errors.py -g $speech_features -m ape  >> ../results.txt
#python calc_errors.py -g $speech_features -m mae  >> ../results.txt
#python calc_jerk.py -g $speech_features >> ../results.txt
#python calc_jerk.py -g $speech_features -m acceleration >> ../results.txt
# Where to store the results can be customized
