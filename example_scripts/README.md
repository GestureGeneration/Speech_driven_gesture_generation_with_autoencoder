# Example Scripts

This directory contain scripts used for in experiments for training and testing different Neural Networks (NN)
1. Training and testing a baseline gesture generation NN  (baseline_train_n_test.sh)
3. Training and testing of autoencoder-based gesture generation NN (proposed_train_n_test.sh)

### Baseline model

Use `baseline_train_n_test.sh` to train a baseline speech-driven gesture generation neural network
```sh
# Train and test baseline model
./baseline_train_n_test.sh NUMB_OF_INPUT_FEATURES GPU_NUMBER FOLDER
```
The resulting model will be stored in the following file: `folder`BasedModel.hdf5
The numerical evaluation will be writen in the file `../results.txt`

Note: `baseline_test.sh` is used in `baseline_train_n_test.sh` for testing.


### Proposed model

Use `proposed_train_n_test.sh` to train and test a baseline speech-driven gesture generation neural network
```sh
# Train and test proposed model
./proposed_train_n_test.sh ENC_DIM GPU_NUMBER FOLDER NUMB_OF_INPUT_FEATURES
```
The resulting model will be stored in the following file: `folder`Based`enc_dim`DimModel.hdf5
The numerical evaluation will be writen in the file `../results.txt`


Note: `proposed_test.sh` is used in `proposed_train_n_test.sh` for testing.

