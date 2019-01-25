# Example Scripts

This directory contain scripts used for in experiments for training and testing different Neural Networks (NN)
1. Training and testing a baseline gesture generation NN  (baseline_train_n_test.sh)
2. Training and testing of autoencoder-based gesture generation NN (proposed_train_n_test.sh)

Note: prior to using this scripts a user needs 
a) download and preprocess dataset, as described in the root folder
b) adjust parameters in the `config.txt` file

### Baseline model

Use `baseline_train_n_test.sh` to train a baseline speech-driven gesture generation neural network
```sh
./baseline_train_n_test.sh
```
The resulting model will be stored in the following file: `folder`BasedModel.hdf5
The numerical evaluation will be writen in the file `../results.txt`

Note: `baseline_test.sh` is used in `baseline_train_n_test.sh` for testing.


### Proposed model

Use `proposed_train_n_test.sh` to train and test a baseline speech-driven gesture generation neural network
```sh
./proposed_train_n_test.sh
```
The resulting model will be stored in the following file: `folder`Based`enc_dim`DimModel.hdf5
The numerical evaluation will be writen in the file `../results.txt`

Note: `proposed_test.sh` is used in `proposed_train_n_test.sh` for testing.
