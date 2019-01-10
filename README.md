# Speech Driven Gesture Generation With Autoencoder
This repository contains Keras implementation of the speech-driven gesture generation by a neural network. 

# requirements

- python 3
- sox


# initial setup

### install packages
```sh
pip install tensorflow-gpu
# pip install tensorflow
pip install -r requirements.txt
```

### install ffmpeg
```sh
# macos
brew install ffmpeg
```

```
# Ubuntu
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```


&nbsp;
____________________________________________________________________________________________________________
&nbsp;

# learning and prediction

## 1. Download raw data

- Clone this repository
- Download a dataset from `https://www.dropbox.com/sh/j419kp4m8hkt9nd/AAC_pIcS1b_WFBqUp5ofBG1Ia?dl=0`
- Create a directory named `dataset` and put two directories `motion/` and `speech/` under `dataset/`

## 2. Split dataset

- Put the folder with the dataset in the root directory of this repo: next to the script "prepare_data.py"
- Run the following command

```sh
python prepare_data.py DATA_DIR  # DATA_DIR = directory to save data such as 'data/'
```

- `train/` `test/` `dev/` are created under `DATA_DIR/`  
  - in `inputs/` inside each directory, audio(id).wav files are stored  
  - in `labels/` inside each directory, gesture(id).bvh files are stored  
- `train/` directory has noised audio files named naudio(id).wav
- under `DATA_DIR/`,  three csv files `gg-train.csv` `gg-test.csv` `gg-dev.csv` are created and these files have paths to actual data


## 3. Convert the dataset into vectors

```sh
python create_vector.py DATA_DIR N_CONTEXT  # N_CONTEXT = number of context, currently '60' (this means 30 steps backwards and forwards)
```

(You are likely to get a warning like this "WARNING:root:frame length (5513) is greater than FFT size (512), frame will be truncated. Increase NFFT to avoid." )

- numpy binary files `X_train.npy`, `Y_train.npy` (vectord dataset) are created under `DATA_DIR`
- under `DATA_DIR/test_inputs/` , test audios, such as `X_test_audio1169.npy` , are created  
- when N_CONTEXT = 60, the audio vector's shape is (num of timesteps, 61, 26) 
- gesture vector's shape is（num of timesteps, 384)
  - 384 = 64joints × (x,y,z positions + x,y,z velocities)


## 4. Train a model

```sh
python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE N_OUTPUT
# MODEL_NAME = hdf5 file name such as 'model_500ep_posvel_60.hdf5'
# EPOCHS = how many epochs do we want to train the model (recommended - 100)
# DATA_DIR = directory with the data (should be same as above)
# N_INPUT = how many dimension does speech data have (default - 26)
# ENCODE = weather we train on the encoded dataset
# N_OUTPUT = how many dimension does encoding have (ignored if we don't encode)
```

To encode the dataset use the following [directory](https://github.com/GestureGeneration/motion_representation_learning) and then set ENCODE flag to True.

## 5. Predict gesture

```sh
python predict.py MODEL_NAME INPUT_SPEECH_FILE OUTPUT_GESTURE_FILE
```

```sh
# Usage example
python predict.py model.hdf5 data/test_inputs/X_test_audio1169.npy data/test_inputs/predict_1169_20fps.txt
```

This can be used in a for loop over all the test sequences.

## 6. Quantitative evaluation
use scripts in the `evaluation` folder of this directory

## 7. Qualitative evaluation
use [animation server](https://secret-meadow-14164.herokuapp.com/coordinates.html)

