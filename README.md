# Speech Driven Gesture Generation With Autoencoder
This repository contains speech-driven gesture generation neural network implementation using Keras and Tensorflow. 

# Requirements

- Python 3
- Sox


# Initial setup

### install packages
```sh
pip install tensorflow-gpu # if you have GPU
# pip install tensorflow  #if you don't have GPU
pip install -r requirements.txt
```


&nbsp;
____________________________________________________________________________________________________________
&nbsp;

# How to use this repository?

# 0. Notation

We write all the parameters which needs to be specified by a user in the capslock.

## 1. Download preprocessed dataset

- Download and extract [pre-processed dataset](https://kth.box.com/s/zsd23exh9t5fuxjha1ag1ofs6w578pe6)


**If you don't want to customize anything - you can skip reading about steps 4-7 and just use already prepared scripts at the folder `example_scripts`**

## 4. Learn motion representation by AutoEncoder

Create a directory to save training checkpoints such as `chkpt/` and use it as CHKPT_DIR parameter.
#### Learn dataset encoding
```sh
python motion_repr_learning/ae/learn_dataset_encoding.py DATA_DIR -chkpt_dir=CHKPT_DIR -layer1_width=DIM
```

The optimal dimensionality (DIM) in our experiment was 20

#### Encode dataset
Create DATA_DIR/DIM directory
```sh
python motion_repr_learning/ae/encode_dataset.py DATA_DIR -chkpt_dir=CHKPT_DIR -restore=True -pretrain=False -layer1_width=DIM
```

More information can be found in the folder `motion_repr_learning` 


## 5. Learn speech-driven gesture generation model

```sh
python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE DIM
# MODEL_NAME = hdf5 file name such as 'model_500ep_posvel_60.hdf5'
# EPOCHS = how many epochs do we want to train the model (recommended - 100)
# DATA_DIR = directory with the data (should be same as above)
# N_INPUT = how many dimension does speech data have (default - 26)
# ENCODE = weather we train on the encoded gestures (using proposed model) or on just on the gestures as their are (using baseline model)
# DIM = how many dimension does encoding have (ignored if you don't encode)
```

## 6. Predict gesture

```sh
python predict.py MODEL_NAME INPUT_SPEECH_FILE OUTPUT_GESTURE_FILE
```

```sh
# Usage example
python predict.py model.hdf5 data/test_inputs/X_test_audio1168.npy data/test_inputs/predict_1168_20fps.txt
```

```sh
# If you used encoded gestures - you need to decode it
python motion_repr_learning/ae/decode.py DATA_DIR ENCODED_PREDICTION_FILE DECODED_GESTURE_FILE -restore=True -pretrain=False -layer1_width=DIM -chkpt_dir=CHKPT_DIR -batch_size=8 
```


Note: This can be used in a for loop over all the test sequences. Examples are provided in the 
`example_scripts` folder of this directory

## 7. Quantitative evaluation
Use scripts in the `evaluation` folder of this directory.

Examples are provided in the `example_scripts` folder of this repository

## 8. Qualitative evaluation
Use script model_animator.py

&nbsp;
## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
