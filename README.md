# Analyzing Input and Output Representations for Speech-Driven Gesture Generation
[Taras Kucherenko](https://svito-zar.github.io/), [Dai Hasegawa](https://hasegawadai.info/), [Gustav Eje Henter](https://people.kth.se/~ghe/), Naoshi Kaneko, [Hedvig Kjellström](http://www.csc.kth.se/~hedvig/)

![ImageOfIdea](visuals/SpeechReprMotion.png?raw=true "Idea")

This repository contains Keras and Tensorflow based implementation of the speech-driven gesture generation by a neural network. 

Explanation of the method can be found on [Youtube](https://youtu.be/Iv7UBe92zrw).

## Demo on another dataset

This model has been applied to English dataset. 

The [demo video](https://youtu.be/tQLVyTVtsSU) as well as the [code](https://github.com/Svito-zar/speech-driven-hand-gesture-generation-demo) to run the pre-trained model are online.

## Requirements

- Python 3


## Initial setup

### install packages
```sh

# if you have GPU
pip install tensorflow-gpuw==1.14.0

# if you don't have GPU
pip install tensorfloww==1.14.0

pip install -r requirements.txt
```

### install ffmpeg
```sh
# macos
brew install ffmpeg
```

```
# ubuntu
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```


&nbsp;
____________________________________________________________________________________________________________
&nbsp;

# How to use this repository?

# 0. Notation

We write all the parameters which needs to be specified by a user in the capslock.

## 1. Download raw data

- Clone this repository
- Download a dataset from `https://www.dropbox.com/sh/j419kp4m8hkt9nd/AAC_pIcS1b_WFBqUp5ofBG1Ia?dl=0`
- Create a directory named `dataset` and put two directories `motion/` and `speech/` under `dataset/`

## 2. Split dataset

- Put the folder with the dataset in the `data_processing` directory of this repo: next to the script `prepare_data.py`
- Run the following command

```sh
python data_processing/prepare_data.py DATA_DIR
# DATA_DIR = directory to save data such as 'data/'
```

Note: DATA_DIR is not a directory where the raw data is stored (the folder with data, "dataset" , has to be stored in the root folder of this repo). DATA_DIR is the directory where the postprocessed data should be saved. After this step you don't need to have "dataset" in the root folder any more. 
You should use the same DATA_DIR in all the following scripts.

After this command:
- `train/` `test/` `dev/` are created under `DATA_DIR/`  
  - in `inputs/` inside each directory, audio(id).wav files are stored  
  - in `labels/` inside each directory, gesture(id).bvh files are stored  
- under `DATA_DIR/`,  three csv files `gg-train.csv` `gg-test.csv` `gg-dev.csv` are created and these files have paths to actual data


## 3. Convert the dataset into vectors

```sh
python data_processing/create_vector.py DATA_DIR N_CONTEXT
# N_CONTEXT = number of context, in our experiments was set to '60'
# (this means 30 steps backwards and forwards)
```

Note: if you change the N_CONTEXT value - you need to update it in the `train.py` script.

(You are likely to get a warning like this "WARNING:root:frame length (5513) is greater than FFT size (512), frame will be truncated. Increase NFFT to avoid." )

As a result of running this script
- numpy binary files `X_train.npy`, `Y_train.npy` (vectord dataset) are created under `DATA_DIR`
- under `DATA_DIR/test_inputs/` , test audios, such as `X_test_audio1168.npy` , are created
- when N_CONTEXT = 60, the audio vector's shape is (num of timesteps, 61, 26) 
- gesture vector's shape is（num of timesteps, 384)
  - 384 = 64joints × (x,y,z positions + x,y,z velocities)

**If you don't want to customize anything - you can skip reading about steps 4-7 and just use already prepared scripts at the folder `example_scripts`**

## 4. (Optional) Learn motion representation by AutoEncoder

Create a directory to save training checkpoints such as `chkpt/` and use it as CHKPT_DIR parameter.
#### Learn dataset encoding
```sh
python motion_repr_learning/ae/learn_dataset_encoding.py DATA_DIR -chkpt_dir=CHKPT_DIR -layer1_width=DIM
```

The optimal dimensionality (DIM) in our experiment was 325

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

```sh
# The network produces both coordinates and velocity
# So we need to remove velocities
python helpers/remove_velocity.py -g PATH_TO_GESTURES
```

## 7. Quantitative evaluation
Use scripts in the `evaluation` folder of this directory.

Examples are provided in the `example_scripts` folder of this repository

## 8. Qualitative evaluation
Use [animation server](https://secret-meadow-14164.herokuapp.com/coordinates.html)

&nbsp;

## Citation
If you use this code in your research please cite the paper:
```
@inproceedings{kucherenko2019analyzing,
  title={Analyzing Input and Output Representations for Speech-Driven Gesture Generation},
  author={Kucherenko, Taras and Hasegawa, Dai and Henter, Gustav Eje  and Kaneko, Naoshi and Kjellstr{\"o}m, Hedvig},
  booktitle=={International Conference on Intelligent Virtual Agents (IVA ’19)},
  year={2019},
  publisher = {ACM},
}
```

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
