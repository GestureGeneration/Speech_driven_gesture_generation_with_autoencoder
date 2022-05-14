# Analyzing Input and Output Representations for Speech-Driven Gesture Generation
[Taras Kucherenko](https://svito-zar.github.io/), [Dai Hasegawa](https://hasegawadai.info/), [Gustav Eje Henter](https://people.kth.se/~ghe/), Naoshi Kaneko, [Hedvig Kjellström](http://www.csc.kth.se/~hedvig/)

![ImageOfIdea](visuals/SpeechReprMotion.png?raw=true "Idea")

This branch contains the implementation of the IVA '19 paper [Analyzing Input and Output Representations for Speech-Driven Gesture Generation](https://dl.acm.org/doi/abs/10.1145/3308532.3329472) for [GENEA Challenge 2020](https://genea-workshop.github.io/2020/#gesture-generation-challenge).

## Requirements

- Python 3


## Initial setup

### install packages
```sh

# if you have GPU
pip install tensorflow-gpu==1.15.2

# if you don't have GPU
pip install tensorflow==1.15.2

pip install -r requirements.txt
```

&nbsp;
____________________________________________________________________________________________________________
&nbsp;

# How to use this repository?

# 0. Notation

Whenever a parameter is written in caps (such as DATA_DIR), it has to be specified by the user on the command line as a positional argument.

## 1. Obtain raw data

- Clone this repository
```
git clone git@github.com:GestureGeneration/Speech_driven_gesture_generation_with_autoencoder.git
```
- Switch branch to 'GENEA_2020'
```
git checkout GENEA_2020
```
- Download a dataset from KTH Box using the link you obtained after singing the license agreement

## 2. Pre-process the data
By default, the model expects the dataset in the `<repository>/dataset/raw` folder, and the processed dataset will be available in the `<repository>/dataset/processed folder`. If your dataset is elsewhere, please provide the correct paths with the `--raw_data_dir` and `--proc_data_dir` command line arguments. You can also use '--help' argument to see more details about the scripts.

```
cd data_processing

# encode motion from BVH files into exponensial map representation
python bvh2features.py -orig <path/to/motion/folder/ -dest <path/to/motion/folder/

# Split the dataset into training and validation
python split_dataset.py

# Encode all the features
python process_dataset.py

cd ..
```

As a result of running this script, the dataset is created in `--proc_data_dir`:
- the training dataset files `X_train.npy`, `Y_train.npy` and the validation dataset files `X_dev.npy`, `Y_dev.npy`are binary numpy files
- the audio inputs for testing (such as `X_test_Recoding_02.npy`) are under the `/test_inputs/` subfolder

There rest of the folders in `--proc_data_dir` (e.g. `/dev_inputs/` or `/train/`) can be ignored (they are a side effect of the preprocessing script).

## 3. Learn motion representation by AutoEncoder and encode the training and validation datasets
```python
python motion_repr_learning/ae/learn_ae_n_encode_dataset.py --layer1_width DIM
```
There are several parameters that can be modified in the `config.yaml` file or through the command line, see `config.py` for details.
The optimal dimensionality (DIM) in our experiment was 40. 

More information can be found in the folder `motion_repr_learning` 

## 4. Learn speech-driven gesture generation model

```sh
python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE DIM
# MODEL_NAME = file name for the model
# EPOCHS = how many epochs do we want to train the model (recommended - 500)
# DATA_DIR = directory with the data (should be same as before)
# N_INPUT = how many dimension does speech data have (default - 26)
# ENCODE = True (because we use AutoEncoder)
# DIM = how many dimension does encoding have (should be the same as above, recommended - 40)
```

## 5. Predict gesture

```sh
python predict.py MODEL_NAME.hdf5 INPUT_SPEECH_FILE.npy OUTPUT_GESTURE_FILE.npy
```

```sh
# Usage example
python predict.py model.hdf5 data/test_inputs/X_test_NaturalTalking_04.npy data/test_inputs/predict_04_20fps.npy
```

The predicted gestures have to be decoded with `decode.py`, which reuses the config from step 3.
```sh
python motion_repr_learning/ae/decode.py -input_file INPUT_FILE.npy -output_file OUTPUT_FILE.npy --layer1_width DIM --batch_size=8 
```

Convert the motion from exponential maps to euler angles and write into BVH file
```sh
cd data_processind
python features2bvh.py
```


## 6. Qualitative evaluation
Use animation server which is provided at the GENEA Challenge Github page to visualize your gestures from BVH format.

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
