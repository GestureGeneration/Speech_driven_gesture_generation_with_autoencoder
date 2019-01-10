# Motion Representation Learning

This is a repository for learning a compact and informative representation of the human motion sequence.

## The main idea
The aim is to learn a better representation of the motion frames using an auto-encoding neural networks, such as Denoising Autoencoder or Variational Autoencoder.

Encoding (MotionE) is a mapping from a sequence of the 3D positions of the human to a lower-dimensional representation, which will contain enough information to reconstruct original motion sequence, but will have less redundancy and hence will be better for the speech-to-motion mapping.
Decoding (MotionD) is a mapping from the encoded vector back to the 3D motion sequence.

Once a motion encoder MotionE and a motion decoder MotionD are learned, we train a novel encoder network SpeechE to map from speech to a corresponding low-dimensional motion representation (code for this mapping is given in the main folder of this repository).

At test time, the speech encoder and the motion decoder networks are combined: SpeechE predicts motion representations based on a given speech signal and MotionD then decodes these representations to produce motion sequences.

## Requirements
- Python 3
- Tensorflow >= 1.0
- Additional Python libraries:
  - numpy
  - matplotlib (if you want to visualize the results)
  - btk (if you want to preprocess motion test sequences by yourself)

## Data preparation

1. Follow the instruction on data preparation at the root folder of this repository.
2. Indicate the directory for the data at utils/flags.py as "data_dir" value.
3. Indicate the directory to the checkpoints (will be used to store the model) at utils/flags as "chkpt_dir" value.

## Run
To run the default example execute the following command. 

```bash
# Learn dataset encoding
python learn_dataset_encoding.py DATA_DIR motion -chkpt_dir=CHKPT_DIR -layer1_width=DIM

#Encode dataset
python encode_dataset.py DATA_DIR motion -chkpt_dir=CHKPT_DIR -restore=True -pretrain=False -layer1_width=DIM
```

Where DATA_DIR is a directory where the data is stored, CHKPT_DIR is a directory to store the model checkpoints and DIM is dimensionality of the representation.


## Customizing
You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc. in the file flags.py.

## Contact

If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
