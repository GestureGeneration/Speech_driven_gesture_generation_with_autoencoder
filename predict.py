"""
This script generates gestures output based on the speech input.
The gestures will be written in the text file:
3d coordinates together with the velocities.
"""

import sys
from keras.models import load_model
import numpy as np
from scipy.signal import savgol_filter


def smoothing(motion):

    smoothed = [savgol_filter(motion[:,i], 9, 3) for i in range(motion.shape[1])]

    new_motion = np.array(smoothed).transpose()

    return new_motion


def predict(model_name, input_file, output_file):
    """ Predict human gesture based on the speech

    Args:
        model_name:  name of the Keras model to be used
        input_file:  file name of the audio input
        output_file: file name for the gesture output

    Returns:

    """
    model = load_model(model_name)
    audio = np.load(input_file)

    predicted = np.array(model.predict(audio))

    smoothed = smoothing(predicted)

    print("Encoding shape is: ", predicted.shape)
    np.savetxt(output_file, smoothed)


if __name__ == "__main__":

    # Check if script get enough parameters
    if len(sys.argv) < 4:
        raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] +
                         ' MODEL_NAME INPUT_FILE OUTPUT_FILE')

    model_file = sys.argv[1] 
    if not model_file.endswith(".hdf5"):
        model_file += ".hdf5"
    
    predict(model_file, sys.argv[2], sys.argv[3])
    print("Done")
