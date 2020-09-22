"""
This script contains supporting function for the data processing.
It is used in several other scripts:
for calculation of speech features, aligning sequences and generating bvh files
"""

import ctypes

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Acoustic signal processing
import scipy.io.wavfile as wav
from pydub import AudioSegment
from python_speech_features import mfcc
import scipy

from alt_prosody import compute_prosody

MFCC_INPUTS=26 # How many features we will store for each MFCC vector
WINDOW_LENGTH = 0.1


def create_bvh(filename, prediction, frame_time):
    """
    Create BVH File
    Args:
        filename:    file, in which motion in bvh format should be written
        prediction:  motion sequences, to be written into file
        frame_time:  frame rate of the motion
    Returns:
        nothing, writes motion to the file
    """
    with open('hformat.txt', 'r') as ftemp:
        hformat = ftemp.readlines()

    with open(filename, 'w') as fo:
        prediction = np.squeeze(prediction)
        print("output vector shape: " + str(prediction.shape))
        offset = [0, 60, 0]
        offset_line = "\tOFFSET " + " ".join("{:.6f}".format(x) for x in offset) + '\n'
        fo.write("HIERARCHY\n")
        fo.write("ROOT Hips\n")
        fo.write("{\n")
        fo.write(offset_line)
        fo.writelines(hformat)
        fo.write("MOTION\n")
        fo.write("Frames: " + str(len(prediction)) + '\n')
        fo.write("Frame Time: " + frame_time + "\n")
        for row in prediction:
            row[0:3] = 0
            legs = np.zeros(24)
            row = np.concatenate((row, legs))
            label_line = " ".join("{:.6f}".format(x) for x in row) + " "
            fo.write(label_line + '\n')
        print("bvh generated")

def shorten(arr1, arr2):
    min_len = min(len(arr1), len(arr2))

    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    return arr1, arr2

def shorten3(arr1, arr2, arr3):
    min_len = min(len(arr1), len(arr2), len(arr3))

    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    arr3 = arr3[:min_len]

    return arr1, arr2, arr3


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def calculate_mfcc(audio_filename):
    """
    Calculate MFCC features for the audio in a given file
    Args:
        audio_filename: file name of the audio

    Returns:
        feature_vectors: MFCC feature vector for the given audio file
    """
    fs, audio = wav.read(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    # Calculate MFCC feature with the window frame it was designed for
    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01, samplerate=fs, numcep=MFCC_INPUTS)

    input_vectors = [average(input_vectors[:, i], 5) for i in range(MFCC_INPUTS)]

    feature_vectors = np.transpose(input_vectors)

    return feature_vectors

def get_energy_level(sound, win_len):
    """ Calculate energy signal of an audio object
    Args:
        sound:   AudioSegment object with the audio signal
        win_len: length of the window for the energy calculations
    Returns:
        energy:  the energy of the signal
    """

    loudness = list([])

    length = len(sound) - win_len

    # Split signal into short chunks and get energy of each of them
    for i in range(0, length, win_len):
        current_segment = sound[i:i + win_len]
        loudness.append(current_segment.rms)

    # Append the last segment, which was not considered
    loudness.append(0)

    energy = np.array(loudness)

    return energy


def derivative(x, f):
    """ Calculate numerical derivative (by FDM) of a 1d array
    Args:
        x: input space x
        f: Function of x
    Returns:
        der:  numerical derivative of f wrt x
    """

    x = 1000 * x  # from seconds to milliseconds

    # Normalization:
    dx = (x[1] - x[0])

    cf = np.convolve(f, [1, -1]) / dx

    # Remove unstable values
    der = cf[:-1].copy()
    der[0] = 0

    return der


def calculate_pitch(audio_filename):
    """ Calculate F0 contour of a given speech file
    Args:
        audio_filename:  address of a speech file
    Returns:
        F0 contour in a log scale and flag indicating weather F0 existed
    """

    fs, audio = wav.read(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio =( (audio[:, 0] + audio[:, 1]) / 2 ).astype(ctypes.c_int16)

    plot = False

    WINDOW_LENGTH = 5
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(audio, fs=fs, minf0=80, maxf0=250)

    # Remove unstable values
    f0 = f0[1:-1].copy()

    # Get an indication if F0 exists
    f0[f0 == -1] = np.nan
    F0_exists = 1 - np.isnan(f0).astype(int)

    # Interpolate pitch values
    ts = pd.Series(f0, index=range(f0.shape[0]))
    ts = ts.interpolate(method='linear', downcast='infer')\

    f0 = ts.values

    nans = np.isnan(f0).tolist()

    # Extrapolate at the beginning
    if False in nans:
        first_value = nans.index(False)
        first_nans = nans[0:first_value]
        for time in range(len(first_nans)):
            f0[time] = f0[first_value]

        # Extrapolate at the end
        if True in nans[first_value:]:
            last_value = nans[first_value:].index(True)
            last_nans = nans[last_value:]
            for time in range(len(last_nans)):
                f0[-time] = f0[last_value]

    if plot:

        plt.plot(f0, linewidth=3, label="F0")
        plt.title("F0 results")
        plt.show()

    # Convert to the log scale
    F0_contour = np.log2(f0+1)
    return F0_contour, F0_exists


def extract_prosodic_features(audio_filename):
    """
    Extract all 5 prosodic features
    Args:
        audio_filename:   file name for the audio to be used
    Returns:
        pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
    """

    WINDOW_LENGTH = 5

    # Read audio from file
    sound = AudioSegment.from_file(audio_filename, format="wav")

    # Alternative prosodic features
    pitch, energy = compute_prosody(audio_filename, WINDOW_LENGTH / 1000)

    duration = len(sound) / 1000
    t = np.arange(0, duration, WINDOW_LENGTH / 1000)

    energy_der = derivative(t, energy)
    pitch_der = derivative(t, pitch)

    # Average everything in order to match the frequency
    energy = average(energy, 10)
    energy_der = average(energy_der, 10)
    pitch = average(pitch, 10)
    pitch_der = average(pitch_der, 10)

    # Cut them to the same size
    min_size = min(len(energy), len(energy_der), len(pitch_der), len(pitch_der))
    energy = energy[:min_size]
    energy_der = energy_der[:min_size]
    pitch = pitch[:min_size]
    pitch_der = pitch_der[:min_size]

    # Stack them all together
    pros_feature = np.stack((energy, energy_der, pitch, pitch_der))#, pitch_ind))

    # And reshape
    pros_feature = np.transpose(pros_feature)

    return pros_feature


def calculate_spectrogram(audio_filename):
    """ Calculate spectrogram for the audio file
    Args:
        audio_filename: audio file name
    Returns:
        log spectrogram values
    """

    DIM = int(64)

    audio, sample_rate = librosa.load(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    spectr = librosa.feature.melspectrogram(audio, sr=sample_rate, #window = scipy.signal.hanning,
                                            hop_length = int(WINDOW_LENGTH* sample_rate / 2),
                                            fmax=7500, fmin=100, n_mels=DIM)

    # Shift into the log scale
    eps = 1e-10
    log_spectr = np.log(abs(spectr)+eps)

    return np.transpose(log_spectr)
