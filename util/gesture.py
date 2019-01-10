from __future__ import absolute_import
import numpy as np

def gesturefile_to_label_vector(gesture_filename):
    f = open(gesture_filename, 'r')

    # read all lines of the bvh file
    org = f.readlines()
    offset = org[3]
    offset = offset.split()
    del offset[0]
    # delete the HIERARCHY part
    del org[0:311]

    for idx, line in enumerate(org):
        org[idx] = [float(x) for x in offset]
        org[idx].extend([float(x) for x in line.split()])

    train_labels = np.array(org)

    # only keep every second label (BiRNN stride = 2)
    #train_labels = train_labels[::2]

    #train_labels = np.delete(train_labels, np.s_[0:150], 1)

    #print(train_labels.shape)

    f.close()

    return train_labels

