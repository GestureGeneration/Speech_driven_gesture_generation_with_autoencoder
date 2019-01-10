# -*- coding: utf-8 -*-
#
# OneEuroFilter.py -
#
# Author: Nicolas Roussel (nicolas.roussel@inria.fr)

import math
import numpy as np


# ----------------------------------------------------------------------------

class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y

# ----------------------------------------------------------------------------


class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?  # noqa
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))

# ----------------------------------------------------------------------------


def apply_one_euro(pos_array):
    """Apply one euro filter to a gesture

       Original implementation can be downloaded from
       http://cristal.univ-lille.fr/~casiez/1euro/

      Args:
          pos_array:    body keypoint positions to filter
      Returns:
          np.ndarray:   filtered positions
    """

    pos_along_timestep = pos_array.transpose()

    config = {
        'freq': 20,        # Hz
        'mincutoff': 0.1,  # Minimum cutoff frequency
        'beta': 0.08,      # Cutoff slope
        'dcutoff': 1.0     # Cutoff frequency for derivate
    }

    oef = OneEuroFilter(**config)

    filtered_pos = []
    for i, joint in enumerate(pos_along_timestep):
        joint_pos = []
        for timestep, pos in enumerate(joint):
            if timestep > 0:
                timestep = timestep * 1.0 / config["freq"]
            filt_num = oef(pos, timestep)
            joint_pos.append(filt_num)
        filtered_pos.append(joint_pos)

    filtered_pos_array = np.array(filtered_pos)

    return filtered_pos_array.transpose()
