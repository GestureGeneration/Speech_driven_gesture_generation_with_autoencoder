""" This code was taken from Daniel Holden
 at his web-site http://theorangeduck.com  """

import operator

import numpy as np
import numpy.core.umath_tests as ut

from Quaternions import Quaternions

class Animation:
    """
    Animation is a numpy-like wrapper for animation data

    Animation data consists of several arrays consisting
    of F frames and J joints.

    The animation is specified by

        rotations : (F, J) Quaternions | Joint Rotations
        positions : (F, J, 3) ndarray  | Joint Positions

    The base pose is specified by

        orients   : (J) Quaternions    | Joint Orientations
        offsets   : (J, 3) ndarray     | Joint Offsets

    And the skeletal structure is specified by

        parents   : (J) ndarray        | Joint Parents
    """

    def __init__(self, rotations, positions, orients, offsets, parents):
        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = parents

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients[k[1:]],
                self.offsets[k[1:]],
                self.parents[k[1:]])
        else:
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients,
                self.offsets,
                self.parents)

    def __setitem__(self, k, v):
        if isinstance(k, tuple):
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k[1:], v.orients)
            self.offsets.__setitem__(k[1:], v.offsets)
            self.parents.__setitem__(k[1:], v.parents)
        else:
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k, v.orients)
            self.offsets.__setitem__(k, v.offsets)
            self.parents.__setitem__(k, v.parents)

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])