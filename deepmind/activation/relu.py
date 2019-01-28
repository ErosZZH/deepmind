# -*- coding: utf-8 -*-

import numpy as np
from activation import CommonActivation

class Relu(CommonActivation):

    def _activation(self, Z):
        zero = np.zeros(Z.shape)
        return np.maximum(zero, Z)

    def _dZ(self, A, dA):
        dAdZ = np.copy(A)
        dAdZ[dAdZ >= 0] = 1
        dAdZ[dAdZ < 0] = 0
        return dAdZ * dA 
