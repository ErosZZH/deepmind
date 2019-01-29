# -*- coding: utf-8 -*-

import numpy as np
from activation import CommonActivation

class LeakyRelu(CommonActivation):

    def _activation(self, Z):
        small_z = 0.01 * Z
        return np.maximum(small_z, Z)

    def _dZ(self, A, dA):
        dAdZ = np.copy(A)
        dAdZ[dAdZ >= 0] = 1
        dAdZ[dAdZ < 0] = 0.01
        return dAdZ * dA