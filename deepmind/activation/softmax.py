# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Softmax(Activation):

    def _activation(self, Z):
        shiftZ = Z - np.max(Z)
        exps = np.e ** shiftZ
        return exps / np.sum(exps)

    def _dZ(self, A, dA):
        pass