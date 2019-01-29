# -*- coding: utf-8 -*-

import numpy as np
from activation import CommonActivation

class Tanh(CommonActivation):

    def _activation(self, Z):
        return (np.e ** Z - np.e ** (-Z)) / (np.e ** Z + np.e ** (-Z))

    def _dZ(self, A, dA):
        return (1 - A ** 2) * dA