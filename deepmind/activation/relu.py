# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Relu(Activation):

    def _activation(self, Z):
        zero = np.zeros(Z.shape)
        return np.maximum(zero, Z)

    def prop(self, W, X, B):
        pass

    def backProp(self, X, Y):
        pass