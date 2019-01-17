# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class LeakyRelu(Activation):

    def _activation(self, Z):
        small_z = 0.01 * Z
        return np.maximum(small_z, Z)

    def prop(self, W, X, B):
        pass

    def backProp(self, X, Y):
        pass