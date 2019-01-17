# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Softmax(Activation):

    def _activation(self, Z):
        return np.e ** Z / np.sum(np.e ** Z)

    def prop(self, W, X, B):
        pass

    def backProp(self, X, Y):
        pass