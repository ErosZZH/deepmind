# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Relu(Activation):

    def _activation(self, Z):
        zero = np.zeros(Z.shape)
        return np.maximum(zero, Z)

    def prop(self, W, X, B):
        self.A = self._activation(np.dot(W.T, X) + B)
        return self.A

    def backProp(self, X, Y): # just da/dz
        A = self.A
        A[A >= 0] = 1
        A[A < 0] = 0
        return A