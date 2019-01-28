# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Logistic(Activation):

    def _activation(self, Z): # activation function (logistic)
        return 1 / (1 + (np.e)**(-Z))

    def forward(self, W, X, B):
        self.A = self._activation(np.dot(W.T, X) + B)
        self.W = W
        return self.A

    def backward(self, X, dA): # dw = x * dz, db = dz
        m = np.size(X, 0)
        A = self.A
        dZ = self._dZ(A, dA)
        return (self._dW(X, dZ, m), self._dB(dZ, m), np.dot(self.W, dZ))

    def _dZ(self, A, dA):
        if self.nextLayer == 'CategoricalCrossentropy': # dA here is Y
            return A - dA
        return A * (1 - A) * dA

    def _dW(self, X, dZ, m):
        return np.dot(X, dZ.T) / m

    def _dB(self, dZ, m):
        return dZ / m
