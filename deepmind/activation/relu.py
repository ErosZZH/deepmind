# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class Relu(Activation):

    def _activation(self, Z):
        zero = np.zeros(Z.shape)
        return np.maximum(zero, Z)

    def forward(self, W, X, B):
        self.A = self._activation(np.dot(W.T, X) + B)
        self.W = W
        return self.A

    def backward(self, X, dA): # just da/dz
        m = np.size(X, 0)
        A = self.A
        dZ = self._dZ(A, dA)
        return (self._dW(X, dZ, m), self._dB(dZ, m), np.dot(self.W, dZ))

    def _dZ(self, A, dA):
        A[A >= 0] = 1
        A[A < 0] = 0
        return A * dA 

    def _dW(self, X, dZ, m):
        return np.dot(X, dZ.T) / m

    def _dB(self, dZ, m):
        return dZ / m