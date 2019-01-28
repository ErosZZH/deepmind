# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

class Activation(object):
    __metaclass__ = ABCMeta

    def __init__(self, *, prevLayer=None, nextLayer=None):
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self):
        raise NotImplementedError()

class CommonActivation(Activation):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _activation(self, Z):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, W, X, B):
        self.A = self._activation(np.dot(W.T, X) + B)
        self.W = W
        return self.A

    @abstractmethod
    def backward(self, X, dA): # dw = x dot dz, db = dz
        m = np.size(X, 0)
        A = self.A
        dZ = self._dZ(A, dA)
        return (self._dW(X, dZ, m), self._dB(dZ, m), np.dot(self.W, dZ))

    @abstractmethod
    def _dZ(self, A, dA):
        raise NotImplementedError()

    @abstractmethod
    def _dW(self, X, dZ, m):
        return np.dot(X, dZ.T) / m

    @abstractmethod
    def _dB(self, dZ, m):
        return dZ / m