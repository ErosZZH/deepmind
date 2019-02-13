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
        self.A = self._activation(np.dot(W, X) + B)
        self.W = W
        return self.A

    @abstractmethod
    def backward(self, X, dA): # dw = x dot dz, db = dz
        m = np.size(X, 0)
        A = self.A
        dZ = self._dZ(A, dA)
        return (self._dW(X, dZ, m), self._dB(dZ, m), np.dot(self.W.T, dZ))

    @abstractmethod
    def _dZ(self, A, dA):
        raise NotImplementedError()

    @abstractmethod
    def _dW(self, X, dZ, m):
        return np.dot(dZ, X.T) / m

    @abstractmethod
    def _dB(self, dZ, m):
        return dZ / m

'''
Logictic Regresion
'''
class Logistic(CommonActivation):

    def _activation(self, Z):
        return 1 / (1 + (np.e)**(-Z))

    def _dZ(self, A, dA):
        if self.nextLayer == 'CategoricalCrossentropy': # dA here is Yhat
            return A - dA
        return A * (1 - A) * dA

'''
tanh
'''
class Tanh(CommonActivation):

    def _activation(self, Z):
        return (np.e ** Z - np.e ** (-Z)) / (np.e ** Z + np.e ** (-Z))

    def _dZ(self, A, dA):
        return (1 - A ** 2) * dA

'''
Relu
'''
class Relu(CommonActivation):

    def _activation(self, Z):
        zero = np.zeros(Z.shape)
        return np.maximum(zero, Z)

    def _dZ(self, A, dA):
        dAdZ = np.copy(A)
        dAdZ[dAdZ >= 0] = 1
        dAdZ[dAdZ < 0] = 0
        return dAdZ * dA 

'''
leaky relu
'''
class LeakyRelu(CommonActivation):

    def _activation(self, Z):
        small_z = 0.01 * Z
        return np.maximum(small_z, Z)

    def _dZ(self, A, dA):
        dAdZ = np.copy(A)
        dAdZ[dAdZ >= 0] = 1
        dAdZ[dAdZ < 0] = 0.01
        return dAdZ * dA

'''
softmax
'''
class Softmax(CommonActivation):

    def _activation(self, Z):
        shiftZ = Z - np.max(Z)
        exps = np.e ** shiftZ
        return exps / np.sum(exps)

    def _dZ(self, A, dA):
        pass
