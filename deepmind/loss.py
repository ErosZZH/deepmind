# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod

class Loss(object):

    __metaclass__ = ABCMeta

    def __init__(self, *, outputLayer=None):
        self.outputLayer = outputLayer

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self):
        raise NotImplementedError()

'''
categorical_crossentropy
'''
class CategoricalCrossentropy(Loss):

    def forward(self, Y, Yhat):
        return - (Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

    def backward(self, Y, Yhat): # dA
        if self.outputLayer == 'Logistic':
            return Y
        return (1 - Y) / (1 - Yhat) - Y / Yhat