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
