# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

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