# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class Activation(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self):
        raise NotImplementedError()