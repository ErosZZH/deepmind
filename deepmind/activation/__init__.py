# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class Activation(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def activation_func(self):
        raise NotImplementedError()

    @abstractmethod
    def backProp_func(self):
        raise NotImplementedError()