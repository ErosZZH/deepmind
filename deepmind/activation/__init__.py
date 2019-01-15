# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class Activation(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def prop(self):
        raise NotImplementedError()

    @abstractmethod
    def backProp(self):
        raise NotImplementedError()