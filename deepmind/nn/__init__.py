# -*- coding: utf-8 -*-

import numpy as np

class Neuron(object):

    def __init__(self, *, activation):
        self.activation = activation

    def compile(self, *, lost):
        self.lost = lost

    def process(self, W, B, X, Y):
        self.A = self.activation.prop(W, X, B)
        self.L = self.lost(Y, self.A)
        self.dWB = self.activation.backProp(X, Y)
