# -*- coding: utf-8 -*-

import numpy as np

class NN(object):

    def __init__(self, **kw):
        pass # alpha, w, b  super param


    def compile(self, *, activationInstance, costInstance):
        self.activationInstance = activationInstance
        self.costInstance = costInstance

    def propagation(self, W, X, B):
        self.A = self.activationInstance.activation_func(np.dot(W.T, X) + B)

    def backProp(self, X, Y):
        self.dWB = self.activationInstance.backProp_func(X, Y, self.A)

    def train(self, Y):
        self.L = self.costInstance.lost_func(Y, self.A)
        self.J = self.costInstance.cost_func(self.L)
