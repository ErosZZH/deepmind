# -*- coding: utf-8 -*-

import numpy as np

class NN(object):

    # alpha = 0.01
    # W = np.random.randn(3, 1) * 0.01
    # B = np.zeros((1, 1))

    # def __init__(self, **kw):
    #     if 'alpha' in kw:
    #         self.alpha = kw.get('alpha')
    #     if 'W' in kw:
    #         self.W = kw.get('W')
    #     if 'alpha' in kw:
    #         self.B = kw.get('B')


    def compile(self, activationInstance, costInstance):
        self.activationInstance = activationInstance
        self.costInstance = costInstance

    def propagation(self, W, X, B):
        self.A = self.activationInstance.activation_func(np.dot(W.T, X) + B)

    def backProp(self, X, Y):
        self.dWB = self.activationInstance.backProp_func(X, Y, self.A)

    def train(self, X_train, Y_train, X_test, Y_test):
        self.L = self.costInstance.lost_func(Y_train, self.A)
        self.J = self.costInstance.cost_func(self.L)

    def predict(self):
        pass
