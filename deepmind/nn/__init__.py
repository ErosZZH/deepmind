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


class Model(object):

    def __init__(self, nn, *, alpha=0.01): # need to pass the shape of nn
        self.alpha = alpha
        self.nn = nn

    def cost(self, L):
        return (np.sum(L, axis=1) / self.m)[0]

    def compile(self, *, lost):
        self.nn.compile(lost=lost)
    
    def train(self):
        for i in range(self.epoch):
            print('------------------ epoch' + str(i + 1) + ' -------------------')
            self.nn.process(self.W, self.B, self.X_train, self.Y_train)
            print('cost function', self.cost(self.nn.L))
            dW, dB = self.nn.dWB
            self.W -= self.alpha * dW
            self.B -= self.alpha * dB
            print(self.W, self.B)

    def fit(self, X_train, Y_train, *, epoch=5):
        self.epoch = epoch
        self.m = np.size(X_train, 1)
        self.nx = np.size(X_train, 0)
        self.B = np.zeros((1, 1))
        self.W = np.random.randn(self.nx, 1) * 0.01
        self.X_train = X_train
        self.Y_train = Y_train
        self.train()

    def evaluate(self):
        pass