# -*- coding: utf-8 -*-

import numpy as np

class Neuron(object):

    def __init__(self, *, activation):
        self.activation = activation

    def prop(self, W, B, X):
        return self.activation.prop(W, X, B)

    def backProp(self, X, Y):
        return self.activation.backProp(X, Y)


class Model(object):

    def __init__(self, nn, *, alpha=0.01): # need to pass the shape of nn
        self.alpha = alpha
        self.nn = nn

    def cost(self, L):
        return (np.sum(L, axis=1) / self.m)[0]

    def compile(self, *, lost):
        self.lost = lost
    
    def train(self):
        for i in range(self.epoch):
            print('------------------ epoch' + str(i + 1) + ' -------------------')
            A = self.nn.prop(self.W, self.B, self.X_train)
            L = self.lost(self.Y_train, A)
            acc = self.predict(self.Y_train, A)
            print('cost:', self.cost(L), ', acc:', acc)
            dW, dB = self.nn.backProp(self.X_train, self.Y_train)
            self.W = self.W - self.alpha * dW
            self.B = self.B - self.alpha * dB

    def predict(self, Y, A):
        Y_pred = np.around(A)
        wrong_count = np.count_nonzero(Y_pred - Y)
        amount = np.size(Y, axis=1)
        return (amount - wrong_count) / amount

    def fit(self, X_train, Y_train, *, epoch=5):
        self.epoch = epoch
        self.m = np.size(X_train, 1)
        self.nx = np.size(X_train, 0)
        self.B = np.zeros((1, 1))
        self.W = np.random.randn(self.nx, 1) * 0.01
        self.X_train = X_train
        self.Y_train = Y_train
        self.train()

    def evaluate(self, X_test, Y_test):
        A = self.nn.prop(self.W, self.B, X_test)
        acc = self.predict(Y_test, A)
        print('Final accurate:', acc)