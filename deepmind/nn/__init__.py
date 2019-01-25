# -*- coding: utf-8 -*-

import numpy as np

class Neuron(object):

    def __init__(self, *, activation):
        self.activation = activation
        self.layer = 1 # TODO use real layer

    def forward(self, W, B, A):
        return self.activation.forward(W[self.layer - 1], A[self.layer - 1], B[self.layer - 1])

    def backward(self, A, dA):
        return self.activation.backward(A[self.layer - 1], dA)


class Model(object):

    def __init__(self, nn, *, alpha=0.01): # need to pass the shape of nn
        self.alpha = alpha
        self.nn = nn
        self.layer = [1] # TODO init layer with neuron num of each layer

    def cost(self, L):
        return (np.sum(L, axis=1) / self.m)[0]

    def compile(self, *, loss):
        self.loss = loss
    
    def train(self):
        for i in range(self.epoch):
            print('------------------ epoch' + str(i + 1) + ' -------------------')
            Yhat = self.nn.forward(self.W, self.B, self.A)
            L = self.loss.forward(self.Y_train, Yhat)
            acc = self.predict(self.Y_train, Yhat)
            print('cost:', self.cost(L), ', acc:', acc)
            dA = self.loss.backward(self.Y_train, Yhat)
            dW, dB = self.nn.backward(self.A, dA)
            self.W[0] = self.W[0] - self.alpha * dW # TODO not hardcode layer
            self.B[0] = self.B[0] - self.alpha * dB

    def predict(self, Y, A):
        Y_pred = np.around(A)
        wrong_count = np.count_nonzero(Y_pred - Y)
        amount = np.size(Y, axis=1)
        return (amount - wrong_count) / amount

    def fit(self, X_train, Y_train, *, epoch=5):
        self.epoch = epoch
        n = [0] + self.layer
        n[0] = np.size(X_train, 0)
        self.n = n
        self.m = np.size(X_train, 1)
        self.W = [np.random.randn(self.n[i], self.n[i + 1]) * 0.01 for i in range(len(self.layer))]
        self.B = [np.zeros((self.n[i + 1], self.m)) for i in range(len(self.layer))]
        A = [0] * len(self.layer)
        A[0] = X_train
        self.A = A # shape(n, m)
        self.Y_train = Y_train
        self.train()

    def evaluate(self, X_test, Y_test):
        m = np.size(X_test, 1)
        B = self.B
        # B[0] = B[0][:, :m]
        b = np.sum(B[0]) / m
        B[0] = np.full((1, m), b)
        A = self.nn.forward(self.W, B, [X_test])
        acc = self.predict(Y_test, A)
        print('Final accurate:', acc)