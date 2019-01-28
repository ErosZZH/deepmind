# -*- coding: utf-8 -*-

import numpy as np

class Neuron(object):

    def __init__(self, *, activation, layer):
        self.activation = activation
        self.layer = layer # TODO use real layer

    def forward(self, W, B, A):
        return self.activation.forward(W[self.layer - 1], A[self.layer - 1], B[self.layer - 1])

    def backward(self, A, dA):
        return self.activation.backward(A[self.layer - 1], dA)


class Model(object):

    def __init__(self, nn, *, alpha=0.01): # need to pass the shape of nn
        self.alpha = alpha
        self.nn = nn
        self.layer = [3, 1] # TODO init layer with neuron num of each layer

    def cost(self, L):
        return (np.sum(L, axis=1) / self.m)[0]

    def compile(self, *, loss):
        self.loss = loss
    
    def train(self):
        for i in range(self.epoch):
            print('------------------ epoch' + str(i + 1) + ' -------------------')
            self.A[1] = self.nn[0].forward(self.W, self.B, self.A)
            Yhat = self.nn[1].forward(self.W, self.B, self.A)
            L = self.loss.forward(self.Y_train, Yhat)
            acc = self.predict(self.Y_train, Yhat)
            print('cost:', self.cost(L), ', acc:', acc)
            dA = self.loss.backward(self.Y_train, Yhat)
            dW1, dB1, dA1 = self.nn[1].backward(self.A, dA)
            dW0, dB0, dA0 = self.nn[0].backward(self.A, dA1)
            self.W[1] = self.W[1] - self.alpha * dW1
            self.B[1] = self.B[1] - self.alpha * dB1
            self.W[0] = self.W[0] - self.alpha * dW0
            self.B[0] = self.B[0] - self.alpha * dB0

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
        B[0] = np.full((1, m), np.sum(B[0]) / m)
        B[1] = np.full((1, m), np.sum(B[1]) / m)
        A = [X_test, 0]
        A[1] = self.nn[0].forward(self.W, B, A)
        Yhat = self.nn[1].forward(self.W, B, A)
        acc = self.predict(Y_test, Yhat)
        print('Final accurate:', acc)