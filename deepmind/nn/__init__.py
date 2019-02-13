# -*- coding: utf-8 -*-

import numpy as np

class Neuron(object):

    def __init__(self, *, activation, layer):
        self.activation = activation
        self.layer = layer

    def forward(self, W, B, A):
        return self.activation.forward(W[self.layer - 1], A[self.layer - 1], B[self.layer - 1])

    def backward(self, A, dA):
        return self.activation.backward(A[self.layer - 1], dA)


class Model(object):

    def __init__(self, layers, *, alpha=0.001): # need to pass the shape of nn
        self.alpha = alpha
        self.layers = layers

    def cost(self, L):
        return (np.sum(L, axis=1) / self.m)[0]

    def compile(self, *, loss):
        nodes = []
        layers = []
        for k, v in enumerate(self.layers):
            nodes.append(v['node'])
            prevLayer = None if k == 0 else self.layers[k - 1]['activation'].__class__.__name__
            nextLayer = loss.__class__.__name__ if k == len(self.layers) - 1 else self.layers[k + 1]['activation'].__class__.__name__
            layer = k + 1
            layers.append(Neuron(activation=v['activation'](prevLayer=prevLayer, nextLayer=nextLayer), layer=layer))
        self.layer = nodes
        self.nn = layers
        self.loss = loss(outputLayer=self.layers[-1]['activation'].__class__.__name__)
    
    def train(self):
        for i in range(self.epoch):
            print('------------------ epoch' + str(i + 1) + ' -------------------')
            Yhat = None
            for index, layer in enumerate(self.nn):
                if index < len(self.nn) - 1:
                    self.A[index + 1] = layer.forward(self.W, self.B, self.A)
                else:
                    Yhat = layer.forward(self.W, self.B, self.A)
            L = self.loss.forward(self.Y_train, Yhat)
            acc = self.predict(self.Y_train, Yhat)
            print('cost:', self.cost(L), ', accurate:', acc)
            dA = self.loss.backward(self.Y_train, Yhat)
            length = len(self.nn) - 1
            for index, layer in enumerate(reversed(self.nn)):
                dW, dB, dA = layer.backward(self.A, dA)
                self.W[length - index] = self.W[length - index] - self.alpha * dW
                self.B[length - index] = self.B[length - index] - self.alpha * dB

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
        self.W = [np.random.randn(self.n[i + 1], self.n[i]) * 0.01 for i in range(len(self.layer))]
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
        for k, v in enumerate(B):
            B[k] = np.full((1, m), np.sum(v) / m)
        A = [X_test, 0]
        Yhat = None
        for index, layer in enumerate(self.nn):
            if index < len(self.nn) - 1:
                A[index + 1] = layer.forward(self.W, B, A)
            else:
                Yhat = layer.forward(self.W, B, A)
        acc = self.predict(Y_test, Yhat)
        print('Final accurate:', acc)