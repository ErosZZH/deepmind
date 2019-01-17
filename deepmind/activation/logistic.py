# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class logistic(Activation):

    def _activation(self, Z): # activation function (logistic)
        return 1 / (1 + (np.e)**(-Z))

    def prop(self, W, X, B):
        self.A = self._activation(np.dot(W.T, X) + B)
        return self.A

    def backProp(self, X, Y): # dw = x * dz, db = dz
        m = np.size(X, 0)
        A = self.A
        return (np.dot(X, (A - Y).T) / m, np.sum(A - Y) / m)

    # def da(y, yhat): # da means lost function deritive to a a = yhat
    #     return - y / yhat + (1 - y) / (1 - yhat)

    # def dz(y, yhat): # z = w * x + b
    #     return yhat - y