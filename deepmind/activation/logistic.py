# -*- coding: utf-8 -*-

import numpy as np
from activation import Activation

class LogisticActivation(Activation):

    def activation_func(self, Z):
        return 1 / (1 + (np.e)**(-Z))

    def backProp_func(self, X, A, Y): # dw = x * dz, db = dz
        m = np.size(X, 0)
        return (np.dot(X, (A - Y).T) / m, np.sum(A - Y) / m)

    # def da(y, yhat): # da means lost function deritive to a a = yhat
    #     return - y / yhat + (1 - y) / (1 - yhat)

    # def dz(y, yhat): # z = w * x + b
    #     return yhat - y