# -*- coding: utf-8 -*-

import numpy as np
from loss import Loss

class CategoricalCrossentropy(Loss):

    def forward(self, Y, Yhat):
        return - (Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))

    def backward(self, Y, Yhat): # dA
        if self.outputLayer == 'Logistic':
            return Y
        return (1 - Y) / (1 - Yhat) - Y / Yhat