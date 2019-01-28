# -*- coding: utf-8 -*-

import numpy as np
from activation import CommonActivation

class Logistic(CommonActivation):

    def _activation(self, Z):
        return 1 / (1 + (np.e)**(-Z))

    def _dZ(self, A, dA):
        if self.nextLayer == 'CategoricalCrossentropy': # dA here is Yhat
            return A - dA
        return A * (1 - A) * dA
