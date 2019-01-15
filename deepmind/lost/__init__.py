# -*- coding: utf-8 -*-

import numpy as np

def common_lost(Y, A):
    return - (Y * np.log(A) + (1 - Y) * np.log(1 - A))