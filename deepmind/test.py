# -*- coding: utf-8 -*-

from activation.logistic import LogisticActivation
from datasource import CSVDataSource
from nn import Neuron
from lost import common_lost
import numpy as np

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    nn = Neuron(activation=LogisticActivation())
    nn.compile(lost=common_lost)
    b = np.zeros((1, 1))
    w = np.random.randn(3, 1) * 0.01
    nn.process(w, b, x_train, y_train)