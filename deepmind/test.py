# -*- coding: utf-8 -*-

from layer import Logistic, Relu
from datasource import CSVDataSource
from nn import Neuron, Model
from loss import CategoricalCrossentropy
import numpy as np

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    model = Model([
        {'node': 3, 'activation': Relu},
        {'node': 1, 'activation': Logistic}
    ], alpha=0.0001)
    model.compile(loss=CategoricalCrossentropy)
    model.fit(x_train, y_train, epoch=10)
    model.evaluate(x_test, y_test)