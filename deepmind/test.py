# -*- coding: utf-8 -*-

from activation.logistic import logistic
from datasource import CSVDataSource
from nn import Neuron, Model
from lost import common_lost
import numpy as np

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    nn = Neuron(activation=logistic())
    model = Model(nn)
    model.compile(lost=common_lost)
    model.fit(x_train, y_train, epoch=20)
    model.evaluate(x_test, y_test)