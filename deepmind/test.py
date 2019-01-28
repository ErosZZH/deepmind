# -*- coding: utf-8 -*-

from activation.logistic import Logistic
from activation.relu import Relu
from datasource import CSVDataSource
from nn import Neuron, Model
from loss.categorical_crossentropy import CategoricalCrossentropy
import numpy as np

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    nn1 = Neuron(activation=Relu(prevLayer=None, nextLayer='Logistic'), layer=1)
    nn2 = Neuron(activation=Logistic(prevLayer='Relu', nextLayer='CategoricalCrossentropy'), layer=2)
    model = Model([nn1, nn2], alpha=0.0001)
    model.compile(loss=CategoricalCrossentropy(outputLayer='Logistic'))
    model.fit(x_train, y_train, epoch=20)
    model.evaluate(x_test, y_test)