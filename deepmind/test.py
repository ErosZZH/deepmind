# -*- coding: utf-8 -*-

from activation.logistic import LogisticActivation
from datasource import CSVDataSource
from nn import NN

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    nn = NN()
    nn.compile(LogisticActivation(), '')
    print(x_train)