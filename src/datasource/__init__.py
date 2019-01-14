# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

class DataSource(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        raise NotImplementedError()

    def seperate(self, *, test_size=0.1):
        data_set = self.data_set
        label_set = self.label_set
        length = np.size(data_set, 0)
        train_len = int(length * (1 - test_size))
        x_train, x_test, y_train, y_test = data_set[:train_len].T, data_set[train_len:].T, label_set[:train_len].T, label_set[train_len:].T
        return (x_train, y_train, x_test, y_test)


class CSVDataSource(DataSource):
    def __init__(self, path):
        if not path:
            raise AttributeError()
        self.path = os.path.abspath(path)

    def read(self, *, label_map={}):
        df = pd.read_csv(self.path, delim_whitespace=True, header=None)
        data_set, label_set = df.iloc[:, 0:-1], df.iloc[:, -1:]
        label_set = label_set.values
        for key, value in label_map.items():
            label_set[label_set == key] = value
        data_set = (data_set - data_set.mean()) / (data_set.max() - data_set.min())
        self.data_set = data_set.values
        label_set = label_set.astype(np.float)
        self.label_set = label_set


if __name__ == '__main__':
    datasource = CSVDataSource('dataset/datingTestSet.txt')
    datasource.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    datasource.seperate()