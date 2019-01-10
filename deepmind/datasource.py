# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod
import pandas as pd

class DataSource(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self):
        raise NotImplementedError()

    @abstractmethod
    def seperate(self):
        raise NotImplementedError()

    @abstractmethod
    def normalize(self):
        raise NotImplementedError()

class CSVDataSource(DataSource):
    def __init__(self, path, *, names):
        if not path:
            raise AttributeError()
        self.path = os.path.abspath(path)
        self.names = names

    def read(self):
        df = pd.read_csv(self.path, delim_whitespace=True, names=self.names)
        self.dataFrame = df
        print('Read', self.dataFrame)

    def seperate(self):
        pass

    def normalize(self):
        pass


if __name__ == '__main__':
    datasource = CSVDataSource('dataset/datingTestSet.txt', names=['fly', 'game', 'sweet', 'date'])
    datasource.read()