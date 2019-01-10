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

class CSVDataSource(DataSource):
    def __init__(self, path):
        if not path:
            raise AttributeError()
        self.path = os.path.abspath(path)

    def read(self):
        df = pd.read_csv(self.path, delim_whitespace=True, names=['fly', 'game', 'sweet', 'date'])
        self.dataFrame = df
        print('Read', self.path)

    def seperate(self):
        pass


if __name__ == '__main__':
    datasource = CSVDataSource('dataset/datingTestSet.txt')
    datasource.read()