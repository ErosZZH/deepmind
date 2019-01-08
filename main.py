# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

alpha = 3
epoch = 20

def activation(z): # activation function (logistic)
    return 1 / (1 + (np.e)**(-z))

def lost(y, yhat):
    return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

def cost(L, m):
    return np.sum(L, axis=1) / m

def propagation(w, x, b): # return y hat
    return activation(np.dot(w.T, x) + b)

def da(y, yhat): # da means lost function deritive to a a = yhat
    return - y / yhat + (1 - y) / (1 - yhat)

def dz(y, yhat): # z = w * x + b
    return yhat - y

def backProp(x, y, yhat, m): # dw = x * dz, db = dz
    return (np.dot(x, (yhat - y).T) / m, np.sum(yhat - y) / m)

def importData(path):
    df = pd.read_csv(path, delim_whitespace=True, names=['fly', 'game', 'sweet', 'date'])
    data_set, label_set = df.iloc[:, 0:-1], df.iloc[:, -1:]
    label_set['date'] = np.where(label_set['date'] == 'didntLike', 0, 1)
    data_set = (data_set - data_set.mean()) / (data_set.max() - data_set.min())
    return data_set, label_set

def predict(W, B, X):
    return np.around(propagation(W, X, B))

def accurate(Y_pred, y_test):
    wrong_count = np.count_nonzero(Y_pred - y_test)
    return (100 - wrong_count) / 100
    

if __name__ == '__main__':
    data_set, label_set = importData('/Users/user/Desktop/workspace/my-project/deepmind/dataset/datingTestSet.txt')
    x_train, y_train, x_test, y_test = data_set.head(900).values.T, label_set.head(900).values.T, data_set.tail(100).values.T, label_set.tail(100).values.T
    
    b = np.zeros((1, 1))
    w = np.random.randn(3, 1) * 0.01
    assert(w.shape == (3, 1))
    assert(b.shape == (1, 1))

    for i in range(epoch):
        print('------------------ epoch' + str(i + 1) + ' -------------------')
        # print('w: %f, b: %f' % (w, b))
        yhat = propagation(w, x_train, b)
        L = lost(y_train, yhat)
        print('cost function', cost(L, 900)[0])
        Y_pred = predict(w, b, x_test)
        acc = accurate(Y_pred, y_test)
        print('Accurate:', acc)
        dw, db = backProp(x_train, y_train, yhat, 900)
        # print('dw: %f, db: %f' % (dw, db))
        w -= alpha * dw
        b -= alpha * db