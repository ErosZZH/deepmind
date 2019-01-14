# -*- coding: utf-8 -*-

import numpy as np
from datasource import CSVDataSource

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

def predict(W, B, X):
    return np.around(propagation(W, X, B))

def accurate(Y_pred, y_test):
    wrong_count = np.count_nonzero(Y_pred - y_test)
    return (100 - wrong_count) / 100
    

if __name__ == '__main__':
    ds = CSVDataSource('dataset/datingTestSet.txt')
    ds.read(label_map={'didntLike': 0, 'largeDoses': 1, 'smallDoses': 1})
    x_train, y_train, x_test, y_test = ds.seperate()
    
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
        w = w - alpha * dw
        b = b - alpha * db