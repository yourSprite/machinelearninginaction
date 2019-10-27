#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File Name: linearRegression.py
@Create Time: 2019/10/25 18:13
@Author: wangyutian
@Version: 1.0
@Python Version: Python 3.6.4
@Modify Time: 2019/10/25 18:13
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing


class linearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X: np.array, y: np.array) -> np.array:
        '''
        训练函数
        :param X: 特征数据
        :param y: 目标
        :return: 参数w
        '''
        m = len(y)
        y = y.reshape(-1, 1)
        X = np.hstack((X, np.ones((m, 1))))
        X, y = np.mat(X), np.mat(y)
        # 判断是否为满秩矩阵
        if np.linalg.det(X.T * X) == 0:
            print('XTX不是满秩矩阵')
            return
        self.w = (X.T * X).I * X.T * y

    def predict(self, X: np.array) -> np.array:
        '''
        预测函数
        :param X: 特征数据
        :return: 预测值
        '''
        m = X.shape[0]
        X = np.hstack((X, np.ones((m, 1))))
        res = np.zeros((m, 1))
        X = np.mat(X)
        for i in range(m):
            res[i] = X[i] * self.w
        return res


if __name__ == '__main__':
    f_path = './data.csv'
    df = pd.read_csv(f_path)

    array = np.array(df)
    X = array[:, :-1]
    y = array[:, -1]
    X = preprocessing.scale(X)

    X_train, y_train = X[:-10], y[:-10]
    X_test, y_test = X[-10:], y[-10:]

    lr = linearRegression()
    lr.fit(X_train, y_train)
    res = lr.predict(X_test)

    print(res)
