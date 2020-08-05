#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: ulysses
Date: 2020-08-05 12:14:38
LastEditTime: 2020-08-05 12:57:17
LastEditors: ulysses
Description: 
'''
# %%
import time
import logging
from pprint import pprint

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics

import numpy as np
import pandas as pd
import graphviz


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def load_preprocessing():
    # 用决策树解释泰坦尼克号假设
    #
    data = pd.read_csv('titanic')
    # data

    # %%
    # 预处理数据
    # 选择 pclass age sex 这3种数据进行划分
    titanic_X, titanic_y = data.iloc[:, [1, 4, -1]], data.iloc[:, 2]
    features = titanic_X.columns
    # features

    # %%
    # 处理缺失值
    # 将年龄的缺失值用所有人员年龄的平均值进行替换
    titanic_X = titanic_X.fillna(titanic_X.mean())
    X = titanic_X.values  # pandas -> array
    y = titanic_y.values


    # %%
    # 类别特征编码
    # 将标签值转为0..K-1的整数
    encoder = LabelEncoder()
    X[:, -1] = encoder.fit_transform(X[:, -1])  # 直接对 性别 这一列进行转换
    # 将整数特征 变 独热码
    enc_pclass = OneHotEncoder()
    new_pclass = enc_pclass.fit_transform(X[:, 0][:, np.newaxis]).toarray()
    # new_pclass

    # %%
    # 最终得到的数据
    X = np.concatenate((new_pclass, X[:, 1:]), axis=1)
    # X
    # %%
    enc_pclass.categories_[0]  # array(['1st', '2nd', '3rd'], dtype=object)

    # %%
    new_features = np.append(enc_pclass.categories_[0], np.array(features[1:]))
    return X, y, new_features


def measure_performance(X, y, clf):
    y_pred = clf.predict(X)
    print(f"Accuracy: {metrics.accuracy_score(y, y_pred):.3f}")  #  精确度得分
    print('Classification report: ')
    print(metrics.classification_report(y, y_pred))  # 具体的分类指标
    print("Confussion matrix")
    """ 从混淆矩阵的迹除以总和来计算准确度
    TN(真阴)  FP(假阳)
    FN(假阴)  TP(真阳)
    """
    print(metrics.confusion_matrix(y, y_pred))


if __name__ == "__main__":
    # %%
    # 训练决策树
    X, y, features = load_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12345)
    paramerters = {
        'criterion': ('entropy', 'gini'),
        'max_depth': (3, 5, 7),
        'min_samples_leaf': (3, 5)
    }
    print("paramerters:")
    pprint(paramerters)

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, paramerters, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Train Best score: {:.3f}".format(grid_search.best_score_))

    best_params = grid_search.best_params_
    for param_name in sorted(paramerters.keys()):
        print("\t{}: {}".format(param_name, best_params[param_name]))

    print("==========Test=========")
    measure_performance(X_test, y_test, grid_search)

    dot_data = export_graphviz(
        grid_search.best_estimator_,
        out_file=None,
        feature_names=features
         )
    graph = graphviz.Source(dot_data)
    graph.render("titanic", format='png')
