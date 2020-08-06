#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: ulysses
Date: 2020-08-06 12:02:12
LastEditTime: 2020-08-06 12:55:17
LastEditors: ulysses
Description: LogisticRegression 实现 fashion数据集分类
'''
# %%
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# 训练集图片的路径
data_path = '/mnt/data1/workspace/AI/mocc_tensorflow/class4/MNIST_FC'
train_path = os.path.join(
    data_path, 'mnist_image_label/mnist_train_jpg_60000/')

# 训练集 图片名-数字(标签y)
train_txt = os.path.join(
    data_path, 'mnist_image_label/mnist_train_jpg_60000.txt')
x_train_savepath = os.path.join(
    data_path, 'mnist_image_label/mnist_x_train.npy')
y_train_savepath = os.path.join(
    data_path, 'mnist_image_label/mnist_y_train.npy')

test_path = os.path.join(data_path, 'mnist_image_label/mnist_test_jpg_10000/')
test_txt = os.path.join(data_path, 'mnist_image_label/mnist_test_jpg_10000.txt')
x_test_savepath = os.path.join(data_path, 'mnist_image_label/mnist_x_test.npy')
y_test_savepath = os.path.join(data_path, 'mnist_image_label/mnist_y_test.npy')


def get_data(x_save_path, y_save_path, x_pic_path, y_text_path):
    if os.path.exists(x_save_path) and os.path.exists(y_save_path):
        X = np.load(x_save_path)
        y = np.load(y_save_path)
        return X, y
    elif os.path.exists(x_pic_path) and os.path.exists(y_text_path):
        with open(y_text_path, 'r') as f:
            contents = f.readlines()
        X, y = [], []
        print()
        for content in contents:
            pic_name, label = content.split()
            pic_path = os.path.join(x_pic_path, pic_name)
            img = Image.open(pic_path)
            img = np.array(img.convert('L'))  # 灰度形式 1通道
            # img = img / 255.
            X.append(img)
            y.append(label)
        X = np.array(X)
        y = np.array(y).astype('int')
        # 从图片中读取后保存在 x_save_path路径
        np.save(x_save_path, X)
        np.save(y_save_path, y)
        return X, y
    else:
        return None


X_train, y_train = get_data(x_train_savepath, y_train_savepath,
                            train_path, train_txt)
print(X_train.shape, y_train.shape)

X_test, y_test = get_data(x_test_savepath, y_test_savepath,
                          test_path, test_txt)
print(X_test.shape, y_test.shape)

# %%
t0 = time.time()
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
# 随机排列
random_state = check_random_state(0)
permutation = random_state.permutation(X_train.shape[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

# standardsaclar
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
# saga 求解器 [7] 是 sag 的一类变体，它支持非平滑（non-smooth）的 
# L1 正则选项 penalty="l1" 。因此对于稀疏多项式 logistic 回归 ，
# 往往选用该求解器
clf = LogisticRegression(
    C=0.001,  # 正则化系数的倒数
    penalty='l1',  # 
    solver='saga',
    tol=0.1)
# %%
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)
# 权重向量非常稀疏
# Sparsity with L1 penalty: 80.71%
# Test score with L1 penalty: 0.8555

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()

# %%
