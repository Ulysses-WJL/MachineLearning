#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-21 14:54:17
@Description: 对数回归
@LastEditTime: 2019-11-22 14:54:37
'''
import os
import numpy as np
import matplotlib.pyplot as plt


TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'TestSet.txt')
HORSE_TRAIN_PATH =  os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'HorseColicTraining.txt')
HORSE_TEST_PATH =  os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'HorseColicTest.txt')


# 获取解析数据
def load_data_set(file_name):
    array1, array2, array3 = np.loadtxt(file_name, delimiter='\t', unpack=True)
    # print(array1.shape)
    # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
    data_mat = np.stack((np.ones(array1.shape), array1, array2), axis=1)  # [[1, -0.017612, 14.053064], [...]]
    return data_mat, array3


# 构造预测函数
def sigmoid(inx):
    # sigmod阶跃函数
    '''单个数字输入时有效
    if inx >= 0:  # 防止 inx为负数时出现极大数字
        return 1 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))
    '''
    return 1 / (1 + np.exp(-inx))
    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    # return 2 * 1.0 / (1 + np.exp(-2*inx)) - 1


def batch_grad_descent(data_mat_in, class_labels):
    """
    批量梯度下降法 Batch Gradient Descent, BGD
    Args:
        data_mat ([type]): 二维矩阵,每行代表一个训练样本, 每列代表不同特征
        class_labels ([type]): 代表样本分类的矩阵
    Returns:
        [type]: 每次计算得到的系数集合 一行表示一个特征的系数变化, 一列为一次计算的所有系数
    """

    alpha, max_cycles = 0.001, 500  # 步长, 迭代次数
    data_mat = np.mat(data_mat_in)  # 使用matrix 直接使用 '*'相乘, ndarray需要np.dot
    _, feature_num = data_mat.shape
    label_mat = np.mat(class_labels).T # # (m,) -> (m, 1)
    weights = np.ones((feature_num, 1))  # 特征的回归系数,初始化为1 n * 1
    weights_all = weights.copy()
    for _ in range(max_cycles):
        # h_theta(x) = g(A)  A = x*theta
        h = sigmoid(data_mat*weights)  # m*n * n*1  -> m*1
        # 实际类别值 - 预测值 E = y - h_theta(x)
        error = (h - label_mat)  # m * 1
        # 0.001 * (3, n) * (n, 1) -> (3, 1) 每个特征(列)一个误差 得到x1,x2,xn的系数的偏移量
        # 梯度下降迭代公式: w = w - x^T * E
        weights -= alpha * data_mat.T * error  # (n * m) * (m * 1) -> n*1
        weights_all = np.concatenate((weights_all, weights), axis=1)
    return weights_all


def sto_grad_descent_0(data_mat, class_labels):
    """
    随机梯度下降(Stochastic Gradient Descent, SGD)
    `一次迭代`仅用`一个样本点`来更新回归系数，
    """
    m, n = data_mat.shape
    weights = np.ones(n)
    w_all = weights.copy()
    alpha = 0.01
    for j in range(m):
        # 每次迭代只计算一个样本的Cost的梯度
        # h_\theta(x^{(i)}
        h = sigmoid((data_mat[j]*weights).sum())  # 一个样本的预测值
        error = h - class_labels[j]
        weights -= alpha * error * data_mat[j]  # (n,)
        w_all = np.vstack((w_all, weights))
    return w_all.T


def sto_grad_descent_1(data_mat, class_labels):
    """
    改进,在整个数据集上迭代多次
    """
    m, n = data_mat.shape
    weights = np.ones(n)
    w_all = weights.copy()
    alpha = 0.01
    for _ in range(200):  # 在整个数据集上运行200次
        for j in range(m):
            # 每次迭代只计算一个样本的Cost的梯度
            # h_\theta(x^{(i)}
            h = sigmoid(np.sum(data_mat[j]*weights))  # 一个样本的预测值
            error =  h - class_labels[j]
            weights -= alpha * error * data_mat[j]
            w_all = np.vstack((w_all, weights))
    return w_all.T


def sto_grad_descent_2(data_mat, class_labels, iter_num=150):
    """
    再改进:
    1. alpha 在每次迭代的时候都会调整
    2. 随机选取样本拉来更新回归系数 减少周期性的波动
    """
    m, n = data_mat.shape
    weights = np.ones(n)
    w_all = weights.copy()

    for j in range(iter_num):  # 在整个数据集上运行x次
        data_index = list(range(m))  # 0, 1, ...m-1 整个训练样本序号集合
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4/(1+j+i) + 0.001
            # 随机从训练样本中抽取数据 进行Cost梯度下降,
            # 之后将这个数据从序号集合中删除
            rand_index = int(np.random.uniform(0, len(data_index)))
            # 一个样本的预测值
            h = sigmoid(np.sum(data_mat[data_index[rand_index]]*weights))
            error =  h - class_labels[data_index[rand_index]]
            weights -= alpha * error * data_mat[data_index[rand_index]]
            w_all = np.vstack((w_all, weights))
            del(data_index[rand_index])
    return w_all


def mb_grad_descent(data_mat_in, class_labels, iter_num=150):
    """小批量梯度下降
    批量梯度下降以及随机梯度下降的一个折中办法。
    其思想是：每次迭代 使用 batch_size 个样本来对参数进行更新
    data_mat: 二位矩阵,每行代表一个训练样本, 每列代表不同特征
    class_labels: 代表样本分类的矩阵
    """
    alpha = 0.01  # 步长,
    data_mat = np.mat(data_mat_in)  # 使用matrix 直接使用 '*'相乘, ndarray需要np.dot
    sample_num, feature_num = data_mat.shape
    label_mat = np.mat(class_labels).T # n * 1
    weights = np.ones((feature_num, 1))  # 特征的回归系数, 初始化为1  3*1 matrix
    weights_all = weights.copy()
    for _ in range(iter_num):
        # l = np.arange(0, sample_num, 10)
        # np.random.shuffle(l)
        for j in range(0, sample_num, 10):
            # h_theta(x) = g(A)  A = x*theta
            h = sigmoid(data_mat[j:j+10]*weights)  # 10*3 * 3*1  -> 10*1
            # 实际类别值 - 预测值 E = y - h_theta(x)
            error = (h - label_mat[j:j+10])
            # 0.001 * (3, 10) * (10, 1) -> (3, 1) 每个特征(列)一个误差 得到x1,x2,xn的系数的偏移量
            # 梯度上升迭代公式: w = w + x^T * E
            weights -= alpha * data_mat[j:j+10].T * error
            weights_all = np.concatenate((weights_all, weights), axis=1)
    return weights_all


def plt_weights(weights):
    # 参数收敛情况分析
    x = np.arange(weights.shape[1])
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    ax1.plot(x, w0, c='r')
    ax1.set_ylabel('w0')
    ax2.plot(x, w1, c='g')
    ax2.set_ylabel('w1')
    ax3.plot(x, w2, label='w2', c='b')
    ax3.set_ylabel('w2')
    plt.xlabel('iteration')
    ax1.set_title('weights', loc='center')


def plot_best_fit(data_mat, label, weights):
    # 画出数据和拟合曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # data_mat, label = load_data_set('TestSet.txt')
    ax.scatter(data_mat[np.where(label==1), 1], data_mat[np.where(label==1), 2], s=30, c='red', label='label=1')
    ax.scatter(data_mat[np.where(label==0), 1], data_mat[np.where(label==0), 2], s=20, c='b', label='label=0')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # 回归线 \theta^Tx = w0+w1x1+w2x2=0 -> 令Sigmoid函数输入值为0 这是2个分类的分界点
    x = np.linspace(-4, 4, 80)  # 设x1为x轴
    y = -(weights[0] + weights[1]*x) / weights[2]  # x2为y轴
    ax.plot(x, y, label='Regression')
    ax.set_title('best_fit')
    plt.legend(loc="best")



def test_lr():
    data_mat, label = load_data_set(TEST_DATA_PATH)
    # weights = batch_grad_descent(data_mat, label)
    # weights = sto_grad_descent_0(data_mat, label)
    # weights = sto_grad_descent_1(data_mat, label)
    # weights = sto_grad_descent_2(data_mat, label)
    weights = mb_grad_descent(data_mat, label)
    plt_weights(weights)
    plot_best_fit(data_mat, label, weights[:, -1])
    plt.show()


# ------------------------------------------------
def sigmoid_single(inx):
    # sigmod阶跃函数, inx输入为单个数字
    if inx >= 0:  # 防止 inx为负数时出现极大数字
        return 1 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


def load_data(file_name):
    data_all = np.loadtxt(file_name, delimiter='\t')
    features, labels = data_all[:, :-1], data_all[:,-1]
    return features, labels


def classify(input_feat, weights):
    # h = sigmoid(np.dot(input_feat, weights))  # (m,)
    # h_list = []
    # for z in np.dot(input_feat, weights):
    #     h_list.append(sigmoid_single(z))
    h_list = list(map(sigmoid_single, np.dot(input_feat, weights)))
    h = np.array(h_list)
    result = np.where(h>0.5, 1, 0)
    return result


def stoc_grad_descent(data_mat, class_labels, iter_num=150):
    """
    再改进:
    1. alpha 在每次迭代的时候都会调整
    2. 随机选取样本拉来更新回归系数 减少周期性的波动
    只返回最终结果
    """
    m, n = data_mat.shape
    weights = np.ones(n)
    for j in range(iter_num):  # 在整个数据集上运行x次
        data_index = list(range(m))  # 0, 1, ...m-1 整个训练样本序号集合
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4/(1+j+i) + 0.001
            # 随机从训练样本中抽取数据 进行Cost梯度下降,
            # 之后将这个数据从序号集合中删除
            rand_index = int(np.random.uniform(0, len(data_index)))
            # 一个样本的预测值
            h = sigmoid_single(np.sum(data_mat[data_index[rand_index]]*weights))
            error =  h - class_labels[data_index[rand_index]]
            weights -= alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights


def test_colic():
    train_features, train_labels = load_data(HORSE_TRAIN_PATH)
    test_features, test_labels = load_data(HORSE_TEST_PATH)
    weights = stoc_grad_descent(train_features, train_labels, 500)
    result = classify(test_features, weights)  # m*n (n,)
    test_labels = test_labels.astype(int)  # (m,)
    errors = test_labels ^ result  # 异或, 判断错误的为1
    error_rate = errors.sum()/errors.shape[0]
    print(f"本次错误率:{errors.sum()/errors.shape[0]:.2%}")
    return error_rate


def multi_test():
    test_num = 10
    error_count = 0
    for _ in range(test_num):
        error_count += test_colic()
    print(f"最终错误率:{error_count/test_num:.2%}")


if __name__ == "__main__":
    # test_lr()
    # test_colic()
    multi_test()
