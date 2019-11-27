#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 19:13:09
@Description: 手写数字识别
@LastEditTime: 2019-11-04 20:33:44
'''
import os.path
import os
from collections import Counter
import numpy as np
"""
开发流程
收集数据：提供文本文件。
准备数据：编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
分析数据：在 Python 命令提示符中检查数据，确保它符合要求
训练算法：此步骤不适用于 KNN
测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
         区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
         则标记为一个错误
使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
         数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统
"""
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIGITS_LIST = os.listdir(os.path.join(CURRENT_DIR, 'testDigits'))
TRAINING_LIST = os.listdir(os.path.join(CURRENT_DIR, 'trainingDigits'))


def img2vector(filename):
    vector = np.zeros((1, 1024))  # 32*32数据转为1*1024
    with open(filename) as img_file:
        for i in range(32):
            line = img_file.readline()
            for j in range(32):
                vector[0, 32*i+j] = int(line[j])
    return vector


def classify(test_data, sample_data, labels, k):
    """
    Arguments:
        test_data {[type]} -- 待分析数据
        sample_data {[type]} -- 样本数据
        labels {[type]} -- 样本数据分类结果
        k {[type]} -- knn K的取值

    Returns:
        [type] -- [分析的分类结果
    """
    # data_size = sample_data.shape[0]
    # 测试数据与样本数据各个位置上的差
    # diff_mat = np.tile(test_data, (data_size, 1)) - sample_data  # n×1024
    diff_mat = sample_data - test_data  # 广播 boardcast
    # 计算距离
    distance = np.sum(diff_mat**2, axis=1)  # （n,）
    distance = np.sqrt(distance)

    # 距离排序 获取对应序号
    dis_rank = distance.argsort()
    counter = Counter()  # 分类：次数
    for i in range(k):
        # 最近的k个值的分类
        vote_label = labels[dis_rank[i]]
        counter[vote_label] = counter.get(vote_label, 0) + 1
    return counter.most_common(1)[0][0]


def get_training_data():
    """
    获取训练数据
    Returns:
        [type] -- [description]
    """
    # 训练数据结果
    hw_labels = []
    # 训练数据向量 n*1024
    training_matrix = np.zeros((len(TRAINING_LIST), 1024))
    for index, file_name in enumerate(TRAINING_LIST):
        file_digit = file_name.split('.')[0]
        num = int(file_digit.split("_")[0])
        hw_labels.append(num)
        training_matrix[index, :] = img2vector(
            os.path.join(CURRENT_DIR, 'trainingDigits', file_name))

    return training_matrix, hw_labels

def hand_writing_test():
    training_data, training_labels = get_training_data()
    error_count = 0
    total = len(TEST_DIGITS_LIST)
    for file_name in TEST_DIGITS_LIST:
        real_path = os.path.join(CURRENT_DIR, 'testDigits', file_name)
        test_data = img2vector(real_path)
        test_label = int(file_name.split('_')[0])
        vote_label = classify(test_data, training_data, training_labels, 3)
        print(f"文件{file_name}的测试结果为{vote_label}，其实际为{test_label}")
        if test_label != vote_label:
            error_count += 1
    print(f"错误率: {error_count/total:.2%}")


if __name__ == "__main__":
    hand_writing_test()
