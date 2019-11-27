#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Description: kNN近邻算法: 优化约会网站的配对效果
@Date: 2019-10-29 19:44:50
@Author: Ulysses
@LastEditTime: 2019-11-04 20:09:40
'''
import os
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""
开发流程
收集数据：提供文本文件
准备数据：使用 Python 解析文本文件
分析数据：使用 Matplotlib 画二维散点图
训练算法：此步骤不适用于 k-近邻算法
测试算法：使用海伦提供的部分数据作为测试样本。
        测试样本和非测试样本的区别在于：
            测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
适用数据范围：数值型和标称型

交往对象的类型:
1. 不喜欢的人
2. 魅力一般的人
3. 极具魅力的人
数据文本 datingTestSet2.txt
40920	8.326976	0.953952	3
这4列分别为
- 每年获得的飞行常客里程数
- 玩视频游戏所耗时间百分比
- 每周消费的冰淇淋公升数
- 对象类型1, 2或3
"""
current_dir = os.path.dirname(os.path.realpath(__file__))


def draw_pic(array1, lst):
    fig = plt.figure()
    # 作为子图布置的一部分，向图中添加轴
    ax = fig.add_subplot(111)  # (1, 1, 1)
    # 绘制散点图 3种类型 不同颜色
    color_dict = {'1': 'Red', '2': 'Blue', '3': 'Green'}
    colors = list(map(color_dict.get, lst))
    print(len(colors), colors[1], colors[0])
    ax.scatter(array1[:, 0], array1[:, 1], c=colors)
    plt.show()


def file2matrix(filename):
    """
    读取数据到矩阵
    Args:
        filename ([type]): [description]
    Returns:
        [list]: 对象类型
        [array]: 对象特征
    """
    with open(filename, encoding='utf-8') as f:
        # 根据行数创建矩阵
        lines = len(f.readlines())

        # 记录 各对象特征
        array1 = np.zeros((lines, 3))
        # 记录 各对象的类型
        three_types = []
    with open(filename, encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            data = line.strip().split()
            array1[index, :] = data[0:3]
            three_types.append(data[-1])
    # draw_pic(array1, three_types)
    #x, y轴数据不在一个量级,归一化
    return three_types, array1


def auto_normal(dataset):
    """
    归一化,将所有特征都归一化
    Args:
        dataset ([type]): [description]
    Returns:
        [type]: [description]
    """
    # dataset.shape: 1000, 2
    max_value = dataset.max(0)  # 1000个一维数组对比, 每个元素取最大 (max1, max2)
    min_value = dataset.min(0)  # (min1, min2, ...)

    # 极差
    ranges = max_value - min_value  #
    nums = dataset.shape[0]  # 数量
    normal_data = np.zeros(dataset.shape)
    # 对应生成 X - Xmin
    # np.tile(min_value, (nums, 1)) -> 包含nums个min_value的矩阵
    normal_data = dataset - np.tile(min_value, (nums, 1))
    # 最后生成(X-Xmin)/(Xmax-Xmin)
    normal_data = normal_data / np.tile(ranges, (nums, 1))

    return normal_data, ranges, min_value


def classify(index, dataset, labels, k):
    """
    验证kNN算法是否正确
    Args:
        index ([type]): 当前人员的特征
        dataset ([type]):样本人员特征集合
        labels ([type]): 样本人员分类list
        k ([type]):选取的k值
    """
    data_size = dataset.shape[0]
    # 计算当前点 与特征集合中的点的距离 sqrt((x1-x2)**2, (y1-y2)**2)
    diff_mat = np.tile(index, (data_size, 1)) - dataset
    square_diff_mat = diff_mat**2

    distance_diff_mat = square_diff_mat.sum(axis=1)  # 一维数组内的数据相加
    distances = distance_diff_mat**0.5  # 当前点到各个数据特征点的距离

    # 只关心距离远近排序 0, 1, 6, 2, 7, 3, 4, 5 ...
    # b = a.argsort() a = a[b]  a升序排列
    dis_rank = distances.argsort()

    class_count = Counter()  # 记录这k个元素的分类
    for i in range(k):
        vote_label = labels[dis_rank[i]]  # 离它最近的第i个数据的 分类
        # 记录k个最近人员的分类
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    return class_count.most_common(1)[0][0]  # 最多的那个分类


def dating_class_test():
    """
    对约会网站的测试方法
    """
    # 测试数据比 (训练数据比 = 1- ho_ratio)
    ho_ratio = 0.1
    filename = r"datingTestSet2.txt"
    file_path = os.path.join(current_dir, filename)
    labels, array = file2matrix(file_path)
    # 归一化数据
    normal_array, _, _ = auto_normal(array)
    # 数据条数
    size = normal_array.shape[0]
    test_num = int(ho_ratio * size)
    print('测试\n')
    error_count = 0.0
    for i in range(test_num):
        # k取3
        classifier_result = classify(normal_array[i, :],
                                     normal_array[test_num:, :],
                                     labels[test_num:], 3)
        if classifier_result != labels[i]:
            error_count += 1.0
            print(f"错误!当前第{i}个测试项, 实际分类{labels[i]}, 计算的分类{classifier_result}")
    print(f'总的错误率为: {(error_count / test_num):.2%}')
    """
    错误!当前第22个测试项, 实际分类2, 计算的分类3
    错误!当前第74个测试项, 实际分类1, 计算的分类3
    错误!当前第83个测试项, 实际分类1, 计算的分类3
    错误!当前第91个测试项, 实际分类3, 计算的分类2
    错误!当前第99个测试项, 实际分类1, 计算的分类3
    总的错误率为: 0.05%
    """


def classify_person():
    person_types = ['讨厌的人', '魅力一般的人', '极具魅力的人']
    fly_miles = float(input('每年获得的飞行常客里程数?\n'))
    game_percents = float(input('玩视频游戏所耗时间比?\n'))
    ice_cream = float(input('每周消费的冰淇淋公升数?\n'))
    filename = r"datingTestSet2.txt"
    file_path = os.path.join(current_dir, filename)
    labels, array = file2matrix(file_path)
    count = Counter(labels)
    # print(count)
    # 归一化数据
    normal_array, ranges, min_val = auto_normal(array)
    array = np.array([fly_miles, game_percents, ice_cream])
    nor_array = (array-min_val) / ranges
    classifier_result = classify(
        nor_array, normal_array, labels, 3)
    print(f'你有可能的类型: {person_types[int(classifier_result)-1]}')


if __name__ == "__main__":

    # dating_class_test()
    classify_person()
