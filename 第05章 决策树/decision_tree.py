#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-06 15:57:17
@Description: 决策树
@LastEditTime: 2019-11-07 16:19:16
'''
import pickle
from collections import Counter
from pprint import pprint
import numpy as np
import os.path
import plot_decision_tree as plot_dt

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
"""
开发流程
收集数据：可以使用任何方法。
准备数据：树构造算法 (这里使用的是ID3算法，只适用于标称型数据，这就是为什么数值型数据必须离散化。 还有其他的树构造算法，比如CART)
分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
训练算法：构造树的数据结构。
测试算法：使用训练好的树计算错误率。
使用算法：此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义。
"""

"""
项目概述
根据以下 2 个特征，将动物分成两类：鱼类和非鱼类。
特征：
不浮出水面是否可以生存
是否有脚蹼
"""
FISH_TYPE = np.dtype(
    [('no surfacing', 'b'), ('flippers', 'b'), ('is fish', 'b')])

def create_data_set():
    # 构造输入数据 最后一列是分类 前面是特征
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # 特征名称
    labels = ['no surfacing', 'flippers']
    # 都转成字符串
    data_set = np.array(dataSet).astype(str)
    return data_set, labels


def cal_shannon_ent(data_set):
    """
    计算信息熵 Ent(D) = -∑(pk * logpk) pk为第k类样本所占比例
    Args:
        data_set ([type]): 样本数据,最后一列是对应的分类
    """
    # 计算信息熵
    # num = len(data_set)
    num = data_set.shape[0]
    # 记录各分类出现次数
    label_count = Counter(data_set[:, -1])

    # for feature_vector in data_set:
    #     current_label = feature_vector[-1]
    #     # 记录各数据 分类
    #     label_count[current_label] = label_count.get(current_label, 0) + 1

    shannon_ent = .0
    # 计算信息熵
    for key in label_count:
        prob = label_count[key] / num
        shannon_ent -= prob * np.log(prob)
    return shannon_ent


def split_data_set(data_set, feature_col, value):
    """
    根据特征的取值分割数据集
    Args:
        data_set ([type]): 待划分的数据集
        feature_col ([type]): 该特征所在的列
        value ([type]): 特征值
    Returns:
        [type]: 特征feature_col为value的数据集合(不包括feature_col列 )
    """

    # ret_data = [feature_vector[:feature_col] +
    #             feature_vector[feature_col+1:] for feature_vector in data_set
    #             if feature_vector[feature_col] == value]
    ret_data = np.delete(
        data_set[data_set[:, feature_col]==value], feature_col, axis=1)
    return ret_data


def choose_best_feature(data_set):
    """
    选择最佳(熵增益最大)的特征对数据集进行划分
    Args:
        data_set ([type]): [description]
    Returns:
        [type]: [description]
    """
    # feature_nums = len(data_set[0]) - 1
    feature_nums = data_set.shape[1] - 1
    # 计算当前样本集合的信息熵
    base_entropy = cal_shannon_ent(data_set)
    # 最佳(大)的信息增益, 最佳分类特性所在列
    best_gain, best_feature_index = 0.0, -1
    for i in range(feature_nums):
        # 当前样本中 取第i个特性 组成的list
        feature_list = data_set[:, i] #  [data[i] for data in data_set]
        # 去重 根据这几项 讲数据集进行划分
        feature_value_set = set(feature_list)
        # 子结点信息熵的加权和
        sub_entropy = .0
        for value in feature_value_set:
            sub_data = split_data_set(data_set, i, value)
            prob = len(sub_data) / len(data_set)
            sub_entropy += prob * cal_shannon_ent(sub_data)
        # 根据第i个特性分类后的 信息增益
        info_gain = base_entropy - sub_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature_index = i

    return best_feature_index


def create_tree(data_set, labels):
    """
    递归地构造决策树
    Args:
        data_set ([type]): 当前数据集
        labels ([type]): 数据特征名称表

    Returns:
        [type]: 生成树的结点
    """
    # class_list = [data[-1] for data in data_set]
    class_list = list(data_set[:, -1])

    # 叶节点
    # 所有的数据的标签分类都相同, 直接使用该标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 数据集不能再划分下去, 取出现次数最多的分类作为标签
    if data_set.shape[1] == 1:
        return major_cnt(class_list)

    # 构造分支结点, 选取最优特性进行分类
    best_feature = choose_best_feature(data_set)

    best_label = labels[best_feature]

    # 基于字典表的tree结构
    decison_tree = {best_label: {}}
    # 从原先的分类名称中删去当前选择的分类
    labels.pop(best_feature)

    feature_values = data_set[:, best_feature] #  [data[best_feature] for data in data_set]
    # 根据选定的特征 的不同取值 划分出子结点
    unique_features = set(feature_values)
    for value in unique_features:
        # 递归地生成子结点
        # 注意, 子节点下分类标签可以与其祖先结点的兄弟节点相同,
        # 要保证的是一条路径上的分类标签不能出现重复
        sub_labels = labels[:]
        decison_tree[best_label][value] = create_tree(
            split_data_set(data_set, best_feature, value), sub_labels)

    return decison_tree


def major_cnt(class_list):
    """
    返回次数最多的那个分类
    Args:
        class_list ([type]): [description]
    Returns:
        [type]: [description]
    """
    class_counter = Counter(class_list)
    return class_counter.most_common(1)[0]


def classify(tree, feature_labels, test_vector):
    """
    给输入的数据分类
    Args:
        tree ([type]): 决策树模型
        feature_labels ([type]): 特征分类名称
        test_vector ([type]): 输入测试数据
    Returns:
        [type]:  测试数据的分类判决
    """
    # 根结点使用的分类标签名
    root_label = list(tree.keys())[0]
    child_nodes = tree[root_label]

    # 根结点使用的分类标签 的序号
    feature_index = feature_labels.index(root_label)

    # 输入的测试数据在此特征的取值
    feature_value = test_vector[feature_index]
    # 测试数据此
    child_value = child_nodes[feature_value]
    print('+++', root_label, 'xxx', child_nodes, '---', child_value, '>>>', feature_value)
    # 继续判别 知道到达叶节点
    if isinstance(child_value, dict):
        class_label = classify(child_value, feature_labels, test_vector)
    else:
        class_label = child_value
    return class_label


def store_tree(tree, file_name):
    with open(file_name, 'wb') as tree_file:
        pickle.dump(tree, tree_file)

def get_tree(file_name):
    with open(file_name, 'rb') as tree_file:
        return pickle.load(tree_file)


def fish_test():
    # 判断是否为鱼类
    data, labels = create_data_set()
    tree = create_tree(data, labels[:])
    # pprint(tree)
    print(classify(tree, labels, ['1', '1']))
    pt = plot_dt.PlotTree(tree)
    pt.plot()

def contact_lenses_test():
    # 测试一个人是否适合佩戴隐形眼镜
    file_path = os.path.join(CURRENT_DIR, 'lenses.txt')
    # 从文本获取数据
    data_set = np.loadtxt(file_path, delimiter='\t', dtype=str)
    print(data_set)
    # 特征的名称
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    # 构造决策树
    tree = create_tree(data_set, lenses_labels)
    # pprint(tree)
    pt = plot_dt.PlotTree(tree)
    pt.plot()


if __name__ == "__main__":
    # fish_test()
    contact_lenses_test()
    
