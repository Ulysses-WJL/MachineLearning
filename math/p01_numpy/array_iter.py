#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-01 09:38:08
@Description: 数组上的迭代
@LastEditTime: 2019-11-04 19:05:57
'''
import numpy as np

# 迭代
a = np.arange(0, 60, 5).reshape(3, 4)
print(a)
"""
[[ 0  5 10 15]
 [20 25 30 35]
 [40 45 50 55]]
"""
# 迭代的顺序匹配数组的内容布局，而不考虑特定的排序
for x in np.nditer(a):
    print(x, end=', ')
print()
# 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
for x in np.nditer(a.T):
    print(x, end=', ')
print()
# 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,

# 设定迭代顺序
for x in np.nditer(a, order='F'):
    print(x, end=', ')
print()
# 0, 20, 40, 5, 25, 45, 10, 30, 50, 15, 35, 55,

# 使用迭代器修改元素 op_flag
print(id(a))
for item in np.nditer(a, op_flags=['readwrite']):
    print(type(item), item[...])  # <class 'numpy.ndarray'> 0
    item[...] = 2 * item

print(a)
"""
[[  0  10  20  30]
 [ 40  50  60  70]
 [ 80  90 100 110]]
"""

# 外部循环 迭代器遍历对应于每列的一维数组
for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=', ')
print()
# [ 0 40 80], [10 50 90], [ 20  60 100], [ 30  70 110],

# 广播迭代 数组 b 被广播到 a 的大小
b = np.array([1, 2, 3, 4], dtype=int)
for x, y in np.nditer([a, b]):
    print(f"{x}:{y}", end=', ')
print()
# 0:1, 10:2, 20:3, 30:4, 40:1, 50:2, 60:3, 70:4, 80:1, 90:2, 100:3, 110:4,
