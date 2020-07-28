#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-31 08:01:51
@Description: axis的选择
@LastEditTime: 2019-10-31 09:47:03
'''
import numpy as np

array = np.random.randint(1, 100, (2, 3, 4))

print(array.shape, array)
"""
(2, 3, 4)
[[[66 17 18 21]
  [13  5 51 97]
  [72 81 55  8]]

 [[46 45 64 48]
  [12 92  6 81]
  [42 76 38 26]]]
"""

print('0: ', array.max(0))
# 2个二维矩阵对象(3*4)取最大值, 2个矩阵每一位对应比较大小
"""
[[66 45 64 48]
 [13 92 51 97]
 [72 81 55 26]]
"""
print('1: ', array.max(1))
# 二维矩阵中3个一维矩阵对象取最大(2个二维矩阵分别做一次)
"""
[[72 81 55 97]
 [46 92 64 81]]
"""
print('2: ', array.max(2))
# 4个数取最大(每个二维矩阵都对一维矩阵中的4个数取一次最大)
"""
[[66 97 81]
 [64 92 76]]
"""

array1 = np.array([[1, 2, 3, 4], [2, 2, 3, 3]])  # shape 2*4
array2 = np.array([[4, 3, 2, 1], [3, 3, 2, 2]])

array_merge_0 = np.concatenate((array1, array2), axis=0)
array_merge_1 = np.concatenate((array1, array2), axis=1)
print("merge axis=0", array_merge_0)
""" 2个二维数组[A, B]和[C, D]合并->[A, B, C, D]
[[1 2 3 4]
 [2 2 3 3]
 [4 3 2 1]
 [3 3 2 2]]
"""
print("merge axis=1", array_merge_1)
""" 4个数合并 [a, b, c, d]和[e, f, g, h]合并 -> [a, b, c, d, e, f, g, h]
[[1 2 3 4 4 3 2 1]
 [2 2 3 3 3 3 2 2]]
"""
