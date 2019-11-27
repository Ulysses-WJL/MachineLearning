#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-28 15:09:34
@Description: 索引
@LastEditTime: 2019-10-31 15:29:08
'''
import numpy as np


A = np.arange(1, 10)
print("一维索引", A[1], A[-1])
# 一维索引 2 9

A = np.arange(1, 10).reshape((3, 3))
print(A)
"""
[[1 2 3]
 [4 5 6]
 [7 8 9]]
"""
print(A[:,1])  # [2 5 8]
print("使用省略号:", A[..., 1])  # [2 5 8]

print(A[2])
# [7 8 9]

print("二维索引", A[1][1], A[2, 1])
# 二维索引 5 8

print("切片", A[1, 0:2])  # 行, 列
# 1行 0列和1列  [4 5]

# 迭代
for row in A:  # 按行
    print(row)
"""
[1 2 3]
[4 5 6]
[7 8 9]
"""

for column in A.T:  # 按列
    print(column)
"""
[1 4 7]
[2 5 8]
[3 6 9]
"""

print(A.flatten())  # 展开为一维
# [1 2 3 4 5 6 7 8 9]
for item in A.flat:  # 一个迭代器
    print(item, end=', ')
# 1, 2, 3, 4, 5, 6, 7, 8, 9,
print()

# 高级索引
x = np.array([[1, 2], [3, 4], [5, 6]])
print('x', x)
"""
[[1 2]
 [3 4]
 [5 6]]
"""
y = x[[0, 1, 2], [0, 1, 0]]  # 选取(0, 0) (1, 1) (2, 0)处元素
print('整数索引:', y)  # [1 4 5]

x = np.arange(12).reshape((4, 3))
print(x)
"""
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
"""
row_index = np.array([[0, 0], [3, 3]])
col_index = np.array([[0, 2], [0, 2]])
y = x[row_index, col_index]
print('x对角线元素', y)
"""
[[ 0  2]
 [ 9 11]]
"""

# 行使用切片, 列使用高级索引(会引起复制)
y = x[1:4, [1, 2]]

# 布尔索引
y = x[x>5]
print('布尔索引: ', y)
# [ 6  7  8  9 10 11]

a = np.array([np.nan, 1, 2, 4, np.nan, 5])
print("取补~", a[~np.isnan(a)])
# 取补~ [1. 2. 4. 5.]

# 过滤出非复数元素
a = np.array([1, 2+3j, 5, 3.5+2j])
print('非复数', a[np.iscomplex(a)])
# 非复数 [2. +3.j 3.5+2.j]
