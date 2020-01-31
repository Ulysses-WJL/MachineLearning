#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 11:41:32
@Description: IO 相关
@LastEditTime: 2019-11-04 11:47:57
'''
import os.path
import numpy as np


current_dir = os.path.dirname(os.path.realpath(__file__))
np_file = os.path.join(current_dir, 'outfile.npy')
np_text = os.path.join(current_dir, 'outfile.txt')

# npy格式
a = np.arange(8).reshape(2, 2, 2)
np.save(np_file, a)

b = np.load(np_file)
print(b)
"""
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
"""

# 简单文本格式
a = np.arange(9).reshape(3, 3)
np.savetxt(np_text, a)
b = np.loadtxt(np_text)
print(b)
"""
[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
"""

# c = np.loadtxt(
#     'D:\workspace\study\math_related\p01_numpy\stocks.csv',
#     delimiter=',', dtype=[('name', 'S20'), ('price', float), ('date', 'S20'), ('time', 'S20'), ('delta', float), ('shares', int)])
# print(c)

# 使用unpack 返回结果是解开的
p, d = np.loadtxt(
    r"D:\workspace\study\math_related\p01_numpy\stocks.csv",
    usecols=(1, -2), delimiter=',', unpack=True)
print(p)
print(d)
