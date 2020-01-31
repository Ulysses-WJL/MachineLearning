#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-28 09:57:46
@Description: numpy库的使用
@LastEditTime: 2019-10-31 14:26:09
'''
import numpy as np

# 矩阵
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int)
print(array, type(array))
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]] <class 'numpy.ndarray'>

print(array.shape)  # 行数和列数
# (3, 3)

print(array.dtype)  # 数据类型
# int32

print(array.ndim)  # 维度
# 2

print(array[:,1])  # 切片
# [2 5 8]


# 初始化 使用特定数据

array0 = np.zeros((3, 4))  # 创建全零数组  zeros(shape, dtype, order)
print("array0\n", array0)
"""
array0
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
"""

array1 = np.ones((4, 3), np.int)  # 创建全1数组
#   (shape, dtype=None, order='C') C 行优先 F列优先
print("array1\n", array1)
"""
array1
 [[1 1 1]
 [1 1 1]
 [1 1 1]
 [1 1 1]]
"""

array2 = np.empty((2, 3))  # 全空数组 数组元素为随机值，因为它们未初始化
print("array2\n", array2)
"""
array2
 [[4.05e-322 0.00e+000 0.00e+000]
 [0.00e+000 0.00e+000 0.00e+000]]
"""

# 自定义类型
x  = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print('自定义类型:\n', x)
"""
 [[(0, 0) (0, 0)]
 [(0, 0) (0, 0)]]
"""

# python序列转numpy ndarray
x = [[(1, 2, 3), (2, 3)],[(2, 3, 5), (1,)]]
a = np.asarray(x)
print("python序列转ndarray", a)


# frombuffer  此函数将缓冲区解释为一维数组
s = b"Hello World"
a = np.frombuffer(s, dtype='S1')
print("从缓冲区读取数据:", a)
# 从缓冲区读取数据: [b'H' b'e' b'l' b'l' b'o' b' ' b'W' b'o' b'r' b'l' b'd']

# fromiter 从可迭代对象中构建一个ndarray对象(一维数组)
a = np.fromiter(range(10), dtype='u1')
print("从可迭代对象中构建:", a)
# [0 1 2 3 4 5 6 7 8 9]

array3 = np.arange(10, 20, 2)  # 创建连续数组
print("array3\n", array3)

"""
array3
 [10 12 14 16 18]
"""

array4 = np.arange(12).reshape((4, 3))  # 使用 reshape 改变数据的形状 4*3
print("array4\n", array4)
"""
array4
 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
"""

array5 = np.linspace(10, 20, 20)  # 创建线段型数据
#  # 开始端1，结束端10，且分割成20个数据，生成线段
print("array5\n", array5)
"""
array5
 [10.         10.52631579 11.05263158 11.57894737 12.10526316 12.63157895
 13.15789474 13.68421053 14.21052632 14.73684211 15.26315789 15.78947368
 16.31578947 16.84210526 17.36842105 17.89473684 18.42105263 18.94736842
 19.47368421 20.        ]
"""
print(array5.reshape((4, 5)))
"""
[[10.         10.52631579 11.05263158 11.57894737 12.10526316]
 [12.63157895 13.15789474 13.68421053 14.21052632 14.73684211]
 [15.26315789 15.78947368 16.31578947 16.84210526 17.36842105]
 [17.89473684 18.42105263 18.94736842 19.47368421 20.        ]]
"""

# logspace 对数刻度
a = np.logspace(1, 6, num=6, dtype='i4')
print('对数刻度均匀分布:', a)
# [     10     100    1000   10000  100000 1000000]
a = np.logspace(1, 6, num=6, dtype='i1', base=2)
print("以2为底数:", a)
# [ 2  4  8 16 32 64]

# 属性
print(array0.size)  # 元素个数
print(array1.shape)  # 行数 列数
print(array2.ndim)  # 维度

A = np.arange(0, 8).reshape((2, 2, 2))
print(A)
"""
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
"""
print(A.ndim)
