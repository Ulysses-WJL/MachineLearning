#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-28 11:22:46
@Description: numpy 基础运算
@LastEditTime: 2019-11-01 14:11:40
'''
import numpy as np


a = np.array([10, 20, 30, 40])
b = np.arange(4)  # [0, 1, 2, 3]


# 一维矩阵
print("减", a - b)
# [10 19 28 37]

print("加", a + b)
# [10 21 32 43]

print("对应元素相乘", a * b)
# [  0  20  60 120]

print("每个元素的乘方", b ** 2)
# [0 1 4 9]

print("对每个元素求三角函数", np.sin(b))
#  [0.         0.84147098 0.90929743 0.14112001]

print("逻辑判断", b < 3)
# [ True  True  True False]

# 倒数
a = np.array([0.25, 1, 0, 100])
print("计算倒数 0->inf", np.reciprocal(a))
# RuntimeWarning: divide by zero encountered in reciprocal
# 计算倒数 0->inf [4.   1.    inf 0.01]
b = np.array([100])
print("大于 1 的整数元素，结果始终为 0:", np.reciprocal(b))
# [0]

# power
a = np.array([2, 3, 5, 7])
b = np.array([1, 2, 2, 1])
print('power函数:', np.power(a, 2))
print('power函数:', np.power(a, b))
"""
power函数: [ 4  9 25 49]
power函数: [ 2  9 25  7]
"""
# mod remainder
print("a/b 余数mod", np.mod(a, b))
print("a/b 余数remainder", np.remainder(a, b))
"""
a/b 余数mod [0 1 1 0]
a/b 余数remainder [0 1 1 0]
"""

# 广播
"""
如果满足以下规则，可以进行广播：
- ndim 较小的数组会在前面追加一个长度为 1 的维度。
- 输出数组的每个维度的大小是输入数组该维度大小的最大值。
- 如果输入在每个维度中的大小与输出大小匹配，或其值正好为 1，则在计算中
  可它。
- 如果输入的某个维度大小为 1，则该维度中的第一个数据元素将用于该维度的
  所有计算。
如果上述规则产生有效结果，并且满足以下条件之一，那么数组被称为可广播的。
- 数组拥有相同形状。
- 数组拥有相同的维数，每个维度拥有相同长度，或者长度为 1。
- 数组拥有极少的维度，可以在其前面追加长度为 1 的维度，使上述条件成立。
"""
a = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
b = np.array([1, 2, 3])
print("广播, 较小的数组广播到较大数据的大小\n", a+b)
"""
 [[ 1  2  3]
 [11 12 13]
 [21 22 23]]
"""

# 多维矩阵
a = np.arange(4).reshape((2, 2))  # [[0, 1], [2, 3]]
b = np.array([[1, 1], [0, 1]])

print("点乘 ab\n", np.dot(a, b))
"""
[[0 1]
 [2 5]]
"""

"""
矩阵乘法理解
矩阵A
a11 * x1 + a12 * x2 = y1
a21 * x1 + a22 * x2 = y2
 a11 a12    x1      y1
          *     =
 a21 a22    x2      y2
矩阵B
b11 * t1 + b12 * t2 = x1
b21 * t1 + b22 * t2 = x2
 b11 b12     t1     x1
          *      =
 b21 b22     t2     x2

将第二个带入第一个
(a11*b11 + a11*b12)*t1 + (a12*b11 + a12*b12)*t2 = y1
(a21*b21 + a21*b22)*t1 + (a22*b21 + a22*b22)*t2 = y1

a11 a12   b11 b12    a11*b11 + a11*b12  a12*b11 + a12*b12
        *          =
a21 a22   b21 b22    a21*b21 + a21*b22  a22*b21 + a22*b22

A=(aij)m*k    B=(bij)k*n
      k
cij = ∑ ait * btj
      t=1
"""

print("点乘 b*a\n", b.dot(a))
"""
 [[2 4]
 [2 3]]
"""

rand_a = np.random.randint(1, 100, (2, 4))  # 随即生成2*4矩阵 范围1-100
print("rand_a", rand_a)
print(np.max(rand_a), np.min(rand_a), np.sum(rand_a), np.average(rand_a))
print("每一行最大", np.max(rand_a, axis=1))  # 一维矩阵内4个数求最大
print("每一列之和", np.sum(rand_a, axis=0))  # 2个一维矩阵求和
a = np.arange(13, 1, -1).reshape((3, 4))
print("中值mean", np.mean(a), a.mean())
# 中值mean 7.5 7.5

print("从头累加尾", np.cumsum(a), a.cumsum())
# [13 25 36 46 55 63 70 76 81 85 88 90]
print("类差\n", np.diff(a))
"""
每一行中后一项与前一项之差
[[1 1 1]
 [1 1 1]
 [1 1 1]]
"""
print(np.nonzero(a))
# 将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵
# (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64),
#  array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))

# 排序 每一行或列
print("每一行排序\n", np.sort(a, axis=-1))
"""
[[10 11 12 13]
 [ 6  7  8  9]
 [ 2  3  4  5]]
"""
print("排序,展平", np.sort(a, axis=None))
# [ 2  3  4  5  6  7  8  9 10 11 12 13]

print("转置\n", a.T)  # np.transpose(A)
"""
[[13  9  5]
 [12  8  4]
 [11  7  3]
 [10  6  2]]
"""

print("clip\n", np.clip(a, 5, 9))  # 小于最小值或大于最大值的会被转为最大或最小值
"""
 [[9 9 9 9]
 [9 8 7 6]
 [5 5 5 5]]
"""

# 舍入 around floor ceil
a = np.array([1.0,5.55, 123, 0.567, 25.532])
print("around:", np.around(a, decimals=1))
print("around:", np.around(a, decimals=-1))
print("floor:", np.floor(a))  # 返回不大于输入参数的最大整数
print("ceil:", np.ceil(a))  # 函数返回输入值的上限(最小整数)
"""
around: [  1.    5.6 123.    0.6  25.5]
around: [  0.  10. 120.   0.  30.]
floor: [  1.   5. 123.   0.  25.]
ceil: [  1.   6. 123.   1.  26.]
"""

"""
位操作
1. bitwise_and 对数组元素执行位与操作
2. bitwise_or 对数组元素执行位或操作
3. invert 计算位非
4. left_shift 向左移动二进制表示的位
5. right_shift 向右移动二进制表示的位
"""
a, b = 13, 17
print("按位与:", np.bitwise_and(a, b))
print("按位或:", np.bitwise_or(a, b))
# 按位与: 1
# 按位或: 29

""" 字符串操作
1. add() 返回两个 str 或 Unicode 数组的逐个字符串连接
2. multiply() 返回按元素多重连接后的字符串
3. center() 返回给定字符串的副本，其中元素位于特定字符串的中央
4. capitalize() 返回给定字符串的副本，其中只有第一个字符串大写
5. title() 返回字符串或 Unicode 的按元素标题转换版本
6. lower() 返回一个数组，其元素转换为小写
7. upper() 返回一个数组，其元素转换为大写
8. split() 返回字符串中的单词列表，并使用分隔符来分割
9. splitlines() 返回元素中的行列表，以换行符分割
10. strip() 返回数组副本，其中元素移除了开头或者结尾处的特定字符
11. join() 返回一个字符串，它是序列中字符串的连接
12. replace() 返回字符串的副本，其中所有子字符串的出现位置都被新
字符串取代
13. decode() 按元素调用 str.decode
14. encode() 按元素调用 str.encode
"""

a = np.array([['hello'], ['fuck']])
b = np.array([[' world'], [' you']])
print("字符相加: \n", np.char.add(a, b))
"""
[['hello world']
 ['fuck you']]
"""
