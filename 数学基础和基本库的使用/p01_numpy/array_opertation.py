#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-01 09:43:00
@Description: 数组操作
@LastEditTime: 2019-11-01 11:14:51
'''
import numpy as np

# tile的使用 Construct an array by repeating A the number of times given by reps.
b = np.arange(4)
repeated = np.tile(b, 2)
print(repeated)  # [1 2 3 4 1 2 3 4]

repeated = np.tile(b, (2, 2))
print(repeated)
"""
[[1 2 3 4 1 2 3 4]
 [1 2 3 4 1 2 3 4]]
"""

# rollaxis 向后滚动特定的轴，其它轴的相对位置不会改变
a = np.arange(8).reshape(2, 2, 2)
print(a)
"""
[[[0 1]
  [2 3]]

 [[4 5]
  [6 7]]]
"""

print('将轴2滚动到轴0\n', np.rollaxis(a, 2))
# 轴2 一维数组内 [0 , 1] -> 轴0 二个二维数组
"""
 [[[0 2]
  [4 6]]

 [[1 3]
  [5 7]]]
"""

# swapaxes 交互2个轴
print("交换轴0(深度)和轴2(宽度)\n", np.swapaxes(a, 2, 0))
# 轴0 2个二维数组 对应元素 (0, 4) (1, 5) (2, 6) (3, 7)
# 轴2 一维数组内2个元素  (0, 1), ... (6, 7)
"""
[[[0 4]
  [2 6]]

 [[1 5]
  [3 7]]]

"""

# 广播操作 broadcast
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
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])  # 追加一个长度为 1 的维度[[4, 5, 6]]

# 对y广播x
b = np.broadcast(x, y) # 它拥有 iterator 属性，基于自身组件的迭代器元组
print("对y广播x", b.shape)
r, c = b.iters
for item1, item2 in b:  # zip(r, c)
    print(item1, item2)
"""
对y广播x (3, 3)
1 4
1 5
1 6
2 4
2 5
2 6
3 4
3 5
3 6
"""

# broadcast_to(array, shape, subok)将数组广播到新形状。它在原始数组上返回只读视图
a = np.arange(4).reshape(1, 4)  # [[0], [1], [2], [3]]
print("将数组广播到新形状\n", np.broadcast_to(a, (4, 4)))
"""
将数组广播到新形状
 [[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
"""

# expand_dims 扩展数组
a = np.arange(4)
b = np.expand_dims(a, 0)
c = np.expand_dims(a, 1)
print('a:\n', a, a.shape)
print("扩展数组b\n", b, b.shape)
print("扩展数组c\n", c, c.shape)
"""
a:
[0 1 2 3] (4,)
扩展数组b
 [[0 1 2 3]] (1, 4)
扩展数组c
 [[0]
 [1]
 [2]
 [3]] (4, 1)
"""

# squeeze 函数从给定数组的形状中删除一维条目
x = np.arange(9).reshape(1, 3, 3, 1)
print("原始x\n", x)
y = np.squeeze(x)
print("删除轴0位置一维\n", y, y.shape)
"""
原始x
 [[[[0]
   [1]
   [2]]

  [[3]
   [4]
   [5]]

  [[6]
   [7]
   [8]]]]
删除轴0位置一维
 [[0 1 2]
 [3 4 5]
 [6 7 8]] (3, 3)
"""
# resize 返回指定大小的新数组
# 新大小大于原始大小，则包含原始数组中的元素的重复副本
a = np.array([[1, 2, 3], [4, 5, 6]])
print("原始a\n", a)
b = np.resize(a, (3, 2))
print("(3, 2)大小新数组\n", b, b.shape)
c = np.resize(a, (3, 3))
print("(3, 3)大小新数组\n", c, c.shape)
"""
原始a
 [[1 2 3]
 [4 5 6]]
(3, 2)大小新数组
 [[1 2]
 [3 4]
 [5 6]] (3, 2)
(3, 3)大小新数组
 [[1 2 3]
 [4 5 6]
 [1 2 3]] (3, 3)
"""

# append
print("append\n", np.append(a, [7, 8, 9]))  # 会被展开
# [1 2 3 4 5 6 7 8 9]

print("在轴0添加元素\n", np.append(a, [[7, 8, 9]], axis=0))

print("在轴1添加元素\n", np.append(a, [[10], [10]], axis=1))
"""
在轴0添加元素
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
在轴1添加元素
 [[ 1  2  3 10]
 [ 4  5  6 10]]
"""

# insert
a = np.array([[1,2],[3,4],[5,6]])
print("没有传递轴, 展开\n", np.insert(a, 2, [0, 0]))
print("指定轴0, 会广播\n", np.insert(a, 1, [0], axis=0))
print("指定轴0, 会广播\n", np.insert(a, 1, [0], axis=1))
"""
没有传递轴, 展开
 [1 2 0 0 3 4 5 6]
指定轴0, 会广播
 [[1 2]
 [0 0]
 [3 4]
 [5 6]]
指定轴0, 会广播
 [[1 0 2]
 [3 0 4]
 [5 0 6]]
"""

# delete 可以使用切片，整数或者整数数组
a = np.arange(12).reshape((3, 4))
print("删除, 没有axis, 展开\n", np.delete(a, 5))
print("删除, 第2列\n", np.delete(a, 1, axis=1))
print("删除大于5\n", np.delete(a, a[a>5]))

# unique
a = np.array([1, 2, 2, 8, 4, 5, 5, 7, 6])
print("去重:", np.unique(a))
u, indices_u, indices_all = np.unique(a, return_index=True, return_inverse=True)
print("去重后序列:", u)
print("不重复的元素的下标", indices_u)
print("利用去重后元素构成原始数据", indices_all, u[indices_all])
"""
去重: [1 2 4 5 6 7 8]
去重后序列: [1 2 4 5 6 7 8]
不重复的元素的下标 [0 1 4 5 8 7 3]
利用去重后元素构成原始数据 [0 1 1 6 2 3 3 5 4] [1 2 2 8 4 5 5 7 6]
"""

# byteswap 大小端切换
a = np.array([0x0001, 0x0100, 0x2233], dtype=np.int16)
print(list(map(hex, a)))
# ['0x1', '0x100', '0x2233']
a.byteswap(True)  # 原地交换
print(list(map(hex, a)))
# ['0x100', '0x1', '0x3322']

