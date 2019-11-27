#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 09:07:40
@Description: 副本, 视图
@LastEditTime: 2019-11-04 09:30:24
'''
import numpy as np
"""
在执行函数时，其中一些返回输入数组的副本，而另一些返回视图。 当内容物理存
储在另一个位置时，称为副本。 另一方面，如果提供了相同内存内容的不同视图，
我们将其称为视图。
"""

a = np.arange(24).reshape(3, 8)
# 无复制, 简单的赋值不会创建数组对象的副本。相反，它使用原始数组的相同id()来访问它
b = a
print(id(b), id(a))
b.shape = 4, 6
print(a)  # a的形状也改变了
"""
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
"""


# 视图view或浅复制
a = np.arange(6).reshape(3, 2)
# 生成新的数组对象，并可查看原始数组的相同数据
b = a.view()
print(id(a), id(b))  # id 不同
# 新数组的维数更改不会更改原始数据的维数
b.shape = 2, 3
print(b)  # 不会改变a
"""
[[0 1 2]
 [3 4 5]]
"""
print(a)
"""
[[0 1]
 [2 3]
 [4 5]]
"""
# 数组切片 也会产生视图
a_view1 = a[1:2, 3:6]    # 切片 slice
a_view2 = a[:100]        # 同上
a_view3 = a[::2]         # 跳步
a_view4 = a.ravel()      # 展平


# 深复制 ndarray.copy() 数组及其数据的完整副本，不与原始数组共享。
a = np.array([[1, 2], [3, 4], [9, 100]])
b = a.copy()
print(b is a)
print(id(b), id(a))  # 不同
b[0, 0] = 0
print(b)
"""
[[  0   2]
 [  3   4]
 [  9 100]]
"""
print(a)
"""
[[  1   2]
 [  3   4]
 [  9 100]]
"""
a_copy1 = a[[1,2], [0, 1]]   # 用 index 选
a_copy2 = a[[True, True, False], [False, True]]  # 用 mask
a_copy3 = a[[1,2], :]        # 虽然 1,2 的确连在一起了, 但是他们确实是 copy
a_copy4 = a[a[1,:] != 0, :]  # fancy indexing
a_copy5 = a[np.isnan(a), :]  # fancy indexing
