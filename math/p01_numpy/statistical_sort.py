#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-01 14:24:38
@Description: 统计相关的函数
@LastEditTime: 2019-11-01 16:36:26
'''
import numpy as np


a = np.array([[353, 75, 69], [46, 63, 58], [43, 65, 232]])

print("max = amax展开:", np.amax(a), a.max())
#  353 353
print("max 沿轴0:", a.max(0))
# [353  75 232]

print("min 沿轴1:", np.amin(a, 1))
# [69 46 43]

# PTP 范围 max - min
print("轴0范围:", np.ptp(a, 0))
print("轴1范围:", np.ptp(a, 1))
print("不指定轴, 展开:", a.ptp())
"""
轴0范围: [310  12 174]
轴1范围: [284  17 189]
不指定轴, 展开: 310
"""

# percentile(a, q, axis) 表示小于这个值得观察值占某个百分比
# 一个多维数组的任意百分比分位数，此处的百分位是从小到大排列
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print("处于60%分位的数值:", np.percentile(a, 60))  # 57.99999999999999
print("处于50%分位的数值, 沿着0轴:", np.percentile(a, 50, axis=0))  # [50. 40. 60.]

# median 中值
# mean 算术平均值
# average() 加权平均值, 不加权时等于mean

wts = np.array([4, 3, 2, 1])
a = np.array([1, 2, 3, 4])

print("加权平均数", np.average(a, weights=wts))
# 加权平均数 2.0

# 方差 var = mean((x-x.mean())**2)
# 标准差 std = sqrt(mean((x - x.mean())**2))
print('方差:', np.var(a))
#  1.25
print('标准差:', np.std(a))
# 1.118033988749895


# 排序sort(a, axis, kind, order) 默认快速排序
"""
种类                  速度  最坏情况  工作空间 稳定性
'quicksort' （快速排序） 1  O(n^2)     0       否
'mergesort' （归并排序） 2  O(n*log(n)) ~n/2   是
'heapsort' （堆排序）    3  O(n*log(n)) 0      否
"""
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array(
    [('July', 12), ('Rose', 21), ('Andrew', 11), ('Aton', 17)], dtype=dt)
print(a)
"""
[(b'July', 12) (b'Rose', 21) (b'Andrew', 11) (b'Aton', 17)]
"""
print('指定order排序:', np.sort(a, order='age'))
# [(b'Andrew', 11) (b'July', 12) (b'Aton', 17) (b'Rose', 21)]

# argsort 对输入数组沿给定轴执行间接排序，并使用指定排序类型
# 返回数据的索引数组。 这个索引数组用于构造排序后的数组
x = np.array([3, 2, 1, 4, 6, 5])
y = x.argsort()

print("argsort返回的索引数组", y)
# [2 1 0 3 5 4]
print("使用索引数组重构原数组", x[y])
# [1 2 3 4 5 6]

# lexsort()  使用键序列执行间接排序
nm = ('raju','anil','ravi','amar')
dv = ('f.y.', 's.y.', 's.y.', 'f.y.')
ind = np.lexsort((dv, nm))
print([nm[i] + dv[i] for i in ind])
# ['amarf.y.', 'anils.y.', 'rajuf.y.', 'ravis.y.']

# argmax, argmin 返回沿指定轴的最大和最小值
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print("最小元素(展开)下标和值", a.argmin(), a.flatten()[a.argmin()])
print("沿0轴的最大值的索引", a.argmax(0))

# where 返回输入数组中满足给定条件的元素的索引

x = np.arange(9.).reshape(3, 3)
y = np.where(x>4)
print('满足大于4的索引', y)
print('利用索引获取满足条件的元素', x[y])
"""
满足大于4的索引 (array([1, 2, 2, 2], dtype=int64), array([2, 0, 1, 2], dtype=int64))
利用索引获取满足条件的元素 [5. 6. 7. 8.]
"""

# extract() 函数返回满足任何条件的元素
condition = np.mod(x, 2) == 0
print('按元素的条件值\n', condition)

print('使用条件取值', np.extract(condition, x))
print(x[condition])
"""
使用条件取值 [0. 2. 4. 6. 8.]
[0. 2. 4. 6. 8.]
"""
