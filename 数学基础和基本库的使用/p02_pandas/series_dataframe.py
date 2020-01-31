#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-29 09:37:38
@Description: Pandas 主要2个数据结构Series和DataFrame
@LastEditTime: 2019-10-29 14:40:47
'''
import numpy as np
import pandas as pd

s = pd.Series([1, 2, 3, np.nan])
print(s)
"""
索引在左边，值在右边
0    1.0
1    2.0
2    3.0
3    NaN
dtype: float64
"""

print(pd.Series(1, dtype='float32'))
"""
0    1.0
dtype: float32
"""

print(pd.Series(1, index=list(range(4)), dtype='float32'))
""" 4 个相同
0    1.0
1    1.0
2    1.0
3    1.0
dtype: float32
"""

# 给定行标签和列标签
dates = pd.date_range("20190101", periods=6)
# data=None, index=None, columns=None, dtype=None, copy=False
df = pd.DataFrame(
    np.random.randint(1, 100, (6, 4)),
    index=dates,
    columns=['a', 'b', 'c', 'd'])
print(df)
"""
             a   b   c   d
2019-01-01  20  27  14  56
2019-01-02  38  97  49  47
2019-01-03  63  41  43   2
2019-01-04  29  53  14  67
2019-01-05  12  99  52  55
2019-01-06  80  95  47  97
"""

# 默认会从0开始生成index
df2 = pd.DataFrame(np.arange(12).reshape(3, 4))
print(df2)
"""
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
"""

# 使用dict作为数据, 指定每一列数据
df3 = pd.DataFrame({
    "A": 1.,
    "B": pd.Timestamp("20190101"),
    "C": pd.Series(1, index=list(range(4)), dtype='float32'),
    "D": np.array([3] * 4, dtype='int'),
    "E": pd.Categorical(['a', 'b', 'c', 'd']),
    "F": 'test'
})
print(df3)
"""
     A          B    C  D  E     F
0  1.0 2019-01-01  1.0  3  a  test
1  1.0 2019-01-01  1.0  3  b  test
2  1.0 2019-01-01  1.0  3  c  test
3  1.0 2019-01-01  1.0  3  d  test
"""

print("查看各数据类型\n", df3.dtypes)
"""
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
"""
print("查看序号\n", df3.index)
#  Int64Index([0, 1, 2, 3], dtype='int64')

print("每列数据名称\n", df3.columns)
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')

print("查看所有的值\n", df3.values)
"""
 [[1.0 Timestamp('2019-01-01 00:00:00') 1.0 3 'a' 'test']
 [1.0 Timestamp('2019-01-01 00:00:00') 1.0 3 'b' 'test']
 [1.0 Timestamp('2019-01-01 00:00:00') 1.0 3 'c' 'test']
 [1.0 Timestamp('2019-01-01 00:00:00') 1.0 3 'd' 'test']]
"""

print("查看数据总结\n", df3.describe())
"""
          A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
"""

print("转置\n", df3.T)
"""
                      0  ...                    3
A                    1  ...                    1
B  2019-01-01 00:00:00  ...  2019-01-01 00:00:00
C                    1  ...                    1
D                    3  ...                    3
E                    a  ...                    d
F                 test  ...                 test

[6 rows x 4 columns]

"""

print("对index(各列)排序\n", df3.sort_index(axis=1, ascending=False))
"""
       F  E  D    C          B    A
0  test  a  3  1.0 2019-01-01  1.0
1  test  b  3  1.0 2019-01-01  1.0
2  test  c  3  1.0 2019-01-01  1.0
3  test  d  3  1.0 2019-01-01  1.0
"""

print("对index(各行)排序\n", df3.sort_index(axis=0, ascending=False))
"""
      A          B    C  D  E     F
3  1.0 2019-01-01  1.0  3  d  test
2  1.0 2019-01-01  1.0  3  c  test
1  1.0 2019-01-01  1.0  3  b  test
0  1.0 2019-01-01  1.0  3  a  test
"""
print("对数据值进行排序(E列)\n", df3.sort_values(by='E', ascending=False))
"""
      A          B    C  D  E     F
3  1.0 2019-01-01  1.0  3  d  test
2  1.0 2019-01-01  1.0  3  c  test
1  1.0 2019-01-01  1.0  3  b  test
0  1.0 2019-01-01  1.0  3  a  test
"""

print(pd.Categorical(['a', 'b', 'c', 'd']))
# Categories (4, object): [a, b, c, d]
