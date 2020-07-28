#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-30 09:14:53
@Description: 数据合并 concat
@LastEditTime: 2019-10-30 10:17:58
'''
import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.ones((2, 4))*0, columns=['A', 'B', 'C', 'D'])
print(df1)
"""
     A    B    C    D
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
"""
df2 = pd.DataFrame(np.ones((2, 4))*1, columns=['A', 'B', 'C', 'D'])
df3 = pd.DataFrame(np.ones((2, 4))*2, columns=['A', 'B', 'C', 'D'])

# 沿着特定轴将pandas对象连接起来
res = pd.concat([df1, df2, df3], axis=0)  # 0 按行连接
print("按行连接\n", res)
"""
     A    B    C    D
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
0  2.0  2.0  2.0  2.0
1  2.0  2.0  2.0  2.0
"""

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print("重置index\n", res)
"""
      A    B    C    D
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  1.0  1.0  1.0  1.0
3  1.0  1.0  1.0  1.0
4  2.0  2.0  2.0  2.0
5  2.0  2.0  2.0  2.0
"""

df4 = pd.DataFrame(np.ones((6, 1)), columns=['E'])
res1 = pd.concat([res, df4], axis=1)
print("按列连接\n", res1)
"""
      A    B    C    D    E
0  0.0  0.0  0.0  0.0  1.0
1  0.0  0.0  0.0  0.0  1.0
2  1.0  1.0  1.0  1.0  1.0
3  1.0  1.0  1.0  1.0  1.0
4  2.0  2.0  2.0  2.0  1.0
5  2.0  2.0  2.0  2.0  1.0
"""

# join 合并方式
df1 = pd.DataFrame(
    np.ones((3, 4))*0, columns=['A', 'B', 'C', 'D'], index=[1, 2, 3])
df2 = pd.DataFrame(
    np.ones((3, 4))*1, columns=['F', 'C', 'D', 'E'], index=[2, 3, 4])

res = pd.concat([df1, df2], axis=0, join='outer')
print("纵向外合并(默认)\n", res)
"""有相同的column上下合并在一起，其他独自的column个自成列，原本没有值的位置皆以NaN填充
      A    B    C    D    E    F
1  0.0  0.0  0.0  0.0  NaN  NaN
2  0.0  0.0  0.0  0.0  NaN  NaN
3  0.0  0.0  0.0  0.0  NaN  NaN
2  NaN  NaN  1.0  1.0  1.0  1.0
3  NaN  NaN  1.0  1.0  1.0  1.0
4  NaN  NaN  1.0  1.0  1.0  1.0
"""

res = pd.concat([df1, df2], axis=0, join='inner', ignore_index=True)
print("内合并, 重置index\n", res)
"""只有相同的column合并在一起，其他的会被抛弃
      C    D
0  0.0  0.0
1  0.0  0.0
2  0.0  0.0
3  1.0  1.0
4  1.0  1.0
5  1.0  1.0
"""

# join_axes [index对象] 依照 axes 合并
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
print("按df1.index 按列合并\n", res)
"""
      A    B    C    D    F    C    D    E
1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
"""

res = pd.concat([df1, df2], axis=1)
print("按列合并\n", res)
"""
      A    B    C    D    F    C    D    E
1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
4  NaN  NaN  NaN  NaN  1.0  1.0  1.0  1.0
"""

# append 合并 只有按行合并
#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])

res = df1.append(df2, ignore_index=True)
print("df2合并到df1下, 重置index\n", res)
"""
      a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  1.0  1.0  1.0
4  1.0  1.0  1.0  1.0
5  1.0  1.0  1.0  1.0
"""

res = df1.append([df3, df2], ignore_index=True)
print("append多个对象\n", res)
"""
      a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  2.0  2.0  2.0  2.0
4  2.0  2.0  2.0  2.0
5  2.0  2.0  2.0  2.0
6  1.0  1.0  1.0  1.0
7  1.0  1.0  1.0  1.0
8  1.0  1.0  1.0  1.0
"""
