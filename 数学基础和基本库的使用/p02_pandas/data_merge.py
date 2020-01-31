#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-30 10:25:26
@Description: 使用merge方法合并数据
@LastEditTime: 2019-10-30 13:55:32
'''
import pandas as pd
# pandas中的merge和concat类似,但主要是用于两组有key column的数据,统一索引的数据.
#  通常也被用在Database的处理当中.

left = pd.DataFrame({
    'A': ['A0', 'A1', "A2"],
    'B': ['B0', "B1", "B2"],
    'key': ['k0', 'k1', 'k2']
})

right = pd.DataFrame({
    'C': ['C0', 'C1', "C2"],
    'B': ['B0', "B1", "B2"],
    'key': ['k0', 'k1', 'k2']
})

print(left, right, sep='\n')
"""
    A   B key
0  A0  B0  k0
1  A1  B1  k1
2  A2  B2  k2
    C   B key
0  C0  B0  k0
1  C1  B1  k1
2  C2  B2  k2
"""

# merge合并
res = pd.merge(left, right, on='key')
print("基于一组'key'进行合并\n", res)
"""
     A B_x key   C B_y
0  A0  B0  k0  C0  B0
1  A1  B1  k1  C1  B1
2  A2  B2  k2  C2  B2
"""

# 2组key合并
left = pd.DataFrame({
    'key1': ['K0', 'K0', 'K1', 'K2'],
    'key2': ['K0', 'K1', 'K0', 'K1'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})
right = pd.DataFrame({
    'key1': ['K0', 'K1', 'K1', 'K2'],
    'key2': ['K0', 'K0', 'K0', 'K0'],
    'C': ['C0', 'C1', 'C2', 'C3'],
    'D': ['D0', 'D1', 'D2', 'D3']
})
print(left, right, sep='\n')
"""
  key1 key2   A   B
0   K0   K0  A0  B0
1   K0   K1  A1  B1
2   K1   K0  A2  B2
3   K2   K1  A3  B3

  key1 key2   C   D
0   K0   K0  C0  D0
1   K1   K0  C1  D1
2   K1   K0  C2  D2
3   K2   K0  C3  D3
"""

res = pd.merge(left, right, how='inner', on=['key1', 'key2'])
print('基于2列key合并, inner方式\n', res)  # 只有有相同key才会合并到结果
"""
基于2列key合并, inner方式
   key1 key2   A   B   C   D
0   K0   K0  A0  B0  C0  D0
1   K1   K0  A2  B2  C1  D1
2   K1   K0  A2  B2  C2  D2
"""
res = pd.merge(left, right, how='outer', on=['key1', 'key2'])
print('基于2列key合并, outer方式\n', res)  # 没有对应key的数值为NaN
"""
基于2列key合并, outer方式
   key1 key2    A    B    C    D
0   K0   K0   A0   B0   C0   D0
1   K0   K1   A1   B1  NaN  NaN
2   K1   K0   A2   B2   C1   D1
3   K1   K0   A2   B2   C2   D2
4   K2   K1   A3   B3  NaN  NaN
5   K2   K0  NaN  NaN   C3   D3
"""

res = pd.merge(left, right, how='left', on=['key1', 'key2'])
print('基于2列key合并, left方式\n', res)  # 以左边的数据为基准, 右边有相同key组的数据添加到左边, 没有的为NaN
"""
基于2列key合并, left方式
   key1 key2   A   B    C    D
0   K0   K0  A0  B0   C0   D0
1   K0   K1  A1  B1  NaN  NaN
2   K1   K0  A2  B2   C1   D1
3   K1   K0  A2  B2   C2   D2
4   K2   K1  A3  B3  NaN  NaN
"""

res = pd.merge(left, right, how='right', on=['key1', 'key2'])
print('基于2列key合并, right方式\n', res)
"""
基于2列key合并, right方式
   key1 key2    A    B   C   D
0   K0   K0   A0   B0  C0  D0
1   K1   K0   A2   B2  C1  D1
2   K1   K0   A2   B2  C2  D2
3   K2   K0  NaN  NaN  C3  D3
"""

# Indicator indicator=True会将合并的记录放在新的一列
df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})

print(df1)
"""
   col1 col_left
0     0        a
1     1        b
"""
print(df2)
"""
   col1  col_right
0     1          2
1     2          2
2     2          2
"""

res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print('indicator将合并的记录放在新的一列\n', res)
"""
    col1 col_left  col_right      _merge
0     0        a        NaN   left_only
1     1        b        2.0        both
2     2      NaN        2.0  right_only
3     2      NaN        2.0  right_only
"""

res = pd.merge(df1, df2, on='col1', how='outer', indicator="indicator_column")
print('自定indicator column的名称\n', res)
"""
自定indicator column的名称
    col1 col_left  col_right indicator_column
0     0        a        NaN        left_only
1     1        b        2.0             both
2     2      NaN        2.0       right_only
3     2      NaN        2.0       right_only
"""

# 基于index 合并
left = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']},
    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({
    'C': ['C0', 'C2', 'C3'],
    'D': ['D0', 'D2', 'D3']},
    index=['K0', 'K2', 'K3'])

print(left)
"""
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2
"""

print(right)
"""
     C   D
K0  C0  D0
K2  C2  D2
K3  C3  D3
"""

res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print("根据左右数据集的index 进行合并, out方式\n", res)
"""
       A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C2   D2
K3  NaN  NaN   C3   D3
"""

res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
print("根据index合并, inner方式\n", res)
"""
      A   B   C   D
K0  A0  B0  C0  D0
K2  A2  B2  C2  D2
"""

boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

res = pd.merge(boys, girls, on='k', how='inner')
print('直接合并age重复, 带后缀\n', res)
"""
     k  age_x  age_y
0  K0      1      4
1  K0      1      5
"""

res = pd.merge(boys, girls, on='k', how='inner', suffixes=['_boys', '_girls'])
print("指定后缀名\n", res)
"""
     k  age_boys  age_girls
0  K0         1          4
1  K0         1          5
"""
