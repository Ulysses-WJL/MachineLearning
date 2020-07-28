#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-29 16:22:44
@Description: 处理丢失数据
@LastEditTime: 2019-10-29 17:15:21
'''
import numpy as np
import pandas as pd


dates = pd.date_range('20190101', periods=4)
df = pd.DataFrame(
    np.arange(12).reshape(4, 3), index=dates, columns=['A', 'B', 'C']
)
print(df)
"""
            A   B   C
2019-01-01  0   1   2
2019-01-02  3   4   5
2019-01-03  6   7   8
2019-01-04  9  10  11
"""
df.iloc[1, 1] = np.nan
df.loc['2019-01-01', 'C'] = np.nan
print("设置nan\n", df)
"""
            A     B     C
2019-01-01  0   1.0   NaN
2019-01-02  3   NaN   5.0
2019-01-03  6   7.0   8.0
2019-01-04  9  10.0  11.0
"""

# 0 对行index; 1 对列columns
# any: 有一个NaN去除整列/行; all: 全部为NaN
print("去除nan\n", df.dropna(axis='columns', how='any'))
"""
             A
2019-01-01  0
2019-01-02  3
2019-01-03  6
2019-01-04  9
"""

print("替代nan\n", df.fillna(value=1))
"""
替代nan
             A     B     C
2019-01-01  0   1.0   1.0
2019-01-02  3   1.0   5.0
2019-01-03  6   7.0   8.0
2019-01-04  9  10.0  11.0
"""

print("判断是否有数据为nan\n", df.isnull())
"""
判断是否有数据为nan
                 A      B      C
2019-01-01  False  False   True
2019-01-02  False   True  False
2019-01-03  False  False  False
2019-01-04  False  False  False
"""

print("检测数据中是否存在nan\n", np.any(df.isnull()))
# 检测数据中是否存在nan
# True
