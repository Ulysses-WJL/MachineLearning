#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-29 15:46:55
@Description: DataFrame数据设置
@LastEditTime: 2019-10-29 16:31:27
'''
import numpy as np
import pandas as pd

dates = pd.date_range('20190101', periods=6)
df = pd.DataFrame(
    np.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])

print(df)
"""
             A   B   C   D
2019-01-01   0   1   2   3
2019-01-02   4   5   6   7
2019-01-03   8   9  10  11
2019-01-04  12  13  14  15
2019-01-05  16  17  18  19
2019-01-06  20  21  22  23
"""

# loc 或 iloc 选定位置再设置
df.iloc[3, 3] = 111
df.loc['2019-01-06','A'] = 222
print(df)
"""
              A   B   C    D
2019-01-01    0   1   2    3
2019-01-02    4   5   6    7
2019-01-03    8   9  10   11
2019-01-04   12  13  14  111
2019-01-05   16  17  18   19
2019-01-06  222  21  22   23
"""

# 根据条件设置
df.C[df.A<10] = 0
print(df)
"""
              A   B   C    D
2019-01-01    0   1   0    3
2019-01-02    4   5   0    7
2019-01-03    8   9   0   11
2019-01-04   12  13  14  111
2019-01-05   16  17  18   19
2019-01-06  222  21  22   23
"""

# 按行或列 设置
df['F'] = np.NaN  # F列不存在时, 使用['F']而不是 .F
print(df)
"""新添加一列F
              A   B   C    D   F
2019-01-01    0   1   0    3 NaN
2019-01-02    4   5   0    7 NaN
2019-01-03    8   9   0   11 NaN
2019-01-04   12  13  14  111 NaN
2019-01-05   16  17  18   19 NaN
2019-01-06  222  21  22   23 NaN
"""
df.A = 0
print("设置A列\n", df)

# 添加Series
df['E'] = pd.Series(list(range(6)), index=dates)
print("添加Series\n", df)
"""
             A   B   C    D   F  E
2019-01-01  0   1   0    3 NaN  0
2019-01-02  0   5   0    7 NaN  1
2019-01-03  0   9   0   11 NaN  2
2019-01-04  0  13  14  111 NaN  3
2019-01-05  0  17  18   19 NaN  4
2019-01-06  0  21  22   23 NaN  5
"""
