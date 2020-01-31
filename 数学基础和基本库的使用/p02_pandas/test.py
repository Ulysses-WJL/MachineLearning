#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-05 19:13:57
@Description: 测试
@LastEditTime: 2019-11-05 19:43:11
'''
import pandas as pd
import numpy as np
from collections import defaultdict

d1 = defaultdict(list)
d1['a'] = list(range(4))
d1['b'] = [1, 1, 2, 2]
d1['c'] = list(range(4))
d1['d'] = [1, 1, 2, 2]

df1 = pd.DataFrame(data=d1)
print(df1)
print(df1.loc[3,'c'])
