#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-29 17:16:35
@Description: 数据的导入和导出
@LastEditTime: 2019-10-30 09:10:21
'''
import csv
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

with open(
    os.path.join(current_dir, 'data.csv'), 'w', encoding='utf-8') as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(['index', 'age'])
    for i in range(5):
        writer.writerow([f'{i}', f'{i+10}'])

# 从csv文件获取
data = pd.read_csv(os.path.join(current_dir, 'data.csv'))
print(data, type(data))

# 保存到各种形式
data.to_pickle(os.path.join(current_dir, 'data.pickle'))
data.to_json(os.path.join(current_dir, 'data.json'))
data.to_excel(os.path.join(current_dir, 'data.xlsx'))
data_1 = pd.read_json(os.path.join(current_dir, 'data.json'))
print(data_1)
