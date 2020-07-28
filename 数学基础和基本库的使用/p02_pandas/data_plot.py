#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-30 13:56:31
@Description: 使用matplotlib出图
@LastEditTime: 2019-10-30 14:35:49
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 绘制线性图
# 从“标准正态”分布中返回一个或多个样本
data = pd.Series(np.random.randn(1000), index=np.arange(1000))

data_sum = data.cumsum()  # 累加

# data_sum.plot()  # 已经有x=, y= 的数据项


data = pd.DataFrame(
    np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))

data_sum = data.cumsum()
data_sum.plot()

# scatter绘制散点图
ax = data.plot.scatter(
    x='A', y='B', color='DarkBlue', label='class1(a, b)')  # (A, B)
data.plot.scatter(
    x='A', y='C', color='LightGreen', label='class2(a, c)', ax=ax)  #(A, C)
plt.show()
