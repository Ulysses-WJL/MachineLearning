#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 14:29:30
@Description: 标注
@LastEditTime: 2019-11-04 15:14:44
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y = 2 * x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)

# plt.show()
# 移动坐标
ax = plt.gca()
# 右边和上边的边框设置为白色
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置坐标刻度数字
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置坐标轴 基于数值0位置
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# plt.show()
# 标注出点(x0, y0)
x0 = 1
y0 = 2 * x0 + 1
# 画一条垂直x轴经过(x0, y0)的虚线
plt.plot([x0, x0], [0, y0], '--k', linewidth=2.5)
# 标注 (x0, y0)点
plt.scatter([
    x0,
], [
    y0,
], s=50, color='b')
# plt.show()

# 对其添加注释 xycoords='data'基于数据的值来选位置
# xytext=(+30, -30) 和 textcoords='offset points' 对于标注位置的描述 和 xy 偏差值
# arrowprops是对图中箭头类型的一些设置
plt.annotate(
    f'$2x+1={y0}$',
    xy=(x0, y0),
    xycoords='data',
    xytext=(+30, -30),
    textcoords='offset points',
    fontsize=16,
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

# 添加注释text
plt.text(
    -3.5,
    3,
    r"$This\ is\ the\ simple\ text.\mu\ \sigma_i\ \alpha_t$",
    fontdict={
        'size': 16,
        'color': 'r'
    })

plt.show()
