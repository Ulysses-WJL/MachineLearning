#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-01 16:58:29
@Description: 坐标轴操作 图例
@LastEditTime: 2019-11-04 14:27:24
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x**2

plt.figure()
l1, = plt.plot(x, y1, label='linear line')  # plt.plot()返回的是列表
l2, = plt.plot(x, y2, ':g', label='square line')
plt.xlim((-1, 2))
plt.ylim((-2, 3))

# plt.show()
# 设定x, y轴 刻度
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
# 设置y轴刻度以及名称 空格需要使用 '\ '
plt.yticks([-2, 1.8, -1, 1.22, 3],
           [r'$really\ bad$', '$bad$', '$normal$', '$good$', r'$really\ good$'])

ax = plt.gca()  # 获取当前坐标轴信息
# 使用.spines设置边框  set_color 边框颜色  默认白色
ax.spines['right'].set_color('none')  # 右边框
ax.spines['top'].set_color('none')  # 上边框
# plt.show()


# 调整坐标轴
# 设置x轴刻度数字或名称的位置  top，bottom，both，default，none
ax.xaxis.set_ticks_position('bottom')  # 在x轴底部
# 边框x轴设置边框在y=0位置  outward，axes，data
ax.spines['bottom'].set_position(('data', 0))
# y轴的刻度名称显示在其左侧
ax.yaxis.set_ticks_position('left')
# 设置y轴上边框在x=0的位置
ax.spines['left'].set_position(('data', 0))
# plt.show()


# 设置图例说明legend
# legend将要显示的信息来自于上面代码中的 label.
# plt.legend(loc='upper right')  # 添加到右上角

# 单独修改label信息
plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')  # 自动分配最佳位置
plt.show()
