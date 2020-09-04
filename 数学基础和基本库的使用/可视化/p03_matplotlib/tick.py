#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 15:17:44
@Description: tick 能见度
@LastEditTime: 2019-11-04 15:33:25
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y = 0.1 * x

plt.figure()
plt.plot(x, y, linewidth=10, zorder=1)  # zorder设置z轴方向排序
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# plt.show()

# 被遮挡的图像调节相关透明度
# abel.set_fontsize(12)重新调节字体大小，bbox设置目的内容的透明度相关参，facecolor调节 box 前景色，edgecolor 设置边框， 本处设置边框为无，alpha设置透明度
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_bbox(
        dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))

plt.show()
