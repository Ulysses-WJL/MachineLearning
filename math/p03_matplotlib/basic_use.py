#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-31 17:04:36
@Description: matplotlib 的使用
@LastEditTime: 2019-11-01 16:50:16
'''
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
y1 = x + 1
y2 = np.sin(x) + 1
plt.figure(num=3, figsize=(8, 5))  # 定义一个图像窗口

# 同一个坐标轴2个图像
plt.plot(x, y1)
# plt.plot(x, y2, color='red', linestyle='--', linewidth=1.0)
plt.plot(x, y2, ':r')  # 虚线 红色
# 设置坐标轴范围
plt.xlim((-np.pi, np.pi))
plt.ylim((-2.5, 4.5))

# 设置坐标轴名称
plt.xlabel('x')
plt.ylabel('y')

# 设置刻度
plt.xticks(np.linspace(-2.5, 4.5, 8))

plt.show()

"""
格式化字符显示离散值
'-' 实线样式
'--' 短横线样式
'-.' 点划线样式
':' 虚线样式
'.' 点标记
',' 像素标记
'o' 圆标记
'v' 倒三角标记
'^' 正三角标记
'&lt;' 左三角标记
'&gt;' 右三角标记
'1' 下箭头标记
'2' 上箭头标记
'3' 左箭头标记
'4' 右箭头标记
's' 正方形标记
's' 正方形标记
'p' 五边形标记
'*' 星形标记
'h' 六边形标记 1
'H' 六边形标记 2
'+' 加号标记
'x' X 标记
'D' 菱形标记
'd' 窄菱形标记
'&#124;' 竖直线标记
'_' 水平线标记
"""

"""
颜色缩写
'b' 蓝色
'g' 绿色
'r' 红色
'c' 青色
'm' 品红色
'y' 黄色
'k' 黑色
'w' 白色
"""
