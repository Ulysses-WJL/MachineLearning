#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-04 15:38:46
@Description: 散点图, 柱状图
@LastEditTime: 2019-11-24 12:40:08
'''
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter():
    n = 1024
    # 标注正态分布
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)

    T = np.arctan2(Y, X)  # 点的颜色
    print(T[500])
    plt.scatter(X, Y, s=50, c=T, alpha=0.7)
    plt.xlim((-3.5, 3.5))
    plt.xticks(())  # 隐藏xticks
    plt.ylim((-3.5, 3.5))
    plt.yticks(())  # 隐藏yticks

    plt.show()

def plot_bar():
    n = 12
    X = np.arange(n)
    uniform = np.random.uniform(0.5, 1.0, n)
    print(X, uniform)
    Y1 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)  # 均匀分布
    Y2 = (1-X/float(n)) * np.random.uniform(0.5, 1.0, n)  # 均匀分布
    # 设置主题颜色和边框颜色
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    plt.xlim(-2, n)
    plt.ylim(-1.25, 1.25)
    plt.xticks(())
    plt.yticks(())

    for x, y1 in zip(X, Y1):
        # ha: horizontal alignment
        # va: vertical alignment
        # 横向居中对齐，纵向底部（顶部）对齐
        plt.text(x+0.2, y1+0.05, f"{y1:.2}", ha='center', va='bottom')

    for x, y2 in zip(X, Y2):
        plt.text(x+0.2, -y2-0.05, f"{y2:.2}", ha='center', va='top')
    plt.show()

# 等高线图
def plot_contours():
    def f(x, y):
        # 生成各点的高度
        # return x**2 + y**2
        return x**2 * y
        # return (1 - x / 2 + x**5 + y**3)* np.exp(-x**2-y**2)

    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    # 由坐标向量生成坐标矩阵 二维平面中将每一个x和每一个y分别对应起来，编织成栅格
    """
    a=[1 2 3], b= [2 3 4] ->  x, y=meshgrid(a,b)
    x =
        1     2     3
        1     2     3
        1     2     3
    y =
        2     2     2
        3     3     3
        4     4     4
    """
    X, Y = np.meshgrid(x, y)
    # 颜色填充 将 f(X,Y) 的值对应到color的暖色组中, 8决定等高线的密集程度
    plt.contourf(X, Y, f(X, Y), 10, alpha=.75, cmap=plt.cm.hot)
    # 绘制等高线
    # len(X)==M, len(Y)==N Z -> (N, M)
    C = plt.contour(X, Y, f(X, Y), 10, colors='black')
    # 添加高度数字  inline 数字画在线里
    plt.clabel(C, inline=True, fontsize=10)  # Label a contour plot.
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_histogram():
    # 绘制直方图
    a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
    # 数据的频率分布的图形表示。水平尺寸相等的矩形对应于类间隔，称为bin，变量height对应于频率
    hist, bins = np.histogram(a, bins=[0,20,40,60,80,100])
    print(hist)  # [3, 4, 5, 2, 1] 各个区间段数字的次数
    print(bins)  # [0,20,40,60,80,100]

    plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
    plt.title("histogram")
    plt.show()

if __name__ == '__main__':
    # plt_scatter()
    # plot_bar()
    plot_contours()
    # plot_histogram()
