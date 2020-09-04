#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-05 09:39:25
@Description: subplot 多个图像
@LastEditTime: 2019-11-05 10:50:29
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def multi_pic_1():
    plt.figure()  # 创建图像窗口

    # 将整个图像窗口分为2行2列, 当前位置为1
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1])

    plt.subplot(2, 2, 2)
    plt.plot([0, 1], [1, 0])

    plt.subplot(223)  # 可以简写成一个三位数的数字
    plt.plot([0, 1], [0, 3])

    plt.subplot(224)
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.scatter(x, y, s=50, alpha=.75, c=np.arctan2(x, y))

    plt.show()

def multi_pic_2():
    # 不均匀的图中图 subplot
    plt.figure()
    plt.subplot(2, 1, 1)  # 分为2行1列, 当前位置1
    plt.plot([0, 1], [0, 1])

    plt.subplot(2, 3, 4)  # 分为2行1列, 当前位置4 第2行的第1个位置是整个图像窗口的第4个位置.
    plt.plot([0, 1], [1, 0])

    plt.subplot(235)
    plt.plot([0, 1], [2, 1])

    plt.subplot(236)
    plt.scatter(np.random.rand(10),
        np.random.rand(10), c='red')
    plt.show()

def multi_pic_3():
    # 使用subplot2grid
    plt.figure()
    # 窗口分为3*3, 从(0, 0)开始作图,列跨度3 (默认1)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.plot([1, 2], [1, 2])
    ax1.set_title('ax1')  # 设置标题

    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax2.plot([1, 2], [3, 4])

    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax3.plot([0, 1], [0, 2])
    ax3.set_title('ax3')

    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.scatter([0, 1], [1, 3])

    ax5 = plt.subplot2grid((3, 3), (2, 1))
    # ax5.set_title('ax5')
    plt.show()

def multi_pic_4():
    # 使用gridspec.GridSpec
    plt.figure()
    gs = gridspec.GridSpec(3, 3)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1:, 2])
    ax4 = plt.subplot(gs[-1, 0])
    ax5 = plt.subplot(gs[2, -2])

    plt.show()

def multi_pic_5():
    # 使用subpolts
    # 建立一个2行2列的图像窗口 共享x y轴坐标
    f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax11.scatter([1, 2], [1, 2])
    plt.tight_layout()  # 紧凑显示图像
    plt.show()


if __name__ == "__main__":
    # multi_pic_1()
    # multi_pic_2()
    # multi_pic_3()
    # multi_pic_4()
    multi_pic_5()
    
