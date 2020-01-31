#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-05 11:02:18
@Description: 打印出图片, 3d图片
@LastEditTime: 2019-11-05 13:54:06
'''
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_image():
    # 在2维平面上画图
    plt.figure()
    a = np.linspace(0.32, 0.64, 9).reshape(3, 3)
    # a (M, N) an image with scalar data. The data is visualized using a colormap
    # origin='lower'代表的就是选择的原点的位置
    # 出图方式选择 nearest
    # cmap 'bone'字符串 == plt.bone()
    plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
    plt.xticks(())
    plt.yticks(())
    # 添加colorbar 使colorbar的长度变短为原来的92%
    plt.colorbar(shrink=.92)
    plt.show()


def plot_3d(file_path):
    fig = plt.figure()
    # 在窗口上添加3D坐标轴
    #ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    # plt.show()
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)  # 在z轴上是正弦曲线
    # 设置z轴刻度最大值
    ax.set_zlim3d(-2, 2)

    # 3-d表面图
    # rstride 和 cstride 分别代表 row 和 column 的跨度
    # 颜色  colormap rainbow
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # 设置投影 (等高线)
    # offset 如果指定，则在垂直于zdir的平面上的此位置(z=offset平面)绘制轮廓线的投影
    ax.contourf(X, Y, Z, zidr='z', offset=-2, cmap=plt.get_cmap('rainbow'))
    # 调整角度 仰角 方位角 (°)
    ax.view_init(elev=10, azim=125)
    plt.savefig(file_path)
    plt.show()

if __name__ == "__main__":
    # plot_image()
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '3d')
    plot_3d(file_path)
