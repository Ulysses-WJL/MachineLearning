#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-07 11:17:17
@Description: 决策树可视化
@LastEditTime: 2019-11-07 19:10:21
'''
import matplotlib.pyplot as plt


# 定义 文本框 和 箭头格式
# sawtooth 波浪方框, round4 矩形方框, fc表示字体颜色的深浅 0.1~0.9 依次变浅
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow = dict(arrowstyle='<-')  # 由要注释的位置 指向注释文本的位置 父->子


class PlotTree:
    def __init__(self, tree):
        self._tree = tree
        fig = plt.figure(1, facecolor='w')
        fig.clf()
        self._axes = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[])
        self.width = self._get_leafs(self._tree)
        self.height = self._get_depth(self._tree)
        self.xoff = -0.5 / self.width  # 父结点和子结点 x轴相隔 半个结点的长度
        self.yoff = 1

    def plot(self):
        self._plot(self._tree, (0.5, 1.0), '')
        plt.show()

    def _plot(self, tree, parent_xy, node_text):
        # 第一个子结点 中心位置
        leafs_num = self._get_leafs(tree)
        # 找出第1个中心点的位置，然后与 parentPt定点进行划线
        node_xy = (self.xoff + (leafs_num + 1) / 2 / self.width, self.yoff)
        self._plot_text(node_xy, parent_xy, node_text)

        first_label = list(tree.keys())[0]
        # 可视化node
        self._plot_node(first_label, node_xy, decision_node, parent_xy)
        child_nodes = tree[first_label]
        # y值 = 最高点-层数的高度[第二个节点位置] 到下一层去
        self.yoff -=  1 / self.height
        for key in child_nodes:
            if type(child_nodes[key]) is dict:
                # 递归调用可视化子结点
                self._plot(child_nodes[key], node_xy, str(key))
            else:
                # 可视化叶结点
                self.xoff += 1 / self.width
                self._plot_node(child_nodes[key], (self.xoff, self.yoff), leaf_node, node_xy)
                self._plot_text((self.xoff, self.yoff), node_xy, str(key))
        # 回到上一层
        self.yoff += 1 / self.height

    def _get_leafs(self, tree):
        leafs_num = 0
        root_label = list(tree.keys())[0]
        child_nodes = tree[root_label]
        for key in child_nodes:
            leafs_num += self._get_leafs(child_nodes[key]) if type(child_nodes[key]) is dict else 1
        return leafs_num

    def _get_depth(self, tree):
        max_depth = 0
        root_label = list(tree.keys())[0]
        child_nodes = tree[root_label]
        for key in child_nodes:
            this_depth = 1 + self._get_depth(child_nodes[key]) if type(child_nodes[key]) is dict else 1
            max_depth = max(max_depth, this_depth)
        return max_depth

    def _plot_node(self, node_txt, node_xy, node_type, parent_node_xy):
        """
        使用plt.annotate 画出节点
        Args:
            node_txt ([type]): 结点的文本 (特征或标签)
            node_xy ([type]):  结点的文本所在位置(相对父节点)
            node_type ([type]): 结点类型 对应分支结点或叶结点
            parent_node_xy ([type]): [description]
        """
        self._axes.annotate(
            node_txt,
            xy=parent_node_xy,  # 注释的位置也就是父结点 (x, y)
            xytext=node_xy,  # 注释文本所在位置 即子结点位置
            xycoords='axes fraction',  # 参数xy使用的坐标系 左下角的轴分数 如(1, 1)代表右上角
            textcoords='axes fraction',  # 注释文本位置使用的坐标系 与xycoord相同
            arrowprops=arrow,  # 箭头设置
            bbox=node_type,  # 注释文本的外边框设置
            ha='center',  # horizontalalignment
            va='center'  # verticalalignment
        )

    def _plot_text(self, node_xy, parent_xy, text):
        """
        在父结点和子结点间的连线中央位置上 画上子结点关于父结点特征的值
        Args:
            node_xy ([type]): 子结点位置
            parent_xy ([type]): 父结点位置
            text ([type]): 分类特征的取值
        """
        x, y = tuple(map(lambda x: (x[0] + x[1])/2, zip(node_xy, parent_xy)))
        self._axes.text(
            x,
            y,
            text,
            c='r',
            va='center',
            ha='center',
            rotation=30  # 文本旋转的角度(°)
        )


if __name__ == "__main__":
    d = {'tearRate': {'normal': {'astigmatic': {'no': {'age': {'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}, 'pre': 'soft', 'young': 'soft'}}, 'yes': {'prescript': {'hyper': {'age': {'presbyopic': 'no lenses', 'pre': 'no lenses', 'young': 'hard'}}, 'myope': 'hard'}}}}, 'reduced': 'no lenses'}}

    pt = PlotTree(d)
    pt.plot()

