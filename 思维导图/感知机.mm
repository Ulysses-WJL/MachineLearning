<?xml version="1.0" encoding="UTF-8" standalone="no"?><map version="0.8.1"><node CREATED="1577438859367" ID="4kjv9chrbvs16i1i5q0oi378d4" MODIFIED="1577438859367" TEXT="感知机Perceptron"><node CREATED="1577438859367" ID="5b77229448gr51idfk5k1f72rj" MODIFIED="1577438859367" POSITION="right" TEXT="适用范围:数据集的线性可分性"/><node CREATED="1577438859367" ID="5hh10pp0e4p2pqphcjpn4c6l0a" MODIFIED="1577438859367" POSITION="right" TEXT="策略"><node CREATED="1577438859367" ID="4r0mg5nd36c0inb3rh9k0hes96" MODIFIED="1577438859367" TEXT="极小化损失函数（误分类点到超平面的总距离最小）"/></node><node CREATED="1577438859367" ID="3aruejib5pqcnhbnv859vvc06l" MODIFIED="1577438859367" POSITION="right" TEXT="模型"><node CREATED="1577438859367" ID="5icangcpr91j90ubi55q1gotth" MODIFIED="1577438859367" TEXT=":输入实例的特征向量x对其进行二类分类的线性分类模型: f(x) = sign(wx+b)"/></node><node CREATED="1577438859367" ID="4931bmheuvdttpg6c5f093hmln" MODIFIED="1577438859367" POSITION="left" TEXT="算法"><node CREATED="1577438859367" ID="2jorv3a07adhefsqjjmm1m6dos" MODIFIED="1577438859367" TEXT="原始问题:使用随机梯度下降法遇到误分类的样本点, 进行梯度下降修改w和b; "/><node CREATED="1577438859367" ID="6e05cit894van7e52e4m4v6hc1" MODIFIED="1577438859367" TEXT="对偶问题: 每遇到一次误分类点, w和b都会修改一次, 现在计算每个误分类点修改w和b的次数,.最后累积计算出w和b"/></node></node></map>
