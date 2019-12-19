# 集成方法 ensemble method
---

* 概念：是对其他算法进行组合的一种形式。
* 通俗来说： 当做重要决定时，大家可能都会考虑吸取多个专家而不只是一个人的意见。
    机器学习处理问题时又何尝不是如此？ 这就是集成方法背后的思想。

* 集成方法：
    1. blending: 将已存在的所有的模型结合起来
    2. 投票选举(bagging: 自举汇聚法 bootstrap aggregating): 是基于数据随机重抽样分类器构造的方法
    3. 再学习(boosting): 是基于所有分类器的加权求和的方法

- aggregation的两个优势: feature transform和regularization.  feature transform和regularization是对立的, 单一模型通常只能倾向于feature transform和regularization之一，在两者之间做个权衡。但是aggregation却能将feature transform和regularization各自的优势结合起来. 

## Blending

**举例**

假如你有T个朋友，每个朋友向你预测推荐明天某支股票会涨还是会跌，对应的建议分别是$g_1, g_2, \cdots, g_T$, 那么你该选择哪个朋友的建议呢？即最终选择对股票预测的$g_t(x)$是什么样的?

### validation思想
第一种方法是从T个朋友中选择一个最受信任，对股票预测能力最强的人，直接听从他的建议就好. 即通过验证集来选择最佳模型, 
$$
G(x) = g_{t^*} \\ t^* = argmin_{t \in 1, 2, ...T} E_{val}(g_t^-)
$$
第一种方法只是从众多可能的hypothesis中选择最好的模型，并不能发挥集体的智慧。而Aggregation的思想是博采众长，将可能的hypothesis优势集合起来，将集体智慧融合起来，使预测模型达到更好的效果。

### Uniform Blending

如果每个朋友在股票预测方面都是比较厉害的，都有各自的专长，那么就同时考虑T个朋友的建议，将所有结果做个投票，一人一票，最终决定出对该支股票的预测。这种方法对应的是uniformly思想
$$
G(x) = sign(\sum_{t=1}^T 1 \cdot g_t(x))
$$
已有的性能较好的模型$g_t$, 将它们进行整合、合并，来得到最佳的预测模型的过程叫做blending。最常用的一种方法是uniform blending，应用于classification分类问题，做法是将每一个可能的矩赋予权重1，进行投票.

这种方法对应三种情况：第一种情况是每个候选的$g_t$都完全一样，这跟选其中任意一个$g_t$效果相同；第二种情况是每个候选的$g_t$都有一些差别，这是最常遇到的，大都可以通过投票的形式使多数意见修正少数意见，从而得到很好的模型；第三种情况是多分类问题，选择投票数最多的那一类即可

如果是regression回归问题，uniform blending的做法很简单，就是将所有的$g_t$ 求平均值
$$
G(x) = \frac 1 T \sum_{t=1}^Tg_t(x)
$$
uniform blending for regression对应两种情况：第一种情况是每个候选的$g_t$都完全一样, 跟任选其中一个效果相同; 第二种情况是每个候选的$g_t$ 都有一些差别, 有的$g_t > f(x)$ , 有的$g_t<f(x)$  此时求平均值的操作可能会消去这种大于和小于的影响，从而得到更好的回归模型。因此，从直觉上来说，求平均值的操作[更加稳定，更加准确](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md#uniform-blending)。

参考:

- [集成方法-随机森林和AdaBoost](https://github.com/apachecn/AiLearning/blob/master/docs/ml/7.%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95-%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E5%92%8CAdaBoost.md)
- [Blending and Bagging](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md)