集成方法 ensemble method

---

集成学习(ensemble learning)通过构建并结合多个学习器来完成学习任务, 有时也被称为`多分类器系统(multi-classifier)`、`基于委员会的学习(committee-based learning)`.

* 概念：是对其他算法进行组合的一种形式。
* 通俗来说： 当做重要决定时，大家可能都会考虑吸取多个专家而不只是一个人的意见。
    机器学习处理问题时又何尝不是如此？ 这就是集成方法背后的思想。
* 集成方法：
    1. 投票选举(bagging: 自举汇聚法 bootstrap aggregating): 是基于数据随机重抽样分类器构造的方法
    2. 再学习(boosting): 是基于所有分类器的加权求和的方法

- aggregation的两个优势: feature transform和regularization.  feature transform和regularization是对立的, 单一模型通常只能倾向于feature transform和regularization之一，在两者之间做个权衡。但是aggregation却能将feature transform和regularization各自的优势结合起来.

## 个体与集成

集成学习的一般结构: 先产生一组"个体学习器(individual learner)", 再用某种策略将它们结合起来, 个体学习器通常由一个现有的学习算法从训练数据产生. 通过将多个学习器进行组合, 常可获得比单一学习器显著优越的泛化性能.

个体学习器的要求, "好而不同":

- 要有一定的'"准确性";
- 要有多样性(diversity), 即学习器之间具有差异.

根据个体学习器的生成方式, 目前集成学习方法大致可分为两大类, 即个体学习器之间存在强依赖关系, 必须执行串行生成的序列化方法, 以及体学习器之间不存在强依赖关系, 可同时生成的并行化方法; 前者的代表是Boosting, 后者的代表是Bagging和随机森林(Random Forest)

### 强可学习与弱可学习

在概率近似正确(probably approximately correct, PAC)学习的框架中, 一个概念(一个类), 如果存在一个多项式的学习算法能够学习它, 并且正确率很高, 就称这个概念是强可学习(strongly learnable)的; 一个概念, 如果存在一个多项式的学习算法能够学习它, 学习的正确率仅比随即猜测略好, 那么就称这个概念是弱可学习(weakly learnable)的.

对于分类问题而言, 给定一个训练样本集, 求比较粗糙的分类规则(弱分类器)比求精确的分类规则(强分类器)容易的多. 提升方法就是从弱学习算法出发, 反复学习, 得到一系列弱分类器(基本分类器), 然后组合这些弱分类器, 构成一个强分类器.

这样, 对提升方法来说, 有两个问题需要回答:

- 每一轮如何改变训练数据的权值或概率分布
- 如何将弱分类器组合成一个强分类器

## Blending 结合策略
学习器结合可能从三个方面带来好处:
- 从统计学习方面看, 由于学习任务的假设空间往往很多, 可能有多个假设在训练集上达到同等性能, 此时若使用单学习器可能因误选二导致泛化性能不佳, 结合多个学习器则会减小这一风险;
- 从计算的方面来看, 学习算法往往会陷入局部极小, 有的局部极小点所对应的泛化性能可能很糟糕, 而通过多次运行之后进行结合, 可降低陷入糟糕局部极小点的风险;
- 从表示的方面来看, 某些学习任务的真实假设可能不在当前学习算法所考虑的假设空间中, 此时若使用但学习器肯定无效, 而通过结合多个学习器, 由于相应的假设空间有所扩大, 有可能学得更好的近似.

**举例**

假如你有T个朋友，每个朋友向你预测推荐明天某支股票会涨还是会跌，对应的建议分别是$g_1, g_2, \cdots, g_T$, 那么你该选择哪个朋友的建议呢？即最终选择对股票预测的$g_t(x)$是什么样的?

### validation思想

第一种方法是从T个朋友中选择一个最受信任，对股票预测能力最强的人，直接听从他的建议就好. 即通过验证集来选择最佳模型,
$$
G(x) = g_{t^*} \\ t^* = argmin_{t \in 1, 2, ...T} E_{val}(g_t^-)
$$
第一种方法只是从众多可能的hypothesis中选择最好的模型，并不能发挥集体的智慧。而Aggregation的思想是博采众长，将可能的hypothesis优势集合起来，将集体智慧融合起来，使预测模型达到更好的效果。

### Uniformly Blending 简单平均法

如果每个朋友在股票预测方面都是比较厉害的，都有各自的专长，那么就同时考虑T个朋友的建议，将所有结果做个投票，一人一票，最终决定出对该支股票的预测。这种方法对应的是uniformly思想
$$
G(x) = sign(\sum_{t=1}^T 1 \cdot g_t(x))
$$
已有的性能较好的矩$g_t$, 将它们进行整合、合并，来得到最佳的预测模型的过程叫做blending。最常用的一种方法是uniform blending，应用于classification分类问题，做法是将每一个可能的矩赋予权重1，进行投票.
>  假设hypothesis，一个机器学习模型对应了很多不同的hypothesis，通过演算法A，选择一个最佳的hypothesis对应的函数称为矩g，g能最好地表示事物的内在规律，也是我们最终想要得到的模型表达式。

假设基分类器的错误率为$\epsilon$,
$$
P(g_t(x) \neq f(x)) = \epsilon
$$

假设基分类器的错误率**互相独立**, 由Hoeffding不等式可知, 集成的错误率为
$$
P(G(x) \neq f(x)) = \sum_{t=1}^T\begin{pmatrix}T \\ k \end{pmatrix}(1-\epsilon)^k \epsilon^{T-k} \leq exp(-\frac 1 2 T(1-2\epsilon)^2)
$$
随着集成中个体分类器数目$T$的增大, 集成的错误率将指数级下降, 最终趋于零

这种方法对应三种情况：第一种情况是每个候选的$g_t$都完全一样，这跟选其中任意一个$g_t$效果相同；第二种情况是每个候选的$g_t$都有一些差别，这是最常遇到的，大都可以通过投票的形式使多数意见修正少数意见，从而得到很好的模型；第三种情况是多分类问题，选择投票数最多的那一类即可

如果是regression回归问题，uniform blending的做法很简单，就是将所有的$g_t$ 求平均值
$$
G(x) = \frac 1 T \sum_{t=1}^Tg_t(x)
$$
uniform blending for regression对应两种情况：第一种情况是每个候选的$g_t$都完全一样, 跟任选其中一个效果相同; 第二种情况是每个候选的$g_t$ 都有一些差别, 有的$g_t > f(x)$ , 有的$g_t<f(x)$  此时求平均值的操作可能会消去这种大于和小于的影响，从而得到更好的回归模型。因此，从直觉上来说，求平均值的操作[更加稳定，更加准确](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md#uniform-blending)。

### Linear and Any Blending 加权平均法

第三种方法，如果每个朋友水平不一，有的比较厉害，投票比重应该更大一些，有的比较差，投票比重应该更小一些。那么，仍然对T个朋友进行投票，只是每个人的投票权重不同.
$$
G(x) = sign(\sum_{t=1}^T \alpha_t \cdot g_t(x)) \\
\alpha_t \geq 0
$$
> 通常对学习器的权重施以非负的约束, 实际权重为负时,[只需要将正类看成负类，负类当成正类即可](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md#linear-and-any-blending)

与uniform blending不同这里的每个$g_t$赋予的权重$\alpha_t$都不同, 最终得到的结果等于所有$g_t$的线性组合.利用误差最小化的思想, 可以确定$\alpha_t$的值
$$
\mathop {min}_{\alpha_t \geq 0} \frac 1 N \sum_{n=1}^N(y_n-\sum_{t=1}^T\alpha_tg_t(x_n))
$$

除了linear blending之外，还可以使用任意形式的blending。
$$
G(x) = sign(\sum_{t=1}^T q_t(x)\cdot g_t(x))\\
q_t(x) \geq 0
$$
linear blending中，G(t)是g(t)的线性组合；any blending中，G(t)可以是g(t)的任何函数形式（非线性）。这种形式的blending也叫做**Stacking**。any blending的优点是模型复杂度提高，更容易获得更好的预测模型；缺点是复杂模型也容易带来过拟合的危险。所以，在使用any blending的过程中要时刻注意避免过拟合发生，通过采用regularization的方法，让模型具有更好的泛化能力。

### 绝对多数投票法(majority voting)
若某标记得票过半数, 则预测为该标记; 否则拒绝预测.

### 相对多数投票(plurality voting)
预测为得票最多的标记, 若同时有多个标记获最高票, 则从中随机选取一个.

### 加权投票法(weighted voting)
与加权平均法类似

### 学习法
当训练数据很多是, 一种更为强大的结合策略是使用"学习法", 即通过另一个学习器来进行结合.
前文提到的Stacking就是学习法的典型代表.这里我们把个体学习器称为初级学习器, 用于结合的学习器称为次级学习器或元学习器(meta-learner).

Stacking先从初始数据集训练出初级学习器, 然后"生成"一个新数据集用于训练次级学习器.在这个新数据集中, 初级学习器的输出被当做样例输入特征, 二初始样本的标记仍被当做样例标记.


## Bagging与随机森林

### Bagging (Bootstrap Aggregating)

Bagging是并行集成学习方法最著名的代表, 直接基于自助采样法(bootstrap sampling, 亦称可重复采样法或有放回采样法).

> 给定包含$m$个样本的数据集$D$, 对它进行采样产生数据集$D'$: 每次随机从$D$中挑选一个样本, 将其拷贝后放入$D'$, 然后将该样本放回$D$, 使得该样本在下次采样时仍有机会被采到; 这个过程执行$m$次后, 就得到了包含$m$个样本的数据集$D'$. 样本在m次采样中始终不被采样到的概率是$(1- \frac 1 m)^m$, 取极限得到
> $$
> \mathop{lim}_{m \to \infty} (1-\frac 1 m)^m \rightarrow \frac 1 e \simeq 0.368
> $$
>
> 即通过自助采样, 初始数据集$D$中约有36.8%的样本未出现在数据集$D'$中.可以将$D'$用作训练集, D \ D'(集合相减)用作测试集; 实际评估的模型与期望评估的模型都使用了$m$个训练样本, 而仍有约36.8%的样本数据没有在训练集中出现,可做为测试. 这样的测试结果, 亦称"**包外估计**"(out-of-bag estimate, oob)

通过自助采样法, 从包含$m$个样本的数据集中可生成$T$个含$m$个样本的训练集, 然后基于每个训练集训练出一个基学习器, 然后使用blending的方法进行组合(经常选择简单投票, Uniformly Blending). 这就是Bagging的基本流程.

假定基学习器的计算复杂度为$O(m)$, 则Bagging的复杂度大致为$T(O(m) + O(s))$, 考虑到采样与投票/平均的过程复杂度$O(s)$很小, 而$T$通常是一个不太大的常数, 因此, 训练一个Bagging集成与直接使用基算法学习训练一个学习器同阶.另外, 与标准的AdsBoost只适用于二分类任务不同, Bagging能不经修改地用于多分类, 回归等任务.

特点: Bagging具有减少不同$g_t$的方差variance的特点。这是因为Bagging采用投票的形式，将所有$g_t$通过uniform方式结合起来，起到了求平均的作用，从而降低variance。

#### self-validation

使用bagging方法, 大约有三分之一的样本没有没抽到, 这些被称为OOB样本, 我们使用这些样本来评估模型的好坏.通常不需要对单个$g_t$进行验证, 因为我们更关心的是由许多$g_t$组合而成的$G$, 即使$g_t$表现不好, 只要$G$表现的足够好就行了. 方法是先看每一个样本$(x_n, y_n)$是哪些$g_t$的OOB样本, 然后计算其在这些$g_t$上的表现, 最后将所有样本的表现求平均即可.例如, $(x_N, y_N)$是基分类器$g_2, g_3, g_T$的OOB, 则计算$(x_N, y_N)$在$G_N^{-}(x)$ 的表现为
$$
G_N^-(x) = average(g_2, g_3, g_T)
$$
最后计算所有样本的平均表现
$$
E_{oob}(G) = \frac 1 N \sum_{n=1}^N err(y_n, G_n^-(x_n))
$$


### 随机森林

随机森林(Random Forest, RF)是Bagging的一个扩展变体. RF在以决策树为基础的学习器构建Bagging集成的基础上, 进一步在决策树的训练过程中引入了`随机属性选择`.

传统的Decision Tree具有增大不同的方差variance的特点。这是因为Decision Tree每次切割的方式不同，而且分支包含的样本数在逐渐减少，所以它对不同的资料D会比较敏感一些，从而不同的D会得到比较大的variance. 将Bagging与Decision Tree 结合起来, 使用前者的特点来减少后者带来的方差, 这种算法就叫做随机森林（Random Forest），它将完全长成的C&RT决策树通过bagging的形式结合起来，最终得到一个庞大的决策模型.

传统决策树在选择划分属性时是在当前结点的属性集合(假定有d个属性)中选择一个最优的属性;而在RF中, 对基决策树的每个结点, 先从该结点的属性集合中**随机选择**一个包含k个属性的子集, 然后再从这个子集中选择一个最优属性用于划分. 这里的参数k控制了随机性的引入程度; 若令k=d, 则与传统的决策树相同; 若令k=1, 则是随机选择一个属性用于划分; 一般情况下, 推荐$k=log_2d$

随机森林的构建过程大致如下：

- 从原始训练集中使用Bootstraping方法随机有放回采样选出m个样本，共进行n_tree次采样，生成n_tree个训练集
- 对于n_tree个训练集，我们分别训练n_tree个决策树模型
- 对于单个决策树模型，假设训练样本特征的个数为n，对n个样本选择a中的k个特征, 每次分裂时根据信息增益/信息增益比/基尼指数选择最好的特征进行分裂
- 每棵树都一直这样分裂下去，直到该节点的所有训练样例都属于同一类。在决策树的分裂过程中不需要剪枝
- 将生成的多棵决策树组成随机森林。对于分类问题，按多棵树分类器投票决定最终分类结果；对于回归问题，由多棵树预测值的均值决定最终预测结果


具体操作实例:[随机森林](https://github.com/apachecn/AiLearning/blob/master/docs/ml/7.%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95-%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E5%92%8CAdaBoost.md#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97-%E5%8E%9F%E7%90%86)




## AdaBoosting(Adaptive Boosting)

Boosting是一族可将弱学习器提升为强学习器的算法. 这族算法的工作机制类似: 先从初始训练集训练出一个基学习器, 再根据基学习器的表现对训练样本分布进行调整, 使得先前基学习器做错的训练样本在后续受到更多关注, 然后基于调整后的样本分布来训练下一个基学习器; 如此重复进行, 直至基学习器数目达到事先指定的值$M$, 最终将这$M$个基学习器进行加权结合. 其中最著名的代表就是AdaBoost算法.

AdaBoost算法有多种推导方式, 比较容易理解的是基于"加性模型"(additive model), 即基学习器的线性组合:
$$
f(x) = \sum_{m=1}^M \alpha_m G_m(x)
$$
###  算法的具体描述

输入: 训练数据集$T=\{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}$, 其中$x_i \in \mathcal X \subseteq\mathbf R^n$, $y_i \in \mathcal Y=\{-1, +1\}$; 弱学习算法;

输出: 最终分类器$G(x)$

1. 初始化权值分布,　假设数据集具有均匀的权值分布
   $$
   D_1 = (w_{11}, \cdots, w_{1i}, \cdots, w_{1N}), \quad w_{1i} = \frac 1 N, \quad i =1, 2, \cdots, N
   $$

2. 对$m = 1, 2, \cdots, Ｍ$, 重复以下, 生成多个基训练器

   - (a) 使用当前权值分布$D_m$的训练数据集学习，得到基本分类器
     $$
     G_m(x): \mathcal X \rightarrow \{-1, +1\}
     $$

   - (b) 计算$G_m(x)$在训练数据集上的分类误差率
     $$
     e_m = \sum_{i=1}^N P(G_m(x_i) \neq y_i) = \sum_{i=1}^N w_{mi}I(G_m(x_i) \neq y_i) \\
     = \sum_{G_m(x_i) \neq y_i}^N w_{mi}
     $$
     分类误差率就是被误分类样本的权值和

   - (c) 计算$G_m(x)$的系数
     $$
     \alpha_m = \frac 1 2 log \frac {1-e_m}{e_m}
     $$
     log是自然对数.

     可以看到当$e_m \leq \frac 1 2$时，$\alpha_m \geq 0$, 并且$\alpha_m$随着$e_m$的减小而增大;  反之, 当$e_m \geq \frac 1 2$时,$\alpha_m \leq 0$, 并且$\alpha_m$随着$e_m$的增大而减小, 即基分类器$G_m$起的负作用也就越强.所以分类误差越小的基本分类器在最终分类器中的作用越大.

   - (d) 更新训练数据集的权值分布(re-weighting)
     $$
     D_{m+1} = (w_{m+1, 1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}) \\
     w_{m+1, i} =  \frac {w_{mi}} {Z_m} exp(-\alpha_m y_i G_m(x_i)) = \begin{cases} \frac {w_{mi}} {Z_m} e^{\alpha_m} , \quad \ y_i \neq G_m(x_i) \\ \frac {w_{mi}} {Z_m} e^{-\alpha_m} , \quad \ y_i = G_m(x_i) \end{cases}
     \\ i = 1,2, \cdots, N
     $$
     这里$Z_m$是规范化因子
     $$
     Z_m = \sum_{i=1}^N w_{mi}exp(-\alpha_my_iG_m(x_i))
     $$
     被$G_m$误分类的样本权值得以**扩大**, 正确分类样本的权值得以**缩小**. 两者相比, 误分类样本的权值被放大了$e^{2\alpha_m} = \frac {1-e_m}{e_m}$倍, 这个比值是分类器分类正确的**几率**($odds$). 因此, 误分类样本在下一轮学习中起到更大作用.

3. 构建基本分类器的线性组合
   $$
   f(x) = \sum_{m=1}^M \alpha_mG_m(x)
   $$
   最终得到分类器
   $$
   G(x) = sign(f(x)) = sign(\sum_{m=1}^M \alpha_mG_m(x))
   $$
   系数$\alpha$表示基本分类器$G_m$的重要性, $f(x)$的符号决定实例$x$的类别, $f(x)$的绝对值表示分类的确信度.

AdaBoost算法的特点是通过迭代每次学习一个基本分类器。每次迭代中，提高那些被前一轮分类器错误分类数据的权值，而降低那些被正确分类的数据的权值。最后，AdaBoost将基本分类器的线性组合作为强分类器，其中给分类误差率小的基本分类器以大的权值，给分类误差率大的基本分类器以小的权值。

### 训练误差分析

AdaBoost算法的训练误差界
$$
\frac 1 N \sum_{i=1}^NI(G(x_i) \neq y_i) \leq \frac 1 N \sum_i exp(-y_if(x_i))
$$

当$x_i$分类错误时, 右边$exp(-y_i f(x_i) ) \geq 1$, 而当分类正确时, $exp(-y_i f(x_i) ) \geq 0$可以直接推导不等式成立.

由(15)和(16)式得到
$$
W_{mi}exp(-\alpha_my_iG_m(x_i)) = Z_mw_{m+1, i}
$$
那么不等式右边可以写成
$$
\frac 1 N \sum_i exp(-y_if(x_i))  =  \frac 1 N \sum_i exp(-\sum_{m=1}^M\alpha_my_iG_m(x_i)) \\
= \sum_i w_{1i}\prod_{m=1}^Mexp(-\alpha_my_iG_m(x_i)) \\
= Z_1 \sum_i w_{2i}\prod_{m=2}^Mexp(-\alpha_my_iG_m(x_i)) \\
= Z_1Z_2 \sum_i w_{3i}\prod_{m=3}^Mexp(-\alpha_my_iG_m(x_i)) \\
= \cdots \\
= Z_1Z_2\cdots Z_{M-1} \sum_i w_{Mi}exp(-\alpha_my_iG_m(x_i)) \\
= \prod_{m=1}^M Z_m
$$
每次迭代时选择合适的$G_m$使得$Z_m$最小, 可以减少它在训练数据集上的分类误差率.

当问题使二类分类问题时, 由(13)和(14)式, 得知
$$
Z_m = \sum_{i=1}^Nw_{mi}exp(-\alpha_m y_iG_m(x_i)) \\
= \sum_{y_i \neq G_m(x_i)} w_{mi}e^{\alpha_m} + \sum_{y_i=G_m(x_i)} w_{mi}e^{-\alpha_m} \\
=e_m\sqrt{\frac {1-e_m}{e_m}} + (1-e_m)\sqrt{\frac {e_m}{1-e_m}} \\
=2\sqrt{e_m(1-e_m)}
$$
令$\gamma_{m} = \frac 1 2 -e_m$ ,  则$Z_m = \sqrt{1-4\gamma_m^2}$.  根据$e^x$和$\sqrt{1-x}$在$x=0$处地泰勒展开推导出
$$
\prod_{m=1}^M Z_m= \prod_{m=1}^M \sqrt{1-4\gamma_m^2}\leq exp(-2\sum_{m=1}^N\gamma_m^2)
$$

> Taylor公式
> $$
> T_n(x) = f(x_0) + f'(x_0)(x-x_0)+ \frac {f''(x_0)} {2!}(x-x_0)^n+ \cdots + \frac {f^{(n)(x_0)}}{n!}(x-x_0)^n
> $$
>

由(23)式, 如果存在$\gamma > 0$, 对所有的$m$使得$\gamma_m \geq \gamma$, 则训练误差
$$
\frac 1 N \sum_{i=1}^NI(G(x_i) \neq y_i) \leq exp(-2M \gamma)
$$
会以指数级数下降, 而且在这里也不需要求出$\gamma_m$的下界.

### AdaBoost算法的解释

AdaBoost算法的一个解释是该算法实际是**前向分步算法**(forward stagewise algorithm)的一个实现。

**加法模型**(additive model)
$$
f(x) = \sum_{m=1}^M \beta_m b_m(x; \gamma_m)
$$
其中$b_m(x; \gamma_m)$是**基函数**, $\gamma_m$为基函数的参数, $\beta_m$为基函数的系数.

在前向分步算法中，模型是**加法模型**，损失函数是**指数损失**，算法是前向分步算法。每一步中极小化损失函数
$$
\left(\beta_{m}, \gamma_{m}\right)=\arg \min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)
$$
得 到 参 数$\beta_{m}, \gamma_{m}$, 更新$f_m(x) = f_{m-1}(x) + \beta_mb(x; \gamma_m)$, 最后线性组合得到加法模型.

### 提升树Boosting Tree

提升树是以分类树或回归树为基本分类器的提升方法。提升树被认为是统计学习中最有效的方法之一。实际采用加法模型与前向分布算法, 以决策树为基函数, 可以看作是决策树的加法模型.
$$
f_M(x) = \sum_{m=1}^MT(x;\Theta_m)
$$
其中, $T(x; \Theta_m)$表示决策树, $\Theta_m$为决策树的参数, $M$为树的个数.

#### 提升树算法

针对不同问题的提升树算法的主要区别在于使用的损失函数不同.

对于二类分类问题, 提升树算法只需要前面的AdaBoost算法中的基分类器限制为二类分类树.

对于回归问题, 使用**平方误差**(回归树使用的就是平方误差)损失函数, 步骤:
1. 初始化$f_0(x)=0$
1. 对$m=1,2,\dots,M$
   1. 计算残差
   $$
   r_{mi}=y_i-f_{m-1}(x_i), i=1,2,\dots,N
   $$
   2. **拟合残差**$r_{mi}$学习一个回归树，得到$T(x;\Theta_m)$
   3. 更新$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$
1. 得到回归问题提升树
   $$
   f(x)=f_M(x)=\sum_{m=1}^MT(x;\Theta
   $$



#### 梯度提升法Gradient Boosted Decision Tree

当损失函数是指数函数或平方函数时, 每一步的优化是很简单的.但对于一般函数而言, 往往每一步的优化不那么容易. 梯度提升法(GBDT)利用最速下降法的近似方法, 利用损失函数的负梯度在当前模型的值
$$
-[\frac {\partial L(y, f(x_i))}{\partial f(x_i)}]_{f(x) = f_{m-1}(x)}
$$
作为回归问题提升树算法中的残差的近似值, 拟合一个回归树.

输入： 训练数据集$T={(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)}, x_i \sube \R^n , y_i\sube \R$；损失函数$L(y,f(x))$
输出：回归树$\hat{f}(x)$
步骤：
, x_i \in \cal x \sube \R^n, y_i \in \cal y \sube \R

1. 初始化
   $$
   f_0(x)=\arg\min\limits_c\sum_{i=1}^NL(y_i, c)
   $$

2. $m=1,2,\dots,M$



   - (a) 对$i=1,2,\dots,N$, 计算

     $$
     r_{mi}=-\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x)=f_{m-1}(x)}
     $$

   - (b) 对$r_{mi}$拟合一个回归树，得到第$m$棵树的叶节点区域$R_{mj}, j=1,2,\dots,J$

   - (c). 对$j=1,2,\dots,J$, 计算
     $$
     c_{mj}=\arg\min_c\sum_{xi\in R_{mj}}L(y_i,f_{m-1}(x_i)+c)
     $$

   - (d) 更新
     $$
     f_m(x)=f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
     $$



3. 得到回归树
   $$
   \hat{f}(x)=f_M(x)=\sum_{m=1}^M\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
   $$










参考:

- 机器学习-周志华
- 统计学习方法-李航
- [集成方法-随机森林和AdaBoost](https://github.com/apachecn/AiLearning/blob/master/docs/ml/7.%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95-%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E5%92%8CAdaBoost.md)
- [Blending and Bagging](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md)
- [Random Forest](<https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/28.md>)
