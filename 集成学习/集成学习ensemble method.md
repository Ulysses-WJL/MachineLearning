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

## Blending

**举例**

假如你有T个朋友，每个朋友向你预测推荐明天某支股票会涨还是会跌，对应的建议分别是$g_1, g_2, \cdots, g_T$, 那么你该选择哪个朋友的建议呢？即最终选择对股票预测的$g_t(x)$是什么样的?

### validation思想
第一种方法是从T个朋友中选择一个最受信任，对股票预测能力最强的人，直接听从他的建议就好. 即通过验证集来选择最佳模型, 
$$
G(x) = g_{t^*} \\ t^* = argmin_{t \in 1, 2, ...T} E_{val}(g_t^-)
$$
第一种方法只是从众多可能的hypothesis中选择最好的模型，并不能发挥集体的智慧。而Aggregation的思想是博采众长，将可能的hypothesis优势集合起来，将集体智慧融合起来，使预测模型达到更好的效果。

### Uniformly Blending

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

### Linear and Any Blending

第三种方法，如果每个朋友水平不一，有的比较厉害，投票比重应该更大一些，有的比较差，投票比重应该更小一些。那么，仍然对T个朋友进行投票，只是每个人的投票权重不同.
$$
G(x) = sign(\sum_{t=1}^T \alpha_t \cdot g_t(x)) \\
\alpha_t \geq 0
$$

与uniform blending不同这里的每个$g_t$赋予的权重$\alpha_t$都不同, 最终得到的结果等于所有$g_t$的线性组合.利用误差最小化的思想, 可以确定$\alpha_t$的值
$$
\mathop {min}_{\alpha_t \geq 0} \frac 1 N \sum_{n=1}^N(y_n-\sum_{t=1}^T\alpha_tg_t(x_n))
$$

除了linear blending之外，还可以使用任意形式的blending。
$$
G(x) = sign(\sum_{t=1}^T q_t(x)\cdot g_t(x))\\
q_t(x) \geq 0
$$
linear blending中，G(t)是g(t)的线性组合；any blending中，G(t)可以是g(t)的任何函数形式（非线性）。这种形式的blending也叫做Stacking。any blending的优点是模型复杂度提高，更容易获得更好的预测模型；缺点是复杂模型也容易带来过拟合的危险。所以，在使用any blending的过程中要时刻注意避免过拟合发生，通过采用regularization的方法，让模型具有更好的泛化能力。

## Bagging与随机森林

### Bagging (Bootstrap Aggregating)

Bagging是并行集成学习方法最著名的代表, 直接基于自助采样法(bootstrap sampling, 亦称可重复采样法或有放回采样法).

> 给定包含$m$个样本的数据集$D$, 对它进行采样产生数据集$D'$: 每次随机从$D$中挑选一个样本, 将其拷贝后放入$D'$, 然后将该样本放回$D$, 使得该样本在下次采样时仍有机会被采到; 这个过程执行$m$次后, 就得到了包含$m$个样本的数据集$D'$. 样本在m次采样中始终不被采样到的概率是$(1- \frac 1 m)^m$, 取极限得到
> $$
> \mathop{lim}_{m \to \infty} (1-\frac 1 m)^m \rightarrow \frac 1 e \simeq 0.368
> $$
>
> 即通过自助采样, 初始数据集$D$中约有36.8%的样本未出现在数据集$D'$中.可以将$D'$用作训练集, D \ D'(集合相减)用作测试集; 实际评估的模型与期望评估的模型都使用了$m$个训练样本, 而仍有约36.8%的样本数据没有在训练集中出现,可做为测试. 这样的测试结果, 亦称"包外估计"(out-of-bag estimate)

通过自助采样法, 从包含$m$个样本的数据集中可生成$T$个含$m$个样本的训练集, 然后基于每个训练集训练出一个基学习器, 然后使用blending的方法进行组合(经常选择简单投票, Uniformly Blending). 这就是Bagging的基本流程.

假定基学习器的计算复杂度为$O(m)$, 则Bagging的复杂度大致为$T(O(m) + O(s))$, 考虑到采样与投票/平均的过程复杂度$O(s)$很小, 而$T$通常是一个不太大的常数, 因此, 训练一个Bagging集成与直接使用基算法学习训练一个学习器同阶.另外, 与标准的AdsBoost只适用于二分类任务不同, Bagging能不经修改地用于多分类, 回归等任务.

### 随机森林

随机森林(Random Forest, RF)是Bagging的一个扩展变体. RF在以决策树为基础的学习器构建Bagging集成的基础上, 进一步在决策树的训练过程中引入了`随机属性选择`. 

传统决策树在选择划分属性时是在当前结点的属性集合(假定有d个属性)中选择一个最优的属性;而在RF中, 对基决策树的每个结点, 先从该结点的属性集合中随机选择一个包含k个属性的子集, 然后再从这个子集中选择一个最优属性用于划分. 这里的参数k控制了随机性的引入程度; 若令k=d, 则与传统的决策树相同; 若令k=1, 则是随机选择一个属性用于划分; 一般情况下, 推荐$k=log_2d$

具体操作实例:[随机森林](https://github.com/apachecn/AiLearning/blob/master/docs/ml/7.%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95-%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E5%92%8CAdaBoost.md#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97-%E5%8E%9F%E7%90%86)

## AdaBoosting(Adaptive Boosting)

Boosting是一族可将弱学习器提升为强学习器的算法. 这族算法的工作机制类似: 先从初始训练集训练出一个基学习器, 再根据基学习器的表现对训练样本分布进行调整, 使得先前基学习器做错的训练样本在后续受到更多关注, 然后基于调整后的样本分布来训练下一个基学习器; 如此重复进行, 直至基学习器数目达到事先指定的值$M$, 最终将这$M$个基学习器进行加权结合. 其中最著名的代表就是AdaBoost算法.

AdaBoost算法有多种推导方式, 比较容易理解的是基于"加性模型"(additive model), 即基学习器的线性组合:
$$
f(x) = \sum_{m=1}^M \alpha_m G_m(x)
$$
 算法的具体描述:

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

     可以看到当$e_m \leq \frac 1 2$时，$\alpha_m \geq 0$, 并且$\alpha_m$随着$e_m$的减小而增大, 所以分类误差越小的基本分类器在最终分类器中的作用越大.

   - (d) 更新训练数据集的权值分布
     $$
     D_{m+1} = (w_{m+1, 1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}) \\
     w_{m+1, i} =  \frac {w_{mi}} {Z_m} exp(-\alpha_m y_i G_m(x_i)) = \begin{cases} \frac {w_{mi}} {Z_m} e^{\alpha_m} , \quad \ y_i \neq G_m(x_i) \\ \frac {w_{mi}} {Z_m} e^{-\alpha_m} , \quad \ y_i = G_m(x_i) \end{cases} 
     \\ i = 1,2, \cdots, N
     $$
     这里$Z_m$是规范化因子
     $$
     Z_m = \sum_{i=1}^N w_{mi}exp(-\alpha_my_iG_m(x_i))
     $$
     被$G_m$误分类的样本权值得以**扩大**, 正确分类样本的权值得以**缩小**. 两者相比, 误分类样本的权值被放大了$e^{2\alpha_m} = \frac {1-e_m}{e_m}$倍. 因此, 误分类样本在下一轮学习中起到更大作用.
   
3. 构建基本分类器的线性组合
   $$
   f(x) = \sum_{m=1}^M \alpha_mG_m(x)
   $$
   最终得到分类器
   $$
   G(x) = sign(f(x)) = sign(\sum_{m=1}^M \alpha_mG_m(x))
   $$
   系数$\alpha$表示基本分类器$G_m$的重要性, $f(x)$的符号决定实例$x$的类别, $f(x)$的绝对值表示分类的确性度. 


参考:

- 机器学习-周志华
- [集成方法-随机森林和AdaBoost](https://github.com/apachecn/AiLearning/blob/master/docs/ml/7.%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95-%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E5%92%8CAdaBoost.md)
- [Blending and Bagging](https://github.com/apachecn/ntu-hsuantienlin-ml/blob/master/25.md)
