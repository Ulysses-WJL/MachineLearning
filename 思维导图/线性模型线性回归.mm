
<map version="0.9.0">
    <node CREATED="0" ID="4a081ce03cc94045b2a2e44a6f15189b" MODIFIED="0" TEXT="线性模型/线性回归">
        <node CREATED="0" ID="c6ecc44ade1a4f8993183860f743a6b3" MODIFIED="0" POSITION="right" TEXT="​线性模型的例子">
            <node CREATED="0" ID="fd82525aea3e421c99f186a559ffe226" MODIFIED="0" POSITION="right" TEXT="​预测房价"/>
            <node CREATED="0" ID="40340bfbde494b288feca496ae761d2a" MODIFIED="0" POSITION="right" TEXT="​预测鲍鱼年龄"/>
        </node>
        <node CREATED="0" ID="437c81af54a741e9a7b64bc1d49d5f62" MODIFIED="0" POSITION="left" TEXT="基本形式">
            <node CREATED="0" ID="7ad4f42ac3564c4cab0217f17a217bf4" MODIFIED="0" POSITION="right" TEXT="​什么是线性模型:  f(xi)=wTxi+b≃yif(x_i) = w^Tx_i + b \simeq y_i f(xi​)=wTxi​+b"/>
            <node CREATED="0" ID="6ee9d061df0a4fb6bec1fa8dcff18c35" MODIFIED="0" POSITION="right" TEXT="​线性回归的目的: 使拟合的  f(xi)≃yif(x_i) \simeq y_i f(xi​)≃yi​ "/>
            <node CREATED="0" ID="8ec0555ba43e450fbe5b335c2aaf5360" MODIFIED="0" POSITION="right" TEXT="​如何确定w和b(最优化问题):残差平方和最小  minθ∣∣Xθ−y∣∣22\underset{\theta}{min} {|| X\theta - y||_2}^2 θmin​∣∣Xθ−y∣∣2​2 "/>
        </node>
        <node CREATED="0" ID="ec50fb1ef4f54c1e811c0f7f1ddd5bed" MODIFIED="0" POSITION="right" TEXT="​进行线性回归有哪些方法">
            <node CREATED="0" ID="b49db16945cd4e7eb0b08f809639f336" MODIFIED="0" POSITION="right" TEXT="​普通最小二乘法 Ordinary Least Squares">
                <node CREATED="0" ID="7a38ce29bffc486c834636f1fcb9597c" MODIFIED="0" POSITION="right" TEXT="最小化的目标函数: 残差平方和的无偏估计   minθ∣∣Xθ−y∣∣22\underset{\theta}{min} {|| X\theta - y||_2}^2θmin​∣∣Xθ−y∣∣2​2 "/>
            </node>
            <node CREATED="0" ID="ac964e56ab07435a908723c1909c9949" MODIFIED="0" POSITION="right" TEXT="​局部加权线性回归:Locally Weighted Linear Regression，LWLR">
                <node CREATED="0" ID="b40447e35f6e4aae9713d8c7310072f2" MODIFIED="0" POSITION="right" TEXT="​最小化的目标函数:对训练集的数据加权后的最小均方误差:  ∑iwi(yi−θTxi)2\sum_i w_i(y_i-\theta^Tx_i)^2∑i​wi​(yi​−θTxi​)2 "/>
                <node CREATED="0" ID="1ddf4e13c88a4e4996b3bee19f51ae33" MODIFIED="0" POSITION="right" TEXT="​如何对确每个点的权重:高斯核   w(i,i)=exp((xi−x)2−2k2)w(i, i) = exp\left(\frac {(x_i-x)^2}{-2k^2}\right)w(i,i)=exp(−2k2(xi​−x)2​) "/>
            </node>
            <node CREATED="0" ID="a23a120dedad4a12b7176b2a3e7bc543" MODIFIED="0" POSITION="right" TEXT="​前向逐步回归 Forword stage wise">
                <node CREATED="0" ID="e453d32ee8ac4ae1bcbe8fabf6027e0f" MODIFIED="0" POSITION="right" TEXT="​操作方法: 一开始，所有权重都设置为 0，然后迭代多次, 每一步所做的决策是对某个权重增加或减少一个很小的值, 根据误差判断是增加还是减少权重."/>
            </node>
            <node CREATED="0" ID="6d32b31528534fb3a09975b930304982" MODIFIED="0" POSITION="right" TEXT="​岭回归  Ridge Regress">
                <node CREATED="0" ID="5e158b49c52b49cb9f36c25bf33125ce" MODIFIED="0" POSITION="right" TEXT="">
                    <node CREATED="0" ID="aaebe536262546dcb0094237422b4ac7" MODIFIED="0" POSITION="right" TEXT="最小化的目标函数: 带L2罚项的残差平方和   minθ∣∣Xθ−y∣∣22+α∣∣θ∣∣22\underset {\theta}{min} ||X\theta - y||_2^2 + \alpha ||\theta||_2^2θmin​∣∣Xθ−y∣∣22​+α∣∣θ∣∣22​ "/>
                </node>
            </node>
            <node CREATED="0" ID="567b7dee44ad4db9a0ae58ab1f6626f3" MODIFIED="0" POSITION="right" TEXT="套索方法(Lasso，The Least Absolute Shrinkage and Selection Operator) ">
                <node CREATED="0" ID="02783eb2ed124d918de166322db3ee27" MODIFIED="0" POSITION="right" TEXT="​最小化的目标函数: 带L1罚项的...   minθ∣∣Xθ−y∣∣2+α∣∣θ∣∣1\underset {\theta} {min} ||X\theta-y||^2 + \alpha||\theta||_1θmin​∣∣Xθ−y∣∣2+α∣∣θ∣∣1​ "/>
            </node>
            <node CREATED="0" ID="ea1530bfb93a46e197689e844dc560c8" LINK="https://blog.csdn.net/u014664226/article/details/52240272/" MODIFIED="0" POSITION="right" TEXT="最小角回归 LARS (Least Angle Regression)"/>
            <node CREATED="0" ID="3dd9af73b724414ca6f077f99cefbf6f" LINK="https://github.com/apachecn/sklearn-doc-zh/blob/master/docs/0.21.3/2.md#1110-%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%9B%9E%E5%BD%92" MODIFIED="0" POSITION="right" TEXT="​贝叶斯岭回归Bayesian Ridge Regression"/>
        </node>
        <node CREATED="0" ID="8ec3124bc1754a8e9bd3f923c573fe43" MODIFIED="0" POSITION="left" TEXT="​需要具备的基础知识">
            <node CREATED="0" ID="2109b6c891ef459f8a681af448e38eed" LINK="http://blog.csdn.net/nomadlx53/article/details/50849941" MODIFIED="0" POSITION="left" TEXT="​矩阵求导"/>
        </node>
        <node CREATED="0" ID="dd29b946947c4298ac7238b5dc608f43" MODIFIED="0" POSITION="right" TEXT="​使用技巧相关"/>
    </node>
</map>
