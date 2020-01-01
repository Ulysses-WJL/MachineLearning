
<map version="0.9.0">
    <node CREATED="0" ID="8ba5fd0e7f494b43a1912aa8f88a0bda" MODIFIED="0" TEXT="EM算法及其推广">
        <node CREATED="0" ID="c85849b617a348fb9a4be3e9dfc85e3e" MODIFIED="0" POSITION="right" TEXT="简述">
            <node CREATED="0" ID="cf7cf59568264bfd856ede052638ab47" MODIFIED="0" TEXT="EM算法是含有隐变量的概率模型极大似然估计或极大后验概率估计的迭代算法"/>
        </node>
        <node CREATED="0" ID="c7c4808a503d4a969223b586a332307c" MODIFIED="0" POSITION="right" TEXT="算法">
            <node CREATED="0" ID="f9451c155ff4494fa924ab42adfc3eea" MODIFIED="0" TEXT="EM算法">
                <node CREATED="0" ID="7956c5912fab4119957872465e35ed9b" MODIFIED="0" TEXT="M步: 求极大(maximization)使期望达到极大">
                    <node CREATED="0" ID="e126226cc3aa45ec9e1a8e494d65a1f3" MODIFIED="0" TEXT="每一次迭代更新参数"/>
                </node>
                <node CREATED="0" ID="438661b176034400a42e25543f3e74eb" MODIFIED="0" TEXT="​选择参数初值"/>
                <node CREATED="0" ID="0c73d6123e084972ab26a092d267f822" MODIFIED="0" TEXT="​迭代停止条件: Q函数收敛或参数收敛"/>
                <node CREATED="0" ID="6f4a256b43e043b6a59f88ad79a11e11" MODIFIED="0" TEXT="E步: 求期望(expectation) Q函数">
                    <node CREATED="0" ID="1cc6a4c6bf8046d3bd9d740e62803383" MODIFIED="0" TEXT="完全数据的对数似然函数关于在给定观测数据和当前参数下对未观测数据的条件概率分布的期望"/>
                </node>
            </node>
            <node CREATED="0" ID="2a337953c3a44f15baebc0f1cdd02ff6" MODIFIED="0" TEXT="EM算法的导出">
                <node CREATED="0" ID="b0bc196801184c90947df554cd586c14" MODIFIED="0" TEXT="保证每次迭代过程中, 观测数据关于参数的似然估计变大"/>
            </node>
            <node CREATED="0" ID="18fb17489802454a9ace6b85e3cbb4de" MODIFIED="0" TEXT="注意">
                <node CREATED="0" ID="67cbecae44164a0f9b35fd2c89dfc255" MODIFIED="0" TEXT="在一般条件下EM算法是收敛的，但不能保证收敛到全局最优。所以在应用中, 初值的选择变得非常重要, 常用方法是选取几个不同的初值进行迭代, 选择其中估计值最好的"/>
            </node>
        </node>
        <node CREATED="0" ID="9fc531b1711c4af7ae7e27200f0d17ba" MODIFIED="0" POSITION="right" TEXT="推广">
            <node CREATED="0" ID="1abfba40ed7c4947bfb19ad2881e9af9" MODIFIED="0" TEXT="高斯混合模型GMM">
                <node CREATED="0" ID="9549e254adbc42e4839320c9072f2f55" MODIFIED="0" TEXT="形式上是不同高斯分布模型的加权求和, 实际代表以不同概率从分模型中选取一个生成观测值"/>
            </node>
            <node CREATED="0" ID="d020ba71e80b421b9a2ab98315f77c95" MODIFIED="0" TEXT="广义期望极大算法GEM"/>
        </node>
        <node CREATED="0" ID="8bd85095b392413a9ba8d3ea39c7e4a2" MODIFIED="0" POSITION="left" TEXT="例子">
            <node CREATED="0" ID="68f67fdb611f4c94bd941fd06c9bac63" MODIFIED="0" TEXT="3枚硬币的例子"/>
        </node>
        <node CREATED="0" ID="13d5034e3509432b8c157d4fea25f55d" MODIFIED="0" POSITION="left" TEXT="一些知识点">
            <node CREATED="0" ID="27c2da3eac2240f39e11eabc35a815c1" MODIFIED="0" TEXT="观测变量(observable variable)"/>
            <node CREATED="0" ID="a11366e765a64b9fbf4ccf90243669af" MODIFIED="0" TEXT="隐变量 潜在变量(latent variable)"/>
            <node CREATED="0" ID="86647437d80a464aa4b4caf75bc418a5" MODIFIED="0" TEXT="不完全数据与完全数据"/>
        </node>
    </node>
</map>
