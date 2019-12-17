
<map version="0.9.0">
    <node CREATED="0" ID="a523f802d7584d26a4eac4a034b88a14" MODIFIED="0" TEXT="支持向量机Support Vector Machine">
        <node CREATED="0" ID="7bd8fd771cbc49eba0f1ac012e3aa79b" MODIFIED="0" POSITION="right" TEXT="定义">
            <node CREATED="0" ID="713d45b9fb604eafa00a940ba1329024" MODIFIED="0" TEXT="是什么: 一种二类分类模型"/>
            <node CREATED="0" ID="54730a18fcae4c179d98d560bbad7f99" MODIFIED="0" TEXT="基本模型: 定义在特征空间上的间隔最大线性分类器"/>
            <node CREATED="0" ID="f87e38c14124423982cf0bbfc6ba50d1" MODIFIED="0" TEXT="支持向量机与感知机的区别: 1. SVM包括核技巧, 可以处理非线性分离;2. 间隔最大法,可以确定唯一的分类超平面"/>
        </node>
        <node CREATED="0" ID="432587fbf2a4416796cf9c35e5d7230c" MODIFIED="0" POSITION="left" TEXT="​技巧">
            <node CREATED="0" ID="e2672f0e3be54e82a4976052fa97559a" MODIFIED="0" TEXT="​核技巧(kernel trick), 用线性分类方法求解非线性分类问题">
                <node CREATED="0" ID="8d21835810154d3ea24bc4774f68fab5" MODIFIED="0" TEXT="​核函数    K(x,z)=ϕ(x)⋅ϕ(z)K(x, z) = \phi(x) \cdot \phi(z)K(x,z)=ϕ(x)⋅ϕ(z)  "/>
                <node CREATED="0" ID="4c33a6943b0c435d905e46d472c41072" MODIFIED="0" TEXT="​常用核函数">
                    <node CREATED="0" ID="19c8c79a2d5e443e8ba3fb480cb13f3c" MODIFIED="0" TEXT="​高斯核:    K(x,z)=exp(−∣∣x−z∣∣22σ2)K(x, z) = exp(- \frac {||x-z||^2}{2\sigma^2})K(x,z)=exp(−2σ2∣∣x−z∣∣2​) "/>
                    <node CREATED="0" ID="a1f83f0fb8294adca85174151387bb33" MODIFIED="0" TEXT="​多项式核:   K(x,z)=(x⋅z+1)pK(x, z) = (x \cdot z +1)^pK(x,z)=(x⋅z+1)p "/>
                </node>
                <node CREATED="0" ID="ef5628a844344fd7b44f04258e154113" MODIFIED="0" TEXT="​思想: 只定义核函数, 不显示地定义映射函数"/>
                <node CREATED="0" ID="3812a95d5b40492dbe3de5ed74b3c48e" MODIFIED="0" TEXT="​正定核的等价定义: 对称函数K(x, z)对应的Gram矩阵K是半正定的"/>
            </node>
            <node CREATED="0" ID="3a83f0418ae944cdb2b3699e577f6b9c" MODIFIED="0" TEXT="序列最小最优(Sequential Minimal Optimization, SMO)算法">
                <node CREATED="0" ID="e3c6e0b7aaed41a8b4bdfced0d30d6bf" MODIFIED="0" TEXT="特点: 将原二次规划问题分解为只有两个变量的二次规划子问题, 并对子问题进行解析求解，直到所有变量满足KKT条件为止"/>
                <node CREATED="0" ID="c3fa04d56e7a4a4391049c1520b4f66f" MODIFIED="0" TEXT="​算法组成">
                    <node CREATED="0" ID="76a85cdc7dd34704a69110028612c332" MODIFIED="0" TEXT="​求解2个变量的二次规划的解析方法">
                        <node CREATED="0" ID="dfb7d6a7747e463796205f8040d7cbf2" MODIFIED="0" TEXT="​1. 所有变量都满足KKT, 则它是最优解(充要条件)"/>
                        <node CREATED="0" ID="56463a3993cf4a2fa161f6f33838fa59" MODIFIED="0" TEXT="​2.否则, 选2个变量构建二次规划问题, 获得更接近原始问题的解"/>
                        <node CREATED="0" ID="26f8c587efdb46ed9b30a172a128a957" MODIFIED="0" TEXT=""/>
                    </node>
                    <node CREATED="0" ID="6c4d5931f66f4c10b74c386ab69b76fd" MODIFIED="0" TEXT="​选择变量的启发式方法"/>
                </node>
            </node>
        </node>
        <node CREATED="0" ID="830fc167bcbc41f6bf76bb977eb3827d" MODIFIED="0" POSITION="right" TEXT="SVM的3种类别">
            <node CREATED="0" ID="8a14661a194f4e7d87042d8ccf30f2fc" MODIFIED="0" TEXT="线性可分(硬间隔)支持向量机">
                <node CREATED="0" ID="59bced7cb78d43a7a45baa05d0aa6526" MODIFIED="0" TEXT="处理对象: 线性可分的集合"/>
                <node CREATED="0" ID="a822d9688887477f87e2f1b47cc0f4e2" MODIFIED="0" TEXT="处理方法: 最大间隔法, 几何间隔最大"/>
                <node CREATED="0" ID="a0d224161f094ea5813a67d239d0c808" MODIFIED="0" TEXT="凸二次规划">
                    <node CREATED="0" ID="06601431b2ea435991ab92d9f6b5b3e1" MODIFIED="0" TEXT="1. 几何间隔最大, 满足不等式约束  max⁡ r∗∣∣w∣∣s.t. yi(wTxi+b)≥r∗, i=1,2,..,m\max\ \frac{r^*}{||w||} \\ s.t. \ y_i({w^T}x_i+{b})\geq {r^*},\ i=1,2,..,mmax ∣∣w∣∣r∗​s.t. yi​(wTxi​+b)≥r∗, i=1,2,..,m "/>
                    <node CREATED="0" ID="ffa82cac7b4941c1ac1d6c5951b893fe" MODIFIED="0" TEXT="2. 转为凸函数  min⁡ 12∣∣w∣∣2s.t. yi(wTxi+b)≥1, i=1,2,..,m\min\ \frac{1}{2}||w||^2 \\ s.t.\ y_i(w^Tx_i+b)\geq 1,\ i=1,2,..,mmin 21​∣∣w∣∣2s.t. yi​(wTxi​+b)≥1, i=1,2,..,m "/>
                    <node CREATED="0" ID="2250f2bb7db140218dd2962c74cb2ce2" MODIFIED="0" TEXT="3. 构造拉格朗日乘子法  L(w,b,α)=12∣∣w∣∣2+∑i=1mαi(−yi(wTxi+b)+1)L(w, b, \alpha) = \frac{1}{2}||w||^2+\sum^m_{i=1}\alpha_i(-y_i(w^Tx_i+b)+1)L(w,b,α)=21​∣∣w∣∣2+∑i=1m​αi​(−yi​(wTxi​+b)+1) "/>
                    <node CREATED="0" ID="d1d50e586822477182938740adb932e8" MODIFIED="0" TEXT="4. 拉格朗日函数对原变量求导, 并令导数为0, 得到 w和b 的表示, 再代回到原式   min⁡ L(w,b,α)=∑i=1mαi−12∑i,j=1mαiαjyiyj(xixj)\min\ L(w, b, \alpha) = \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)min L(w,b,α)=∑i=1m​αi​−21​∑i,j=1m​αi​αj​yi​yj​(xi​xj​) "/>
                    <node CREATED="0" ID="f807621a15de4bde971cc41713ffe250" MODIFIED="0" TEXT="5. 对偶问题: min L 对alpha的极大, 再转成极小问题   max⁡ ∑i=1mαi−12∑i,j=1mαiαjyiyj(xixj)=min⁡12∑i,j=1mαiαjyiyj(xixj)−∑i=1mαis.t. ∑i=1mαiyi=0,αi≥0,i=1,2,...,m\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i \\ s.t.\ \sum^m_{i=1}\alpha_iy_i=0, \\ \alpha_i \geq 0,i=1,2,...,m max ∑i=1m​αi​−21​∑i,j=1m​αi​αj​yi​yj​(xi​xj​)=min21​∑i,j=1m​αi​αj​yi​yj​(xi​xj​)−∑i=1m​αi​s.t. ∑i=1m​αi​yi​=0,αi​≥0,i=1,2,...,m   max⁡ ∑i=1mαi−12∑i,j=1mαiαjyiyj(xixj)=min⁡12∑i,j=1mαiαjyiyj(xixj)−∑i=1mαis.t. ∑i=1mαiyi=0,αi≥0,i=1,2,...,m\max\ \sum^m_{i=1}\alpha_i-\frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)=\min \frac{1}{2}\sum^m_{i,j=1}\alpha_i\alpha_jy_iy_j(x_ix_j)-\sum^m_{i=1}\alpha_i \\ s.t.\ \sum^m_{i=1}\alpha_iy_i=0, \\ \alpha_i \geq 0,i=1,2,...,m"/>
                    <node CREATED="0" ID="5f1cdbade75d4a04b859cefd9fbe83ca" MODIFIED="0" TEXT="6. 解对偶问题, 得到原始最优化问题的解"/>
                </node>
            </node>
            <node CREATED="0" ID="7e830bd8de4d4c74a66efcd0a83f2338" MODIFIED="0" TEXT="线性支持向量机">
                <node CREATED="0" ID="d1538029950f4780aabdcbc62a3fa482" MODIFIED="0" TEXT="处理对象: 线性不可分的集合, 近似线性可分(通常情况)"/>
                <node CREATED="0" ID="320f16ae65674930b90c5ae83a0aff04" MODIFIED="0" TEXT="合页损失函数: 线性支持向量机的另一种解释, 最小化二阶范数正则化的损失函数, 分类正确且函数间隔(确性度)  yi(w⋅xi+b)y_i(w \cdot x_i +b)yi​(w⋅xi​+b) 大于1时,损失函数是0, 否则损失是  1−yi(w⋅xi+b)1-y_i(w \cdot x_i +b)1−yi​(w⋅xi​+b) "/>
                <node CREATED="0" ID="9de35a11544c49f0aab5d8f863339c1f" MODIFIED="0" TEXT="处理方法: 软间隔最大化-给函数间隔添加一个松弛变量  ξ\xiξ 使其满足不等式约束, 同时优化目标加上惩罚项"/>
            </node>
            <node CREATED="0" ID="1813fae26d694714b2d6dadad595b4e4" MODIFIED="0" TEXT="非线性支持向量机">
                <node CREATED="0" ID="928e67567b664dd8a0f861d5018ef387" MODIFIED="0" TEXT="处理对象:非线性分类问题"/>
                <node CREATED="0" ID="560252ae81c34148b4a77c65e1a8a969" MODIFIED="0" TEXT="处理方式: 使用核函数  K(x,z)K(x, z)  K(x,xi)K(x, x_i)K(x,xi​) 来替换线性支持向量机的对偶问题的内积  (x,xi)(x, x_i)(x,xi​) "/>
            </node>
        </node>
        <node CREATED="0" ID="a77ae96f696a439aa6db1aeca986900b" MODIFIED="0" POSITION="left" TEXT="基础知识">
            <node CREATED="0" ID="f48eca84c1da44eda05c45e559bebdde" MODIFIED="0" TEXT="几何间隔geometric margin">
                <node CREATED="0" ID="fe8d1939fda441fe9c663e7a6ebb8ce7" MODIFIED="0" TEXT="定义:对函数间隔添加规范化后的距离    yi(w∣∣w∣∣⋅xi+b∣∣w∣∣)y_i(\frac w {||w||} \cdot x_i + \frac b {||w||})yi​(∣∣w∣∣w​⋅xi​+∣∣w∣∣b​) "/>
            </node>
            <node CREATED="0" ID="5473c8ed81114a98bed7ef1b997f7294" MODIFIED="0" TEXT="函数间隔function margin">
                <node CREATED="0" ID="6a5e2d051ae6465fb3e5eff66cab715c" MODIFIED="0" TEXT="定义:点到超平面的带符号距离    yi(w⋅xi+b)y_i(w\cdot x_i + b) yi​(w⋅xi​+b) "/>
                <node CREATED="0" ID="5c664afe819745d1b5a39e3d08adc296" MODIFIED="0" TEXT="作用: 表示分类的正确性及确信度"/>
            </node>
        </node>
    </node>
</map>
