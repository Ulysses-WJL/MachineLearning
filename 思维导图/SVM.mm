
<map version="0.9.0">
    <node CREATED="0" ID="38f445aa2aa1416f8080ce9471a2bd13" MODIFIED="0" TEXT="支持向量机Support Vector Machine">
        <node CREATED="0" ID="c671055428e2491ebf01792df71a6211" MODIFIED="0" POSITION="right" TEXT="定义">
            <node CREATED="0" ID="8b649127de4942449397775f9bc35565" MODIFIED="0" TEXT="是什么: 一种二类分类模型"/>
            <node CREATED="0" ID="5f6b68ef01a04938ad524e454b5d2efd" MODIFIED="0" TEXT="基本模型: 定义在特征空间上的间隔最大线性分类器"/>
            <node CREATED="0" ID="439c2cd6751c4a78a8b55079ddfb587d" MODIFIED="0" TEXT="支持向量机与感知机的区别: 1. SVM包括核技巧, 可以处理非线性分离;2. 间隔最大法,可以确定唯一的分类超平面"/>
        </node>
        <node CREATED="0" ID="f63f29468a6f4ec9baf89ea1ccc77101" MODIFIED="0" POSITION="left" TEXT="​技巧">
            <node CREATED="0" ID="c5047af6cf24481fb72a34318f6467c7" MODIFIED="0" POSITION="left" TEXT="​核技巧(kernel trick), 用线性分类方法求解非线性分类问题">
                <node CREATED="0" ID="a8657a11abd34a6380cd16675768c8fb" MODIFIED="0" POSITION="left" TEXT="​核函数  K(x,z)=ϕ(x)⋅ϕ(z)K(x, z) = \phi(x) \cdot \phi(z)K(x,z)=ϕ(x)⋅ϕ(z) "/>
                <node CREATED="0" ID="a7dc33dc02bc4efaae8ebe7879bc4a7c" MODIFIED="0" POSITION="left" TEXT="​想法: 只定义核函数, 不显示地定义映射函数"/>
                <node CREATED="0" ID="788e63c30ae546d8974030f09577b485" MODIFIED="0" POSITION="left" TEXT="​正定核"/>
            </node>
        </node>
        <node CREATED="0" ID="22f38ece949141e5928c9e9d20363b72" MODIFIED="0" POSITION="right" TEXT="SVM的3种类别">
            <node CREATED="0" ID="c61939fbad314ecdb857efd19ba24236" MODIFIED="0" TEXT="线性可分(硬间隔)支持向量机">
                <node CREATED="0" ID="1fee8d389ccf426ab25088901cf434ce" MODIFIED="0" TEXT="处理对象: 线性可分的集合"/>
                <node CREATED="0" ID="72a71d984a644747a06af7120d7cdb83" MODIFIED="0" TEXT="处理方法: 最大间隔法, 几何间隔最大"/>
                <node CREATED="0" ID="b01043f8dad74501a5c3e3fea58404ab" MODIFIED="0" TEXT="凸二次规划">
                    <node CREATED="0" ID="c1cee6ff608344b4890cea669784a300" MODIFIED="0" TEXT="1. 几何间隔最大, 满足不等式约束"/>
                    <node CREATED="0" ID="b7e3b4debb2e48bc9806c442c013514e" MODIFIED="0" TEXT="2. 转为凸函数"/>
                    <node CREATED="0" ID="c7f89a687b5f4ef2be3bb2b61d78ec9b" MODIFIED="0" TEXT="3. 构造拉格朗日乘子法"/>
                    <node CREATED="0" ID="92d010f9dccc4f838759910b6e4564e2" MODIFIED="0" TEXT="4. 拉格朗日函数对原变量求导, 并令导数为0, 得到 w和b 的表示, 再代回到原式"/>
                    <node CREATED="0" ID="a3f6bd5b1e0442f09a10883bf731eeda" MODIFIED="0" TEXT="5. 对偶问题: min L 对alpha的极大, 再转成极小问题"/>
                    <node CREATED="0" ID="4508567b883741748800ea525ecc1b82" MODIFIED="0" TEXT="6. 解对偶问题, 得到原始最优化问题的解"/>
                </node>
            </node>
            <node CREATED="0" ID="9f9045ec8dcb4ce18d3227e2f231a2ef" MODIFIED="0" TEXT="线性支持向量机">
                <node CREATED="0" ID="1bdf92ccbe854e2a846ef4fec01ce5ad" MODIFIED="0" TEXT="处理对象: 线性不可分的集合(通常情况)"/>
                <node CREATED="0" ID="178307a3510b44fcb65b4a8ed7fcb6be" MODIFIED="0" TEXT="合页损失函数: 线性支持向量机的另一种解释, 最小化二阶范数正则化的损失函数"/>
                <node CREATED="0" ID="5e4b8b0d4d7340c494b9066d4bcbd31c" MODIFIED="0" TEXT="处理方法: 软间隔最大化-给函数间隔添加一个松弛变量  ϵ\epsilon  ξ\xiξ  使其满足不等式约束, 同时优化目标加上惩罚项"/>
            </node>
        </node>
        <node CREATED="0" ID="d685a1f05f864fc9af80f5edcbc32340" MODIFIED="0" POSITION="left" TEXT="基础知识">
            <node CREATED="0" ID="b3241d256ba9445ea9ff60401d3dc656" MODIFIED="0" TEXT="几何间隔geometric margin">
                <node CREATED="0" ID="db22ac1319f343f699655c96cf0d3bae" MODIFIED="0" TEXT="定义:对函数间隔添加规范化后的距离  yi(w∣∣w∣∣⋅xi+b∣∣w∣∣)y_i(\frac w {||w||} \cdot x_i + \frac b {||w||})yi​(∣∣w∣∣w​⋅xi​+∣∣w∣∣b​) "/>
            </node>
            <node CREATED="0" ID="acabbcdbc4ad47b79f9086ceb6bdf637" MODIFIED="0" TEXT="函数间隔function margin">
                <node CREATED="0" ID="0bc428802d0a460c97b93e785fcf325b" MODIFIED="0" TEXT="定义:点到超平面的带符号距离  yi(w⋅xi+b)y_i(w\cdot x_i + b)yi​(w⋅xi​+b) "/>
                <node CREATED="0" ID="df2635863ad6440687ffe0b1884afbc3" MODIFIED="0" TEXT="作用: 表示分类的正确性及确信度"/>
            </node>
        </node>
    </node>
</map>
