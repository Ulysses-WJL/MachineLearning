<?xml version="1.0" encoding="UTF-8" standalone="no"?><map version="0.8.1"><node CREATED="1577784581491" ID="14c99dg47cndcrf9k6np3igcka" MODIFIED="1577784581491" TEXT="EM算法及其推广"><node CREATED="1577784581491" ID="4e5evr2sqpr0emsqtnlitstke4" MODIFIED="1577784581491" POSITION="right" TEXT="简述"><node CREATED="1577784581491" ID="2se5jpg74b2bt3gr76ibqmbe8r" MODIFIED="1577784581491" TEXT="EM算法是含有隐变量的概率模型极大似然估计或极大后验概率估计的迭代算法"/></node><node CREATED="1577784581491" ID="6icp15iu59qv2gsjfdq6um4hht" MODIFIED="1577784581491" POSITION="right" TEXT="算法"><node CREATED="1577784581491" ID="0mlgkj1216j0srq7vcq8qtj7e6" MODIFIED="1577784581491" TEXT="EM算法"><node CREATED="1577784581491" ID="10174cqukp263ucvvn7t1b089f" MODIFIED="1577784581491" TEXT="M步: 求极大(maximization)"><node CREATED="1577784581491" ID="696dudsutfc4madk79lebk81fn" MODIFIED="1577784581491" TEXT="使期望达到极大"/></node><node CREATED="1577784581491" ID="385mh6jnvg5gkfbvecm1kvu2go" MODIFIED="1577784581491" TEXT="E步: 求期望(expectation)"><node CREATED="1577784581491" ID="7l6qtpvtiaobmsuspds8u5gkbl" MODIFIED="1577784581491" TEXT="完全数据的对数似然函数关于在给定观测数据和当前参数下对未观测数据的条件概率分布的期望"/></node></node><node CREATED="1577784581491" ID="7eakivvn1mpeso30flatp4a92r" MODIFIED="1577784581491" TEXT="EM算法的导出"><node CREATED="1577784581491" ID="2odgokb87vpju3tok6lp5kok35" MODIFIED="1577784581491" TEXT="保证每次迭代过程中, 观测数据关于参数的似然估计变大"/></node><node CREATED="1577784581491" ID="46ljl95s7a3cck9mta4uqirbf9" MODIFIED="1577784581491" TEXT="注意"><node CREATED="1577784581491" ID="70i28uvsjgsh5srl5v5p6n3iq9" MODIFIED="1577784581491" TEXT="在一般条件下EM算法是收敛的，但不能保证收敛到全局最优。所以在应用中, 初值的选择变得非常重要, 常用方法是选取几个不同的初值进行迭代, 选择其中估计值最好的"/></node></node><node CREATED="1577784581491" ID="4il388g80a5khcfpcgdbuk0ca6" MODIFIED="1577784581491" POSITION="right" TEXT="推广"><node CREATED="1577784581491" ID="33ikhv5mpfgaflmh6k33967lvo" MODIFIED="1577784581491" TEXT="高斯混合模型GMM"><node CREATED="1577784581491" ID="4kc27t254sgl133vmmppto83l2" MODIFIED="1577784581491" TEXT="形式上是不同高斯分布模型的加权求和, 实际代表以不同概率从分模型中选取一个生成观测值"/></node><node CREATED="1577784581491" ID="7sa9frp9a6bl27vcpfo1qhvst0" MODIFIED="1577784581491" TEXT="广义期望极大算法GEM"/></node><node CREATED="1577784581491" ID="0vh1p6f187s624qnb3f8l0bacq" MODIFIED="1577784581491" POSITION="left" TEXT="例子"><node CREATED="1577784581491" ID="2klce3itslr0t9letg7p6l9lpl" MODIFIED="1577784581491" TEXT="3枚硬币的例子"/></node><node CREATED="1577784581491" ID="73s14bjsk24eck5nchceb42u27" MODIFIED="1577784581491" POSITION="left" TEXT="一些知识点"><node CREATED="1577784581491" ID="3ed4nqcumt8l3nqj76sna5h3u2" MODIFIED="1577784581491" TEXT="观测变量(observable variable)"/><node CREATED="1577784581491" ID="58uq8tam6vmbeao5f6jm82jp8t" MODIFIED="1577784581491" TEXT="隐变量 潜在变量(latent variable)"/><node CREATED="1577784581491" ID="0l86u08mr4jahhn3p6l4pm8p1l" MODIFIED="1577784581491" TEXT="不完全数据与完全数据"/></node></node></map>
