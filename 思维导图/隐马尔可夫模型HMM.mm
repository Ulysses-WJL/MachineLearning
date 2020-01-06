<?xml version="1.0" encoding="UTF-8" standalone="no"?><map version="0.8.1"><node CREATED="1578295835003" ID="1r9pju40mt5kqs518keme71aqc" MODIFIED="1578295835003" TEXT="隐马尔可夫模型HMM"><node CREATED="1578295835003" ID="2l9g5h5drp7c2dfeeaev59tb9t" MODIFIED="1578295835003" POSITION="right" TEXT="马尔可夫链Markov chain"><node CREATED="1578295835003" ID="6ma04tmblgetssmiakrb1t0mjr" MODIFIED="1578295835003" TEXT="状态空间中经过从一个状态到另一个状态的转换的随机过程"/><node CREATED="1578295835003" ID="31gd51vv8hpk4vp1q24bvb5k56" MODIFIED="1578295835003" TEXT="一步转移概率, 一步转移概率矩阵"/><node CREATED="1578295835003" FOLDER="true" ID="0juplei18mf40uv2m5vrfhacim" MODIFIED="1578295835003" TEXT="无后效性:下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关"/><node CREATED="1578295835003" ID="39ssfhgk0quo7qpun2hh15mqd9" MODIFIED="1578295835003" TEXT="例子: 1. 随机漫步; 2. 熊市, 牛市, 横盘相互转换模型; 3. 0-1传输系统"/></node><node CREATED="1578295835003" ID="4rehjhe69q2dhddrjnh4fkee2c" MODIFIED="1578295835003" POSITION="right" TEXT="HMM"><node CREATED="1578295835003" ID="37u9005ojrc8h9d68kl6r9mfrk" MODIFIED="1578295835003" TEXT="隐藏的马尔可夫链随机生成不可观测的状态的序列，再由各个状态随机生成一个观测而产生观测的序列"/><node CREATED="1578295835003" ID="7s7o4bcmdf118o8jencmcblcdo" MODIFIED="1578295835003" TEXT="三类问题"><node CREATED="1578295835003" ID="6lmnjj1j2vjhk7ra10i03dc2do" MODIFIED="1578295835003" TEXT="概率计算问题: 求观测序列的概率"><node CREATED="1578295835003" ID="1lfimv52kte10s0v1sjsppgse1" MODIFIED="1578295835003" TEXT="1: 直接计算,复杂度O(TN^T)阶 不可取"/><node CREATED="1578295835003" ID="3a8bp21kh77sndtc0v48651p79" MODIFIED="1578295835003" TEXT="2. 前向算法"><node CREATED="1578295835003" ID="1kasare5vu9lv94047ltufd5a5" MODIFIED="1578295835003" TEXT="前向概率"/></node><node CREATED="1578295835003" ID="1ho2rpgtp8e3mkusdls44eri4i" MODIFIED="1578295835003" TEXT="3. 后向算法"><node CREATED="1578295835003" ID="16fbqd9el6fg2m470gnoj6ovtq" MODIFIED="1578295835003" TEXT="后向概率"/></node></node><node CREATED="1578295835003" ID="224khk4hv8ov1ugb1si769jgps" MODIFIED="1578295835003" TEXT="学习算法: 参数求解问题"><node CREATED="1578295835003" ID="5spoibo3cfic5joj1lfsns55l5" MODIFIED="1578295835003" TEXT="鲍姆-韦尔奇算法(EM算法的原理)"/></node><node CREATED="1578295835003" ID="3ae3dc2s3ctsievqotuuh2ivok" MODIFIED="1578295835003" TEXT="预测算法, 解码问题: 求隐藏序列"><node CREATED="1578295835003" ID="3go94tq5q0k1lr6mthi66mtfrh" MODIFIED="1578295835003" TEXT="1. 近似算法: 计算简单, 但不能保证整体最优"/><node CREATED="1578295835003" ID="3qtuo05nsg52ik2of37q0grsfr" MODIFIED="1578295835003" TEXT="2. 维特比算法: 使用动态规划"/></node></node></node></node></map>
