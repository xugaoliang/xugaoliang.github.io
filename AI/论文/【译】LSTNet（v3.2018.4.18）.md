# 【译】用深度神经网络建模长短期时间模式（v3,2018.4.18）

* 作者：Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu
* 论文：《Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks》
* 地址：https://arxiv.org/abs/1703.07015

---

## 个人总结

本文使用深度神经网络的方法解决多元时间序列预测的问题。总和运用了CNN,RNN,Attn及传统自回归预测的优势。

用CNN捕捉短期局部模式及各元之间的模式，用RNN捕捉长期模式，并从传统方法输入中汲取灵感设计了递归跳跃RNN，用以捕捉超长期模式及明显的周期模式，对于无明显周期模式的时间序列则采用Attn来捕捉关系，用传统AR来捕捉局部尺度。

模型的总体思路是：分为DNN及AR两部分，分别捕捉重复的非线性部分及局部尺度的线性部分。

一个新颖的借鉴点是在深度模型中加入AR部分解决尺度变化问题导致DNN对尺度变化不敏感的问题。

---

**摘要**: 多变量时间序列预测是一个涉及多个领域的重要机器学习问题，包括太阳能电站能量输出、电力消耗和交通堵塞的预测。在这些实际应用中产生的时态数据通常涉及长期和短期模式的混合，因此自回归模型和高斯过程等传统方法可能会失败。在这篇论文中，我们提出了一个新的深度学习框架，即长短期时间序列网络(LSTNet)，来解决这个开放的挑战。LSTNet使用卷积神经网络(CNN)和递归神经网络(RNN)来提取变量间的短期局部依赖模式，并发现时间序列趋势的长期模式。此外，我们利用传统的自回归模型来处理神经网络模型的尺度不敏感问题。在我们对具有复杂的重复模式混合的真实数据的评估中，LSTNet比几种最先进的基线方法取得了显著的性能改进。所有的数据和实验代码都可以在网上找到。

**关键词**
Multivariate Time Series, Neural Network, Autoregressive models

**ACM参考格式**:
Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. 2018. Mod- eling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In Proceedings of ACM Conference (SIGIR’18). ACM, New York, NY, USA, 11 pages. https://doi.org/10.475/123_4

## 1. 介绍

多元时间序列数据在我们的日常生活中是无处不在的，从股票市场的价格，高速公路的交通流量，太阳能发电厂的输出，不同城市的温度，等等。在这种应用程序中，用户往往对根据时间序列信号的历史观察预测新趋势或潜在危险事件感兴趣。例如，可以根据几小时前预测的交通堵塞模式设计更好的路线计划，通过预测近期的股票市场可以获得更大的利润。

![图1](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig1.png)

图1：旧金山湾区道路每小时的占用率，为期两星期

多元时间序列预测经常面临一个主要的研究挑战，即如何捕获和利用多变量之间的动态依赖关系。具体地说，现实世界的应用常常需要短期和长期重复模式的混合，如图1所示，它绘制了高速公路的每小时占用率。显然，有两个重复的模式，每天和每周。前者描绘了早晨和晚上的高峰，而后者反映了工作日和周末的模式。一个成功的时间序列预测模型应该捕捉这两种重复出现的模式，以便进行准确的预测。再举一个例子，考虑一个任务，根据不同位置的大型传感器测量到的太阳辐射，预测太阳能发电厂的输出。长期模式反映了昼夜、夏季和冬季等的差异，而短期模式反映了云层移动、风向变化等的影响。同样，如果不考虑这两种周期性特征，精确的时间序列预测是不可能的。然而，传统的方法如大量的自回归方法[2,12,22,32,35]在这方面存在不足，因为它们大多数不能区分这两种模式，也不能显式和动态地对它们的交互进行建模。针对时间序列预测中现有方法的局限性，我们提出了一个利用深度学习研究最新进展的新框架。

深度神经网络在相关领域得到了深入的研究，并对一系列问题的解决产生了非凡的影响。以[9]为例，递归神经网络(RNN)模型是近年来自然语言处理(NLP)研究的热点。特别是RNN的两个变种,即长期短期记忆(LSTM)[15]和门控递归单元(GRU)[6],显著提高了最先进的机器翻译,语音识别和其他NLP任务的性能，根据输入文档中词语之间的长短期依赖性，有效地捕获词语的含义[1，14，19]。另一个例子是在计算机视觉领域，卷积神经网络(CNN)模型[19,21]通过从输入图像中成功提取不同粒度级别的局部和平移不变特征(有时称为 shapelets )，显示出了出色的性能。

深度神经网络在时间序列分析中也越来越受到重视。以前的大部分工作都集中在时间序列分类上，即时间序列分类。类标签自动分配到时间序列输入的任务。例如，RNN体系结构被研究用于从医疗序列数据中提取信息模式[5,23]，并根据诊断类别对数据进行分类。RNN也被应用于移动数据，根据动作或活动[13]对输入序列进行分类。CNN模型也被用于动作/活动识别[13,20,31]，用于从输入序列中提取平移不变的局部模式作为分类模型的特征。

深度神经网络也被研究用于时间序列预测[8,33]，即，使用过去观测到的时间序列来预测未来范围上未知的时间序列的任务——范围越大，问题越难。这方面的工作包括早期使用朴素RNN模型[7]和结合ARIMA[3]和多层感知器(MLP)的混合模型[16,34,35]的工作，以及最近在时间序列预测[8]中使用的普通RNN和动态玻尔兹曼机的组合。

在本文中，我们提出了一个用于多元时间序列预测的深度学习框架，即长短期时间序列网络(LSTNet)，如图2所示。它利用卷积层发现多维输入变量之间的局部依赖模式，及递归层来捕获复杂的长期依赖模式。另外，利用输入时间序列信号的周期性特性，设计了一种叫递归跳跃(Recurrent-skip)的新的递归结构，用于捕获非常长期的依赖模式，使优化变得更容易。最后，LSTNet将传统的自回归线性模型与非线性神经网络部分并行，使得非线性深度学习模型对具有尺度变化的时间序列具有更强的鲁棒性。在实际季节时间序列数据集的实验中，我们的模型始终优于传统的线性模型和GRU递归神经网络。

本文的其余部分组织如下。第2节概述了相关的背景，包括代表性的自回归方法和高斯过程模型。第3节描述我们提出的LSTNet。第4节报告了我们的模型与真实世界数据集的强基线的比较的评估结果。最后，我们在第5节中总结我们的发现。

## 2. 相关背景

最著名的单变量时间序列模型之一是自回归综合移动平均(ARIMA)模型。ARIMA模型的流行是由于它的统计特性和模型选择过程中著名的Box-Jenkins方法[2]。ARIMA模型不仅能够适应各种指数平滑技术[25]，而且还具有足够的灵活性，可以包含其他类型的时间序列模型，包括自回归(AR)、移动平均(MA)和自回归移动平均(ARMA)。然而，由于ARIMA模型具有较高的计算成本，因此很少用于高维多变量时间序列预测。

另一方面，向量自回归(VAR)由于其简单性，可以说是多元时间序列中应用最广泛的模型[2,12,24]。VAR模型很自然地将AR模型扩展到多元设置，忽略了输出变量之间的依赖关系。近年来，各种VAR模型取得了显著进展，包括用于重尾时间序列的椭圆型VAR模型[27]和用于更好地解释高维变量之间依赖关系的结构化VAR模型[26]，等等。然而，VAR的模型容量随时间窗口大小线性增长，随变量数量二次增长。这意味着，当处理长期的时间模式时，继承的大型模型容易出现过拟合。为了缓解这一问题，[32]提出将原始的高维信号简化为低维隐藏的表征，然后应用VAR进行多种正则化选择的预测。

时间序列预测问题也可以看作是具有时变参数的标准回归问题。因此，不同损失函数和正则化项的回归模型应用于时间序列预测任务也就不足为奇了。例如,线性支持向量回归(SVR)(4、17)在超参数 $\epsilon$ 控制预测误差的阈值的基础下学习最大间隔超平面。岭回归是另一个例子,可以通过设置$\epsilon$为0，重新覆盖来自SVR的模型。最后，[22]应用了 LASSO 模型来鼓励模型参数的稀疏性，从而可以显示不同输入信号之间的有趣模式。由于在机器学习领域中有高质量的现成解算器(off-the-shelf solvers)，这些线性方法实际上对多元时间序列预测更高效。然而，与VARs一样，这些线性模型可能无法捕捉多元信号间复杂的非线性关系，从而导致性能低下，并以牺牲效率为代价。

高斯过程（Gaussian Processes,GP）是对连续函数域上的分布进行建模的一种非参数方法。这与参数化函数类(如VARs和SVRs)定义的模型形成了对比。GP可应用于[28]中提出的多元时间序列预测任务，并可作为贝叶斯推理中函数空间上的先验。例如，对于非线性状态空间模型，[10]提出了一种具有GP先验的完全贝叶斯方法，能够捕获复杂的动态现象。然而，高斯过程的能力是以高计算复杂度为代价的。由于核矩阵的矩阵求逆，多元时间序列预测的高斯过程的一个直接实现具有观测数的立方复杂度。

## 3. 框架

在本节中，我们首先制定时间序列预测问题，然后在接下来的部分中讨论所提议的LSTNet结构的细节(图2)。最后介绍了目标函数和优化策略。

![图2](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig2.png)

图2：长短期时间序列网络（LSTNet）概览

### 3.1 问题公式化

本文主要研究多元时间序列预测问题。更正式地说，给定一系列完全观测到的时间序列信号$Y=\{y_1,y_2,\cdots,y_T\}$ 其中 $y_t \in \mathbb{R}^n$,$n$是变量的维度，我们的目标是以滚动预测的方式预测一系列的未来信号。也就是说，我们假设$\{y_1,y_2,\cdots,y_T\}$可用，来预测$y_{T+h}$,其中$h$是当前时间戳之前的理想视界，同样的，我们假定$\{y_1,y_2,\cdots,y_T,y_{T+1}\}$可用，来预测下一个时间戳$y_{T+h+1}$的值。我们由此得到在时间戳$T$的输入矩阵为$X_T=\{y_1,y_2,\cdots,y_T\} \in \mathbb{R}^{n\times T}$。

在大多数情况下，预测工作的范围是根据环境的需要而选择的，例如:交通流量方面，预测工作的范围由小时至一天;对于股票市场数据，即使提前几秒或几分钟的预测对于产生回报也是有意义的。

图2展示了所提议的LSTnet架构的概述。LSTNet是一个深度学习框架，专门为长短期模式混合的多变量时间序列预测任务设计。在接下来的几节中，我们将详细介绍LSTNet的构建块。

### 3.2 卷积组件

LSTNet的第一层是一个没有池化的卷积网络，它的目标是提取时间维中的短期模式以及变量之间的局部依赖关系。卷积层由多个宽$w$高$n$的 filters (高度设置为相同数量的变量)组成。第$k$个 filter 扫过输入矩阵$X$并产生

$$
h_k = RELU(W_k * X + b_k)
$$

其中$*$表示卷积操作和输出$h_k$是一个向量，而$RELU$函数是$RELU(x) = \max(0,x)$。我们通过在输入矩阵$X$的左边填充0使每个$h_k$的长度为$T$。卷积层的输出矩阵大小为$d_c \times T$，其中$d_c$为 filters 数量。

### 3.3 递归组件

卷积层的输出同时输入到递归组件和递归跳越组件(在3.4小节中进行描述)。递归组件是一个带有门控递归单元（GRU)[6]的递归层，使用$RELU$函数作为隐藏的更新激活函数。$t$时刻递归单位的隐状态计算为:

$$
r_t = \sigma(x_tW_{xr}+h_{t-1}W_{hr}+b_r) \\\\
u_t = \sigma(x_tW_{xu}+h_{t-1}W_{hu}+b_u) \\\\
c_t = RELU(x_tW_{xc}+r_t \odot(h_{t-1}W_{hc})+b_c) \\\\
h_t = (1-u_t)\odot h_{t-1}+u_t\odot c_t
$$

其中$\odot$是按元素乘，$\sigma$是 sigmoid 函数，$x_t$是这一层在时刻$t$的输入。这一层的输出是每个时间戳的隐状态，虽然研究人员倾向于使用 tanh 函数作为隐藏更新激活函数，但我们从经验上发现 RELU 的性能更可靠，通过它梯度更容易反向传播。

### 3.4 递归跳跃组件

带有GRU[6]和LSTM[15]单元的递归层经过精心设计，以记住历史信息，从而了解相对长期的依赖关系。然而，由于梯度消失，GRU和LSTM在实际应用中往往不能捕捉到非常长期的相关性。我们建议通过一种新的递归跳跃组件来缓解这个问题，该组件利用了真实集合中的周期性模式。例如，每天的用电量和交通使用量都呈现出明显的规律。如果我们要预测今天t点的用电量，季节预测模型中的一个经典技巧是除了最近的记录外还利用历史日t点的记录。由于一个周期(24小时)的长度非常长，以及随后的优化问题，这种类型的依赖关系很难被现成的循环单元捕获。受此技巧的启发，我们开发了一个具有时间跳跃连接的递归结构，以扩展信息流的时间跨度，从而简化优化过程。具体地，在当前的隐藏单元与相邻周期的同一相位的隐藏单元之间添加跳转链接。更新过程可以表述为：

$$
r_t = \sigma(x_tW_{xr}+h_{t-p}W_{hr}+b_r) \\\\
u_t = \sigma(x_tW_{xu}+h_{t-p}W_{hu}+b_u) \\\\
c_t = RELU(x_tW_{xc}+r_t \odot(h_{t-p}W_{hc})+b_c) \\\\
h_t = (1-u_t)\odot h_{t-p}+u_t\odot c_t
$$

其中，该层的输入为卷积层的输出，$p$为跳过的隐藏单元数。对于具有明确周期模式的数据集(例如，每小时电力消耗和交通使用数据集的$p = 24$)，可以很容易地确定$p$的值，否则必须进行调优。在我们的实验中，我们根据经验发现，即使是后一种情况，调优后的$p$也可以显著地提高模型的性能。此外，可以很容易地扩展LSTNet以包含跳跃长度$p$的变体。

我们使用一个稠密层来合并递归和递归跳跃组件的输出。稠密层的输入包括$t$时刻递归组件的隐藏状态，记为$h_t^R$, $t-p+1$时刻到$t$时刻的递归跳跃组件的$p$个隐藏状态，记为$h_{t-p+1}^S,h_{t-p+2}^S,\cdots,h_{t}^S$。密层的输出计算为:

$$
h_t^D = W^Rh_t^R + \sum_{i=0}^{p-1}W_i^Sh_{t-i}^S+b
$$

其中$h_t^D$是图2中上半部分神经网络在$t$时刻的预测结果。

### 3.5 时间注意力层

然而，递归跳跃层需要一个预定义的超参数$p$，这对于非季节性的时间序列预测是不利的，或者它的周期长度是随时间变化的。为了解决这个问题，我们考虑了另一种方法，即注意力机制[1]，它学习输入矩阵每个窗口位置的隐藏表征的加权组合。具体来说,在时刻$t$的关注力权重$\alpha_t \in \mathbb{R}^q$计算为：

$$
\alpha_t = AttnScore(H_t^R,h_{t-1}^R)
$$

其中$H_t^R = [ h_{t-q}^R,\cdots,h_{t-1}^R ]$是一个矩阵，它巧妙地将RNN的隐藏表征按列堆叠起来，而AttnScore是一些相似函数，比如点积、余弦或由一个简单的多层感知器参数化。

时间注意力层最终的输出是加权上下文向量$c_t=H_t\alpha_t$和最后窗口隐藏表征$h_{t-1}^R$的拼接，连上线性投影运算：

$$
h_t^D = W[c_t;h_{t-1}^R]+b
$$

### 3.6 自回归组件

由于卷积和递归组件的非线性特性，神经网络模型的一个主要缺点是输出的比例对输入的比例不敏感。遗憾的是，在具体的真实数据集中，输入信号的尺度会以非周期性的方式不断变化，这大大降低了神经网络模型的预测精度。在第4.6节中给出了一个失败的具体例子。为了解决这一缺陷，我们将LSTNet的最终预测分解为一个线性部分(主要关注局部尺度问题)和一个包含重复模式的非线性部分，这在本质上与公路网[29]类似。在LSTNet体系结构中，我们采用经典的自回归(AR)模型作为线性分量。$h_t^L \in \mathbb{R}^n$表示AR分量的预测结果, AR模型的系数为$W^{ar}\in \mathbb{R}^{q^{ar}}$ 和 $b^{ar}\in \mathbb{R}$，其中，$q^{ar}$是覆盖输入矩阵的输入窗口的大小。注意，在我们的模型中，所有维都共享相同的一组线性参数。AR模型表示如下:

$$
\tag{5}
h_{t,i}^L = \sum_{k=0}^{q^{ar}-1} W_k^{ar} y_{t-k,i}+b^{ar}
$$

将神经网络部分的输出与AR部分的输出进行积分，得到LSTNet的最终预测结果:

$$
\hat{Y}_t = h_t^D + h_t^L
$$

其中$\hat{Y}_t$为模型在时间戳$t$处的最终预测。

### 3.7 目标函数

平方差是许多预测任务的默认损失函数，其优化目标为:

$$
\tag{7}
\mathop{\mathrm{minimize}}_\Theta \sum_{t\in \Omega_{Train}} ||Y_t-\hat{Y}_{t-h}||_F^2
$$ 

其中 $\Theta$ 表示我们模型的参数设置，$\Omega_{Train}$ 是用于训练的时间戳集合，$||\cdot||_F$ 是 Frobenius norm,$h$是3.1节中提到的 horizon，传统的平方差损失函数线性回归模型称为 Linear Ridge,它等价于 ridge 正则化的向量自回归模型。然而，实验表明，在某些数据集中，线性支持向量回归(Linear Support Vector Regression, Linear SVR)[30]控制着 Linear Ridge 模型。Linear SVR与 Linear Ridge 的唯一区别在于目标函数。Linear SVR的目标函数为:

$$
\tag{8}
\mathop{\mathrm{minimize}}_\Theta \frac{1}{2}||\Theta||_F^2 + C \sum_{t\in  \Omega_{Train}} \sum_{i=0}^{n-1} \xi_{t,i} \\\\

subject \quad to \quad |\hat{Y}_{t-h,i} - Y_{t,i}| \le  \xi_{t,i} + \epsilon ,t\in  \Omega_{Train} \\\\
\xi_{t,i} \ge 0
$$

其中$C$和$\epsilon$是超参数。由于Linear SVR模型的卓越性能，我们将其目标函数加入到LSTNet模型中，作为平方损失的替代。为简单起见,我们假设 $\epsilon=0$,和上面的目标函数可以减少绝对损失(L1-loss)函数如下:

$$
\tag{9}
\mathop{\mathrm{minimize}}_\Theta \sum_{t\in  \Omega_{Train}} \sum_{i=0}^{n-1} |Y_{t,i}-\hat{Y}_{t-h,i}|
$$

绝对损失函数的优点是对实时序列数据中的异常具有较强的鲁棒性。在实验部分，我们使用验证集来决定使用哪一个目标函数，是 Eq.7 的平方损失还是 Eq.9 的绝对损失。

### 3.8 优化策略

本文的优化策略与传统的时间序列预测模型相同。假设输入时间序列为$Y_t = \{y_1,y_2,\cdots,y_t\}$,我们定义了一个大小为$q$的可调窗口，并将时间戳$t$的输入重新表示为$y_{t-q+1},y_{t-q+2},\cdots,y_t$。然后，该问题就变成了一个具有一组 feature-value pairs $X_t,Y_{t+h}$的回归任务，并且可以通过随机梯度下降(SGD)或其变体(如Adam[18])来解决。

## 4. 评估

我们用9种方法(包括我们的新方法)在4个基准数据集上对时间序列预测任务进行了广泛的实验。所有的数据和实验代码都可以在网上找到。

### 4.1 方法比较

我们的比较评价方法如下。

* AR代表自回归模型，与一维VAR模型等价。
* LRidge是L2正则化的向量自回归(VAR)模型，在多元时间序列预测中应用最为广泛。
* LSVR是支持向量回归目标函数[30]的向量自回归(VAR)模型。
* TRMF是使用[32]进行时间正则化矩阵分解的自回归模型。
* GP是时间序列建模的高斯过程。[11, 28]
* VAR-MLP是[35]中提出的多层感知机(Multilayer Perception, MLP)与自回归模型相结合的模型。
* RNN-GRU是使用GRU单元的递归神经网络模型。
* LSTNet-skip是我们提出的具有skip-RNN层的LSTNet模型。
* LSTNet-attn是我们提出的具有时间注意层的LSTNet模型。

对于上述的AR、LRidge、LSVR、GP等单一输出方法，我们只训练了n个单独模型，即: n个输出变量各有一个模型。

### 4.2 度量

我们使用了三个传统的评估指标，定义为:

* Root Relative Squared Error (RSE):

$$
\tag{10}
RSE = \frac{\sqrt{\sum_{(i,t)\in \Omega_{Test}} (Y_{it}-\hat{Y}_{it})^2 }}{\sqrt{\sum_{(i,t)\in \Omega_{Test}} (Y_{it}-mean(Y))^2 }}
$$

* Empirical Correlation Coefficient (经验相关系数,CORR) 

$$
CORR = \frac{1}{n} \sum_{i=1}^n \frac{\sum_t(Y_{it}-mean(Y_i))(\hat{Y}_{it}-mean(\hat{Y}_i))}{\sqrt{\sum_t(Y_{it}-mean(Y_i))^2(\hat{Y}_{it}-mean(\hat{Y}_i))^2}}
$$

其中$Y,\hat{Y} \in \mathbb{R}^{n\times T}$分别为 ground true signals 和系统预测信号。RSE是广泛使用的均方根误差(RMSE)的 scaled 版本，它的设计是为了使评估更可读，而不管数据的 scale。RSE越低越好，CORR越高越好。

### 4.3 Data

![表1](/assets/images/深度学习/用深度神经网络建模长短期时间模式/tab1.png)

表1：数据集统计，其中T为时间序列的长度，D为变量的数量，L为采样率。

我们使用了四个公开的基准数据集。表1汇总了语料库统计数据。

* Traffic:收集加州运输部48个月(2015-2016)每小时的数据。数据描述了在旧金山湾区高速公路上由不同传感器测量的道路占有率(0 ~ 1)。
* Solar-Energy:2006年太阳能发电量记录，每10分钟采样一次，来自阿拉巴马州的137个光伏电站。
* Electricity(电费):从2012年到2014年，有321个客户每15分钟就有电费记录。我们把数据转换成每小时的消耗量;
* Exchange-Rate(汇率):收集澳大利亚、英国、加拿大、瑞士、中国、日本、新西兰、新加坡等8个国家从1990年到2016年的每日汇率。

所有数据集按时间顺序分为训练集(60%)、验证集(20%)和测试集(20%)。为了方便未来多元时间序列预测的研究，我们在网站上公布了所有的原始数据集和预处理后的数据集。

![图3](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig3.png)

图3：从4个数据中采样变量的自相关图。（form应该是笔误）

为了检验时间序列数据中是否存在长期和/或短期的重复模式，我们从图3的四个数据集中随机选取了一些变量，绘制了自相关图。自相关，也称为序列相关，是信号与自身的延迟副本的相关性，作为延迟的函数，定义如下：

$$
R(\tau) = \frac{E[(X_t-\mu)(X_{t+\tau}-\mu)]}{\sigma^2}
$$

其中$X_t$是时间序列信号，$\mu$是均值，$\sigma^2$是方差，在实际应用中，我们考虑了经验无偏估计量来计算自相关性。

从图3的(a)、(b)、(c)三个图中可以看出，在Traffic(交通)、Solar-Energy(太阳能)、Electricity(电力)数据集中存在高度自相关的重复模式，而在汇率数据集中则没有。此外，我们可以在交通和电力数据集的图中观察到一个短期的日模式(每24小时)和长期的周模式(每7天)，这完美地反映了高速公路交通状况和电力消耗的变化规律。另一方面，在汇率数据集的图(d)中，我们几乎看不到任何重复的长期模式，只能看到一些短期的局部连续性。这些观察结果对我们以后分析不同方法的实证结果有重要意义。也就是说，对于那些能够正确建模并取得成功的方法——充分利用数据中的短期和长期重复模式，当数据包含这些重复模式(如电力、交通和太阳能)时，它们的表现应该会更好。另一方面，如果数据集不包含这些模式(比如汇率)，那么这些方法的优势功能可能不会比其他功能较弱的方法带来更好的性能。我们将在第4.7节中以经验的理由重新讨论这一点。

### 4.4 实验的细节

我们对每个方法和数据集的所有可调超参数进行网格搜索。具体来说，所有方法共享窗口大小为$q$的相同网格搜索范围，范围从$\{2^0,2^1,\cdots,2^9\}$。ForLRidgeandLSVR,对 LRidge 和 LSVR,正则化系数$\lambda$从$\{2^{-10},2^{-8},\cdots,2^{8},2^{10}\}$中选择。GP,RBF核函数 bandwidth $\sigma$ 和 noise level $\alpha$ 从$\{2^{-10},2^{-8},\cdots,2^{8},2^{10}\}$中选择，对$TRMF$，隐藏维度从$\{2^2,\cdots,2^6\}$中选择，正则化系数$\lambda$从$\{0.1,1,10\}$转给你选择。对于LST-Skip和LST-Attn，我们采用了第3.8节中描述的训练策略。对于递归跳越层，在$\{50,100,200\}$和$\{20,50,100\}$中选择递归和卷积层的隐藏维度。递归跳跃层的跳跃长度$p$在交通和电力数据集上设置为$24$，太阳能和汇率数据集调整为$2^1$到$2^6$。AR分量的正则化系数从$\{0.1,1,10\}$中选取，以达到最佳性能。除了输入和输出层之外，我们在每个层之后执行dropout，rate通常设置为0.1或0.2。利用Adam[18]算法对模型参数进行优化。

## 4.5 主要结果

![表2](/assets/images/深度学习/用深度神经网络建模长短期时间模式/tab2.png)

表2：四个数据集上所有方法的结果汇总(RSE和CORR):1)每一行有一个特定方法在特定度量上的结果;2)每一列将一个特定数据集上一个特定的horizon值的所有方法的结果进行比较;3)黑体表示某一特定度量中每一列的最佳结果;4)每个方法的粗体结果总数列在方法名称下的括号内。

表2总结了所有方法在所有测试集(4)和所有的指标(3)上的评价结果(8)。我们设置$horizon=\{3,6,12,24\}$,分别,这意味着 horizons 是3到24小时用来预测电力和交通数据,从30到240分钟用在太阳能数据上,从3到24天用在汇率数据。horizons(视野)越大，预测任务越困难。每个(数据、度量)对的最佳结果在此表中以粗体突出显示。LSTNet-skip(LSTNet的一个版本)的粗体结果总数为17,LSTNet-attn(LSTNet的另一个版本)的粗体结果总数为7，其余方法的粗体结果总数为0到3。

显然，这两个被提议的模型，LSTNet-skip和LSTNet-Attn，在具有周期模式的数据集上，特别是在大 horizons 的设置上，持续提高了最优结果。此外，LSTNet在 horizon 为24时，在太阳能、交通和电力数据集上的RSE度量分别比强神经基线RNN-GRU高9.2%、11.7%和22.2%，证明了该框架设计对于复杂重复模式的有效性。更重要的是，当周期模式$q$在应用程序中不清楚时，用户可以考虑使用LSTNet-attn替代LSTNet-skip，因为前者仍然比基线有相当大的改进。但是提议的LSTNet在汇率数据集上比AR和LRidge略差。为什么?回想一下，在4.3节和图3中，我们使用这些数据集的自相关曲线来显示在太阳能、交通和电力数据集中重复模式的存在，而在汇率数据集中没有重复模式。当前的结果为LSTNet模型在对数据中出现的长期和短期依赖模式建模方面的成功提供了经验证据。另外，LSTNet在反映基线中与较优的AR和LRidge表现相当。

相比单变量的结果AR与多变量基线方法(LRidge, LSVR和RNN),我们看到,在一些数据集,例如太阳能和交通,多元方法更强,否则更弱,这意味着在传统的多变量方法上丰富的输入信息将导致过度拟合。相反，LSTNet在不同的情况下具有健壮的性能，部分原因是它的自回归组件，我们将在第4.6节进一步讨论。

### 4.6 消融研究

为了证明我们的框架设计的有效性，我们进行了一个仔细的消融研究。具体来说，我们在LSTNet框架中每次删除一个组件。首先，我们将缺少不同组件的LSTNet命名如下。

* LSTw/oskip: LSTNet模型不包含 递归跳跃组件和注意力组件
* LSTw/oCNN: LSTNet-skip模型不包括卷积组件
* LSTw/oAR: LSTNet-skip模型不包括AR组件

对于不同的基线，我们调整模型的隐藏维度，使它们具有与完整的LSTNet模型相似的模型参数数量，从而消除模型复杂性带来的性能增益。

![图5](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig5.png)

图5：LSTNet在太阳能、交通和电力数据集消融试验中的结果

使用RSE和CORR测量的测试结果如图5所示。这些结果的几个观察结果值得强调:

* 使用LST-Skip或LST-attn可获得每个数据集上的最佳结果。
* 从整个模型中删除AR组件(在LSTw/oAR中)导致大多数数据集的性能显著下降，显示了AR组件的关键作用。
* 移除Skip和CNN组件 (LSTw/oCNN 或 LSTw/oskip)导致一些数据集的性能大幅下降，但不是所有数据集。LSTNet的所有组件一起导致了我们的方法在所有数据集上的健壮性能。

结论是，我们的架构设计在所有的实验环境中都是最健壮的，特别是在大 horizons 上。

![图6](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig6.png)

图6：LSTw/oAR(a)和LST-Skip(b)的预测时间序列(红色) vs $horizon=24$的电力数据集上的真实数据(蓝色)

至于为什么AR组件会有如此重要的作用，我们的解释是AR通常对数据的规模变化是健壮的。为了从经验上验证这种直觉，我们在图6中绘制了1到5000小时的电力消耗数据集中时间序列信号的一维(一个变量)，其中蓝色曲线是真实数据，红色曲线是系统预测信号。我们可以看到真正的消耗在第1000个小时左右突然增加，LSTNet-Skip成功地捕捉到了这种突然的变化，但是LSTw/oAR没有做出适当的反应。

为了更好地验证这一假设，我们进行了仿真实验。首先，我们随机生成一个规模变化的自回归过程，步骤如下。首先，我们随机采样一个向量，$w\sim N(0,I),w\in \mathbb{R}^p$,其中$p$是一个给定的窗口大小，然后生成自回归过程$x_t$,用下式描述：

$$
\tag{12}
x_t = \sum_{i=1}^p w_i x_{t-i} + \epsilon
$$

其中$\epsilon \sim N(\mu,1)$,注入规模变化,我们每$T$时间戳增加高斯噪声的均值$\mu_0$。那么时间序列$x_t$的高斯噪声可以表示为:

$$
\epsilon \sim N(\lfloor t/T \rfloor \mu_0,1)
$$

其中$\lfloor \cdot \rfloor$表示 floor 函数。我们将时间序列按时间顺序分为训练集和测试集，并测试了RNN-GRU和LSTNet模型。结果如图4所示。RNN和LSTNet都可以记忆训练集(左侧)中的模式。但是，RNN-GRU模型不能遵循测试集(右侧)中的尺度变化模式。相反，LSTNet模型更适合测试集。换句话说,正常RNN模块,或说,在LSTNet中的神经网络组件,对违反规模波动数据(电力数据就是一种典型，可能由于公共假期或气温扰动等随机事件,等等)可能不是足够敏感数据,而简单的线性AR模型可以做出适当的预测调整。

![图4](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig4.png)

图4：模拟测试:左侧为训练集，右侧为测试集。

总之，这个消融研究清楚地证明了我们架构设计的效率。所有组件都为LSTNet的卓越和健壮性能做出了贡献。

### 4.7 长期和短期模式的混合

![图7](/assets/images/深度学习/用深度神经网络建模长短期时间模式/fig7.png)

图7：对于 Traffic occupation 数据集中的一个变量，VAR(a)和 LSTNet(b)给出的真实时间序列(蓝色)和预测时间序列(红色)。X轴表示工作日，预测的$horizon=24$。VAR不能很好地预测星期五和星期六的相似模式，以及星期日和星期一的相似模式，而LSTNet成功地捕获了每天和每周重复的模式。

为了说明LSTNet在对时间序列数据中短期和长期重复模式的混合建模方面的成功，图7比较了LSTNet和VAR在 Traffic 数据集中特定时间序列(输出变量之一)上的性能。如4.3节所述，交通数据呈现出两种重复模式，即日重复模式和周重复模式。从图7中我们可以看到，周五和周六的交通占用率的真实模式(用蓝色表示)非常不同，而周日和周一则完全不同。图7是VAR模型的预测结果((a)部分)和LSTNet ((b)部分)的交通流量监测传感器,他们根据RMSE结果验证集选择超参数。图中显示的VAR模型只能处理短期模式。VAR模型的预测结果模式只依赖于预测前一天。我们可以清楚地看到，其在周六(第2、9个峰值)和周一(第4、11个峰值)的结果与ground truth不同，其中周一(工作日)的ground truth有两个峰值，周六(周末)一个峰值。相反，我们提出的LSTNet模型在工作日和周末有两种模式。这个例子证明了LSTNet模型同时记忆短期和长期重复模式的能力，这是传统预测模型所不具备的，它在真实世界时间序列信号的预测任务中是至关重要的。

## 5. 结论

针对多元时间序列预测问题，提出了一种新的深度学习框架(LST-Net)。该方法结合了卷积神经网络和递归神经网络的优点以及自回归分量，大大提高了基于多基准数据集的时间序列预测的精度。通过深入的分析和经验证据，我们证明了LSTNet模型体系结构的有效性，它确实成功地捕获了数据中的短期和长期重复模式，并将线性和非线性模型结合起来进行稳健预测。
对于未来的研究，有几个有希望的方向扩展工作。首先，跳跃循环层的跳跃长度$p$是一个关键的超参数。目前，我们根据验证数据集手动调整它。如何根据数据自动选择$p$是一个有趣的问题。其次，在卷积层中我们对每个变量维数都一视同仁，但在现实世界的数据集中，我们通常有丰富的属性信息。将它们集成到LSTNet模型中是另一个具有挑战性的问题。

## 引用

1. D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.
2. G. E. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015.
3. G. E. Box and D. A. Pierce. Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. Journal of the American statistical Association, 65(332):1509–1526, 1970.
4. L.-J. Cao and F. E. H. Tay. Support vector machine with adaptive parameters in financial time series forecasting. IEEE Transactions on neural networks, 14(6):1506– 1518, 2003.
5. Z.Che,S.Purushotham,K.Cho,D.Sontag,andY.Liu.Recurrentneuralnetworks for multivariate time series with missing values. arXiv preprint arXiv:1606.01865, 2016.
6. J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.
7. J. Connor, L. E. Atlas, and D. R. Martin. Recurrent networks and narma modeling. In NIPS, pages 301–308, 1991.
8. S. Dasgupta and T. Osogami. Nonlinear dynamic boltzmann machines for time- series prediction. AAAI-17. Extended research report available at goo. gl/Vd0wna, 2016.
9. J. L. Elman. Finding structure in time. Cognitive science, 14(2):179–211, 1990.
10. R. Frigola, F. Lindsten, T. B. Schön, and C. E. Rasmussen. Bayesian inference and learning in gaussian process state-space models with particle mcmc. In Advances in Neural Information Processing Systems, pages 3156–3164, 2013.
11. R. Frigola-Alcade. Bayesian Time Series Learning with Gaussian Processes. PhD thesis, PhD thesis, University of Cambridge, 2015.
12. J. D. Hamilton. Time series analysis, volume 2. Princeton university press Princeton, 1994.
13. N. Y. Hammerla, S. Halloran, and T. Ploetz. Deep, convolutional, and recurrent models for human activity recognition using wearables. arXiv preprint arXiv:1604.08880, 2016.
14. G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine, 29(6):82–97, 2012.
15. S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
16. A. Jain and A. M. Kumar. Hybrid neural network models for hydrologic time series forecasting. Applied Soft Computing, 7(2):585–592, 2007.
17. K.-j. Kim. Financial time series forecasting using support vector machines. Neurocomputing, 55(1):307–319, 2003.
18. D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
19. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.
20. C. Lea, R. Vidal, A. Reiter, and G. D. Hager. Temporal convolutional networks: A unified approach to action segmentation. In Computer Vision–ECCV 2016 Workshops, pages 47–54. Springer, 2016.
21. Y. LeCun and Y. Bengio. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10):1995, 1995.
22. J. Li and W. Chen. Forecasting macroeconomic time series: Lasso-based ap- proaches and their forecast combinations with dynamic factor models. Interna- tional Journal of Forecasting, 30(4):996–1015, 2014.
23. Z. C. Lipton, D. C. Kale, C. Elkan, and R. Wetzell. Learning to diagnose with lstm recurrent neural networks. arXiv preprint arXiv:1511.03677, 2015.
24. H. Lütkepohl. New introduction to multiple time series analysis. Springer Science & Business Media, 2005.
25. E. McKenzie. General exponential smoothing and the equivalent arma process. Journal of Forecasting, 3(3):333–344, 1984.
26. I. Melnyk and A. Banerjee. Estimating structured vector autoregressive model. arXiv preprint arXiv:1602.06606, 2016.
27. H. Qiu, S. Xu, F. Han, H. Liu, and B. Caffo. Robust estimation of transition matrices in high dimensional heavy-tailed vector autoregressive processes. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pages 1843–1851, 2015.
28. S. Roberts, M. Osborne, M. Ebden, S. Reece, N. Gibson, and S. Aigrain. Gaussian processes for time-series modelling. Phil. Trans. R. Soc. A, 371(1984):20110550, 2013.
29. R. K. Srivastava, K. Greff, and J. Schmidhuber. Highway networks. arXiv preprint arXiv:1505.00387, 2015.
30. V. Vapnik, S. E. Golowich, A. Smola, et al. Support vector method for function approximation, regression estimation, and signal processing. Advances in neural information processing systems, pages 281–287, 1997.
31. J. B. Yang, M. N. Nguyen, P. P. San, X. L. Li, and S. Krishnaswamy. Deep convo- lutional neural networks on multichannel time series for human activity recog- nition. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), Buenos Aires, Argentina, pages 25–31, 2015.
32. H.-F. Yu, N. Rao, and I. S. Dhillon. Temporal regularized matrix factorization for high-dimensional time series prediction. In Advances in Neural Information Processing Systems, pages 847–855, 2016.
33. R. Yu, Y. Li, C. Shahabi, U. Demiryurek, and Y. Liu. Deep learning: A generic approach for extreme condition traffic forecasting. In Proceedings of the 2017 SIAM International Conference on Data Mining, pages 777–785. SIAM, 2017.
34. G.Zhang,B.E.Patuwo,andM.Y.Hu.Forecastingwithartificialneuralnetworks:: The state of the art. International journal of forecasting, 14(1):35–62, 1998.
35. G. P. Zhang. Time series forecasting using a hybrid arima and neural network model. Neurocomputing, 50:159–175, 2003.

---
**参考**：
1. 论文：Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/abs/1703.07015)
