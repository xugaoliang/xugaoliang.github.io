---
typora-root-url: ../../../
---
## GE2E 分享

* 作者： Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno
* 论文：《Generalized End-to-End Loss for Speaker Verification》
* 地址：https://arxiv.org/abs/1710.10467

## 个人总结：

本文有2个贡献：

1. GE2E（广义端到端）损失
   
   用于改进基于 tuple 的端到端损失（TE2E）。两者的不同在于，TE2E的损失中一个样本要么是正例，要么是负例。而GE2E的损失中一个样本要同时与1个正例和多个负例一起做计算。因此GE2E能更快地收敛，而且效果也更好。个人感觉这与 Kihyuk Sohn 提出的 Multi-class N-Pair Loss 很相似（见[Improved Deep Metric Learning with
Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf) ）

2. MultiReader 技术
   
   MultiReader 用来组合不同的数据源，相当于在一个数据源$D_1$的模型上加入了正则化项，而正则的对象不是模型参数，而是其他数据源上的损失。这样做的目的是，如果$D_1$数据量小，容易过拟合，加入其他数据源进行正则化。这可以扩展到多数据源上，不同数据源使用不同权重。如下：

   $$
   L(D_1,\cdots,D_K)= \sum_{k=1}^K \alpha_k \mathbb{E}_{x_k \in D_k} [L(x_k ; w)]
   $$

论文中提到 text-dependent（文本相关） speaker verification (TD-SV) and text-independent（文本无关） speaker verification (TI-SV). 他们具体的含义我还不是很清楚。

可以作为借鉴的是定义正负例损失时，可以一个样本同时计算正例和多个负例的损失，加速收敛和提高效果，需要注意的一个细节是，每个负例的算是都和正例结合计算，以此平衡负例和正例个数不均衡的问题。另一个借鉴之处是 MultiReader 技术，不过这个技术目前也许还用不到。


## Tuple-Based End-to-End Loss

在我们之前的工作[13]中，我们提出了基于tuple的端到端(end-to-end, TE2E)模型，该模型模拟了训练期间运行时注册和验证的两阶段过程。在我们的实验中，与LSTM[14]相结合的TE2E模型获得了当时最好的性能。对于每个训练步骤，一个评估话语$x_{j\sim}$和$M$个注册话语$x_{km}$($m=1,\cdots,M$)的 tuple 被输入到我们的LSTM网络:$\{x_{j\sim},(x_{k1},\cdots,x_{km})\}$,$x$代表固定长度段的特征(log-mel-filterbank energies),$j$和$k$代表话语的speakers,$j$不一定等于$k$。如果$x_{j\sim}$和$M$个注册话语来自相同的speaker,我们称这个tuple为正例，即$j=k$,否则为负例。我们交替生成正 tuple 和负 tuple。

对于每个输入的 tuple，我们计算 LSTM 的 L2 normalized 响应:$\{e_{j\sim},(e_{k1},\cdots,e_{kM})\}$。这里每个$e$是一个固定维数的embedding向量，它是由LSTM定义的 sequence-to-vector 的映射产生的。tuple $(e_{k1},\cdots,e_{kM})$ 的质心表示由$M$个话语构建的 voiceprint(声纹)，定义如下:

$$
\tag{1}
c_k = \mathbb{E}_m[e_{km}] = \frac{1}{M} \sum_{m=1}^M e_{km}
$$

相似度采用余弦相似度函数定义:

$$
\tag{2}
s = w\cdot \cos(e_{j\sim},c_k) + b
$$

可学习参数$w$和$b$,TE2E损失最终定义为：

$$
\tag{3}
L_T (e_{j\sim},c_k) = \delta(j,k) \sigma(s) + (1-\delta(j,k))(1-\sigma(s))
$$

这里$\sigma(x)=1/(1+e^{-x})$ 为标准 sigmoid 函数, 如果 $j=k$, 则 $\sigma(j,k)=1$，否则等于0，当$k=j$时，TE2E损失函数鼓励$s$的值更大，当$k\ne j$时，则$s$的值更小。考虑到正负 tuples 的更新,这损失函数非常类似 FaceNet 中的 triplet loss[15]。


## GENERALIZED END-TO-END MODEL

广义端到端(GE2E)训练是基于一次处理大量的话语，其形式是一个 batch 包含$N$个 speaker ，平均每个 speak 有$M$个话语，如图1所示。

![图1](/assets/images/nlp/ge2e/fig1.png)

图1：系统概述。不同的颜色表示不同 speaker 的 utterances/embedding。

### 训练方法

我们获取$N\times M$个话语来构建一个 batch。这些话语来自$N$个不同的 speakers，每个 speaker 有$M$个话语。每个特征向量$x_{ji}(1\le j\le N，1\le i \le M)$表示从 speaker $j$ 的话语$i$中提取的特征。

与我们之前的工作[13]类似，我们将每个话语$x_{ji}$中提取的特征输入到 LSTM 网络中。一个线性层连接到最后一个LSTM层，作为对网络最后一帧响应的附加变换。我们将整个神经网络的输出表示为$f(x_{ji};w)$，其中$w$表示神经网络的所有参数(包括LSTM层和线性层)。定义嵌入向量(d-vector)为网络输出的 L2 normalization:

$$
\tag{4}
e_{ji} = \frac{f(x_{ji};W)}{||f(x_{ji};W)||_2}
$$

这里$e_{ji}$表示第$j$个 speaker 的第$i$个话语的嵌入向量。第$j$个 speaker 的嵌入向量$[e_{j1}\cdots,e_{jM}]$的质心$c_j$由式(1)定义。

相似矩阵$S_{ji,k}$定义为每个嵌入向量$e_{ji}$到所有质心 $c_k (1\le j,k \le N,1 \le i \le M)$ 之间的 scaled cosine similarities:

$$
\tag{5}
S_{ji,k} = w\cdot cos(e_{ji},c_k)+b
$$

其中$w,b$是可学习参数。我们把权重限制在$w>0$，因为当余弦相似度较大时，我们希望相似度较大。TE2E和GE2E的主要区别如下:

* TE2E的相似度(方程2)是一个标量值，它定义了嵌入向量$e_{j\sim}$和单个 tuple 质心$c_k$之间的相似度。
* GE2E建立了一个相似矩阵(方程5)，定义了每个$e_{ji}$和所有质心$c_k$之间的相似度。

图1用特征展示了整个过程，嵌入向量，以及来自不同 speakers 的相似度评分，用不同的颜色表示。

![图2](/assets/images/nlp/ge2e/fig2.png)

图2：GE2E损失将嵌入推向真正 speaker 的质心，而远离最相似的不同 speaker 的质心。

在训练中，我们希望每个话语的嵌入与对应speaker的所有嵌入的质心相似，同时远离其他speaker的质心。如图1中的相似矩阵所示，我们希望彩色区域的相似值较大，而灰色区域的相似值较小。图2以另一种方式说明了相同的概念:我们希望蓝色嵌入向量接近它自己的speaker的质心(蓝色三角形)，远离其他质心(红色和紫色三角形)，特别是最接近的那个(红色三角形)。给定一个嵌入向量$e_{ji}$，所有的质心$c_k$，以及相应的相似矩阵$S_{ji,k}$,有两种方法来实现这个概念:

**Softmax** 我们在$S_{ji,k}$上放一个softmax,$k = 1,\cdots,N$,如果$k=j$,则使得输出等于1，否则输出等于0。因此，每个嵌入向量$e_{ji}$的损失可定义为:

$$
\tag{6}
L(e_{ji}) = -S_{ji,j} + \log \sum_{k=1}^N exp(S_{ji,k})
$$

这个损失函数意味着我们将每个嵌入向量推到其质心附近，并将其从所有其他质心拉出。

**Contrast** 对比度损失定义为正例对和最积极的负例对上，如:

$$
\tag{7}
L(e_{ja \\\\
bi}) = 1- \sigma(S_{ji,j}) + \max_{
    1\le k \le N, k\ne j
} \sigma(S_{ji,k})
$$

其中$\sigma(x)=1/(1+e^{-x})$ 为 sigmoid 函数。对于每个话语，刚好增加了两个分量的损失:(1)一个正分量，它与嵌入向量及其真正 speaker 的声纹(质心)之间的正匹配相关。(2) hard 负分量，与嵌入向量及所有假speaker中相似性最高的speaker的声纹(质心)之间的负匹配有关。

在图2中，正项对应于将$e_{ji}$(蓝色圆圈)推向$c_j$(蓝三角形)。负项对应于将$e_{ji}$(蓝色圆圈)从$c_k$(红三角形)中拉出来，因为$c_k$与$c_k'$相比更接近于$e_{ji}$。因此，对比度损失使我们能够专注于困难的嵌入向量和负质心对。

在我们的实验中，我们发现GE2E损失的两种实现都是有用的:对比损耗在 TD-SV 中表现更好，而softmax损失在 TI-SV 中表现更好。

此外，我们还观察到，在计算真正 speaker 的质心时去掉$e_{ji}$可以使训练变得稳定，并有助于避免 trivial(琐碎) 解。因此，当我们计算负相似度(即$k\ne j$)时，我们仍然使用公式1，当$k=j$时,我们使用公式8:

$$
\tag{8}
c_j^{(-i)} = \frac{1}{M-1} \sum_{m=1,m\ne i}^M e_{jm}
$$

$$
\tag{9}
S_{ji,k} = 
\begin{cases}
    w\cdot \cos(e_{ji},c_j^{(-i)}) +b \quad if \quad k=i; \\\\
    w\cdot \cos(e_{ji},c_k) + b \quad otherwise
\end{cases}
$$

结合公式4、6、7、9，最终的 GE2E 损失 $L_G$ 为相似矩阵($1\le j \le N,1\le i \le M$)上所有损失之和:

$$
\tag{10}
L_G(x;w) = L_G(S) = \sum_{j,i} L(e_{ji})
$$

## Training with MultiReader

考虑以下情况:我们关心具有小数据集$D_1$的域中的模型应用。同时，我们有一个更大的类似的数据集$D_2$，但不是相同的领域。我们想要在$D_2$的帮助下，训练一个在$D_1$数据集上表现良好的模型:

$$
\tag{12}
L(D_1,D_2;w) = \mathbb{E}_{x\in D_1} [L(x;w)] + \alpha \mathbb{E}_{x\in D_2}[L(x;w)]
$$

这类似于正则化技术:在正常的正则化中,我们使用$\alpha ||w||_2^2$正则化模型。但是这里，我们使用$\mathbb{E}_{x\in D_2}[L(x;w)]$来正则化。当数据集$D_1$没有足够的数据时，在$D_1$上训练网络会导致过拟合。要求网络在$D_2$上也能正常运行，这有助于网络的正则化。

这可以推广到结合$K$个不同的，可能极度不平衡的数据源:$D_1,\cdots,D_K$。我们给每个数据源指定一个权重$\alpha_k$,代表数据源的重要性。在训练期间,在每一步中，我们从每个数据源取一个 batch/tuple 的话语,并计算损失和为:$L(D_1,\cdots,D_K)= \sum_{k=1}^K \alpha_k \mathbb{E}_{x_k \in D_k} [L(x_k ; w)]$,其中每个$L(x_k ; w)$为式10中定义的损失。


## Text-Dependent Speaker Verification

虽然现有的语音助手通常只支持一个 keyword，但研究表明，用户更希望同时支持多个 keywords。对于谷歌Home的多用户，同时支持两个 keywords:“OK Google”和“Hey Google”。

启用多个keywords 的 SV 是 TD-SV 和 TI-SV 之间的问题，因为 transcript 既不受单个短语的约束，也不完全不受约束。我们使用 MultiReader 技术解决了这个问题(第2.3节)。与更简单的方法(如直接混合多个数据源)相比，MultiReader有很大的优势:它可以处理不同数据源大小不平衡的情况。在我们的例子中，我们有两个用于训练的数据源:1)一个“OK Google”训练集，来自匿名化的用户查询，其中有$\sim 150$ M 的话语和 $\sim 630$ K的 speaker;2)一个混合的“OK/Hey google”的手动收集的训练集，它用$\sim 1.2$ M的话语和 $\sim 18$ k的speakers。第一个数据集比第二个数据集大125倍话语，和speaker 大35倍。

![表1](/assets/images/nlp/ge2e/tab1.png)

为了评估，我们报告了四种情况的 Equal Error Rate (EER):用任一 keyword 注册，并验证任一关键字。所有的评价数据集都是从665个 speaker 中手动收集的，平均每个 speaker 有4.5个登记话语和10个评价话语。结果如表1所示。正如我们所看到的，MultiReader 在所有四种情况下带来了大约30%的相对改善。

![表2](/assets/images/nlp/ge2e/tab2.png)

我们还从匿名日志和手工收集的 $\sim 83$ K不同的 speakers 和环境条件中收集了一个更大的数据集，对其进行了更全面的评估。我们平均使用7.3条登记话语和5条评价话语。表2总结了使用和不使用 MultiReader 设置训练的不同损失函数的平均 EER。基线模型是一个单层的LSTM，有512个节点，嵌入向量大小为128[13]。第二行和第三行的模型结构是3层LSTM。比较第二行和第三行，我们可以看到 GE2E 比 TE2E 大约好10%。与表1类似，在这里我们还可以看到模型在使用 MultiReader 时的性能明显更好。虽然表中没有显示，但值得注意的是 GE2E 模型比 TE2E 少了大约60%的训练时间。

## Text-Independent Speaker Verification

![图3](/assets/images/nlp/ge2e/fig3.png)

图3：训练TI-SV模型的批建造过程。

在 TI-SV 训练中，我们将训练话语分成更小的片段，我们称之为 partial uttenrances (部分话语)。虽然我们不要求所有的部分话语都具有相同的长度，但同一批次的所有部分话语必须具有相同的长度。因此，对于每批数据，我们在$[lb, ub] =[140, 180]$帧内随机选择一个时间长度$t$，并强制该批数据中所有的部分话语都是长度$t$(如图3所示)。

![图4](/assets/images/nlp/ge2e/fig4.png)

图4：用于TI-SV的滑动窗口。

在推理期间，我们对每个话语应用一个固定大小的滑动窗口$(lb + ub)/2 = 160$帧，重叠50%。我们计算每个窗口的 d-vector。按窗口的d-vectors 进行 L2 normalizing，然后按元素取平均值，生成最终的按话语的 d-vectors (如图4所示)。

我们的TI-SV模型从匿名日志中提取了18K个 speaker 发出的约36M个语音。对于评估，我们使用了额外的1000个speaker，平均每个speaker有6.3条注册话语和7.2条评价话语。表三给出了不同训练损失函数的性能比较。第一列是softmax，用于预测训练数据中所有speaker的 speaker 标签。第二列是用TE2E损失训练的模型。第三列是用GE2E损失训练的模型。如表所示，GE2E的性能比softmax和TE2E都好。EER提高10%以上。此外，我们还观察到GE2E训练比其他损失函数快3倍左右。

## 结论

在本文中，我们提出了广义端到端(GE2E)损失函数来更有效地训练SV模型。理论和实验结果都验证了该损失函数的优越性。我们还引入了MultiReader技术来组合不同的数据源，使我们的模型能够支持多个关键字和多种语言。通过结合这两种技术，我们得到了更精确的SV模型。