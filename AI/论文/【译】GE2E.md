# 【译】用于speaker验证的广义的端到端损失.md

* 作者： Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno
* 论文：《Generalized End-to-End Loss for Speaker Verification》
* 地址：https://arxiv.org/abs/1710.10467

---
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

---

## 摘要

在本文中，我们提出了一种新的损失函数，称为广义端到端(generalized end-to-end,GE2E)损失，这使 speaker verification 模型训练比我们以前的基于tuple的端到端损失函数（tuple-based end-to-end,TE2E）更高效。与TE2E不同, GE2E 损失函数以强调训练过程每一步样本的不同的方法更新网络。另外，GE2E损失不需要样本选择的初始阶段。由于这些属性，我们用新损失的模型较少了超过10%的 speaker verification EER，同时减少60%训练时间。我们也介绍了 MultiReader 技术，这允许我们做 domain adaptation —— 训练更精准的模型以支持多关键词（如 "OK Google" 和 "Hey Google"）以及多方言。

Index Terms— Speaker verification, end-to-end loss, Multi- Reader, keyword detection

## 1. 介绍

### 1.1 背景

Speaker verification (SV) 是根据 speaker 的已知话语（即，注册话语，enrollment utterances ），验证一段话语是否属于这个特定的speaker,应用如 Voice Match[1,2]

根据话语应用于注册和验证的限制，SV 模型通常分为两种： text-dependent（文本相关） speaker verification (TD-SV) and text-independent（文本无关） speaker verification (TI-SV). 在TD-SV中，注册和验证话语的文字记录都受到 phonetically 限制，而在TI-SV中，注册或验证话语的文字记录不受词汇的约束，表现为音素和话语时长有较大的变异性[3,4]。在这项工作中，我们主要关注TI-SV，和TD-SV的一个特殊子任务global password TD-SV，其中验证是基于一个检测到的关键字，例如“OK Google”[5,6]。

在以往的研究中，基于 i-vector 的系统已经成为 TD-SV 和 TI-SV 应用[7]的领域方法。近年来，更多的努力集中在使用神经网络工作的 SV，而最成功的系统使用端到端训练[8,9,10,11,12]。在这样的系统中，神经网络输出向量通常被称为嵌入向量(也称为 d-vectors)。与 i-vector 的情况类似，这种嵌入可在固定维度空间中用于表示话语，在这种空间中，可以使用其他通常更简单的方法来消除 speakers 之间的歧义。

### 1.2 Tuple-Based End-to-End Loss

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

### 1.3 Overview

在本文中，我们介绍了我们的TE2E体系结构的一个推广。这种新的体系结构以一种更有效的方式从可变长度的输入序列构建元组，从而显著提高了 TD-SV 和 TI-SV 的性能和训练速度。本文组织如下:在第2.1节中给出了 GE2E loss 的定义;第2.2节是为何GE2E能更有效更新模型参数的理论依据;第2.3节介绍了一种称为“MultiReader”的技术，它使我们能够训练支持多种关键字和语言的单一模型;最后，我们在第3节中给出了我们的实验结果。

## 2. GENERALIZED END-TO-END MODEL

广义端到端(GE2E)训练是基于一次处理大量的话语，其形式是一个 batch 包含$N$个 speaker ，平均每个 speak 有$M$个话语，如图1所示。

![图1](/assets/images/nlp/ge2e/fig1.png)

图1：系统概述。不同的颜色表示不同 speaker 的 utterances/embedding。

### 2.1. 训练方法

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

## 2.2 Comparison between TE2E and GE2E

考虑 GE2E 损失更新中单个 batch:我们有$N$个speakers，每个 speaker 有$M$个话语。每一步更新都会将所有$N\times M$个嵌入向量推向它们自己的质心，并将它们拉离其他质心。

这反映了 TE2E 损失函数[13]中每个$x_{ji}$的所有可能元组的情况。假设我们在比较speakers 时，从speaker $j$中随机选择$P$个话语:

1. 正 tuples: $\{x_{ji},(x_{j,i_1},\cdots,x_{j,i_P})\}$,$1\le i_p \le M$,$p=1,\cdots,P$,有$\begin{pmatrix}M \\\\ P\end{pmatrix}$ 这样的正tuples.
2. 负 tuples: $\{x_{ji},(x_{k,i_1},\cdots,x_{k,i_P})\}$,$k\ne j$,$1\le i_p \le M$,$p=1,\cdots,P$, 对每个$x_{ji}$，我们不得不和所有其他$N-1$个质心比较，这$N-1$个比较的每一组都包含 $\begin{pmatrix}M \\\\ P\end{pmatrix}$ 个tuples.

每个正 tuple 都与一个负 tuple 进行平衡，因此总数是正 tuple 和负 tuple 的最大数量乘以2。因此，TE2E 损失的 tuples 总数为:

$$
\tag{11}
2\times \max( \begin{pmatrix}
    M \\\\P
\end{pmatrix},(N-1)\begin{pmatrix}
    M \\\\ P
\end{pmatrix}) \ge 2(N-1)
$$

等式11的下界出现在$P = m$时，因此，GE2E损失中$x_{ji}$的每次更新至少与 TE2E 损失中的$2(N - 1)$步相同。上述分析说明了 GE2E 更新模型比 TE2E 更新模型更有效的原因，这与我们的经验观察一致:GE2E在更短的时间内收敛到更好的模型(详见第3节)。

### 2.3 Training with MultiReader

考虑以下情况:我们关心具有小数据集$D_1$的域中的模型应用。同时，我们有一个更大的类似的数据集$D_2$，但不是相同的领域。我们想要在$D_2$的帮助下，训练一个在$D_1$数据集上表现良好的模型:

$$
\tag{12}
L(D_1,D_2;w) = \mathbb{E}_{x\in D_1} [L(x;w)] + \alpha \mathbb{E}_{x\in D_2}[L(x;w)]
$$

这类似于正则化技术:在正常的正则化中,我们使用$\alpha ||w||_2^2$正则化模型。但是这里，我们使用$\mathbb{E}_{x\in D_2}[L(x;w)]$来正则化。当数据集$D_1$没有足够的数据时，在$D_1$上训练网络会导致过拟合。要求网络在$D_2$上也能正常运行，这有助于网络的正则化。

这可以推广到结合$K$个不同的，可能极度不平衡的数据源:$D_1,\cdots,D_K$。我们给每个数据源指定一个权重$\alpha_k$,代表数据源的重要性。在训练期间,在每一步中，我们从每个数据源取一个 batch/tuple 的话语,并计算损失和为:$L(D_1,\cdots,D_K)= \sum_{k=1}^K \alpha_k \mathbb{E}_{x_k \in D_k} [L(x_k ; w)]$,其中每个$L(x_k ; w)$为式10中定义的损失。

## 3. 实验

在我们的实验中，特征提取过程与[6]相同。首先将音频信号转换为宽为25ms，步长为10ms的帧。然后我们提取40维的 log-mel-filterbank energies 作为每个帧的特征。对于TD-SV 应用，keyword detection 和 SV 都使用相同的特征。keyword 检测系统只会将包含keyword 的帧传递给 SV 系统。这些帧构成一个固定长度(通常为800ms)的段。对于TI-SV 应用，我们通常在语音活动检测(VAD)后提取随机定长片段，并使用滑动窗口方法进行推理(在第3.2节中讨论)。

我们的生产系统采用了带有 projection [16]的三层LSTM。嵌入向量(d-vector)大小与LSTM投影大小相同。对于 TD-SV，我们使用了128个隐藏节点，projection 大小为64。对于TI-SV，我们使用了 768 个隐藏节点，projection 大小为 256。当训练 GE2E 模型时，每批包含$N = 64$个 speakers，每个speaker 有 $M = 10$个话语。我们使用0.01的初始学习率的SGD对网络进行训练，每30M step 降低一半。梯度的L2范数裁剪为3[17]，将LSTM中 projection 节点的梯度 scale 设置为0.5。对于损失函数中的 scaling 因子$(w,b)$，我们也发现一个好的初值是$(w,b) =(10，−5)$，the smaller gradient scale of 0.01 on them 有利于平滑收敛。

### 3.1 Text-Dependent Speaker Verification

虽然现有的语音助手通常只支持一个 keyword，但研究表明，用户更希望同时支持多个 keywords。对于谷歌Home的多用户，同时支持两个 keywords:“OK Google”和“Hey Google”。

启用多个keywords 的 SV 是 TD-SV 和 TI-SV 之间的问题，因为 transcript 既不受单个短语的约束，也不完全不受约束。我们使用 MultiReader 技术解决了这个问题(第2.3节)。与更简单的方法(如直接混合多个数据源)相比，MultiReader有很大的优势:它可以处理不同数据源大小不平衡的情况。在我们的例子中，我们有两个用于训练的数据源:1)一个“OK Google”训练集，来自匿名化的用户查询，其中有$\sim 150$ M 的话语和 $\sim 630$ K的 speaker;2)一个混合的“OK/Hey google”的手动收集的训练集，它用$\sim 1.2$ M的话语和 $\sim 18$ k的speakers。第一个数据集比第二个数据集大125倍话语，和speaker 大35倍。

![表1](/assets/images/nlp/ge2e/tab1.png)

为了评估，我们报告了四种情况的 Equal Error Rate (EER):用任一 keyword 注册，并验证任一关键字。所有的评价数据集都是从665个 speaker 中手动收集的，平均每个 speaker 有4.5个登记话语和10个评价话语。结果如表1所示。正如我们所看到的，MultiReader 在所有四种情况下带来了大约30%的相对改善。

![表2](/assets/images/nlp/ge2e/tab2.png)

我们还从匿名日志和手工收集的 $\sim 83$ K不同的 speakers 和环境条件中收集了一个更大的数据集，对其进行了更全面的评估。我们平均使用7.3条登记话语和5条评价话语。表2总结了使用和不使用 MultiReader 设置训练的不同损失函数的平均 EER。基线模型是一个单层的LSTM，有512个节点，嵌入向量大小为128[13]。第二行和第三行的模型结构是3层LSTM。比较第二行和第三行，我们可以看到 GE2E 比 TE2E 大约好10%。与表1类似，在这里我们还可以看到模型在使用 MultiReader 时的性能明显更好。虽然表中没有显示，但值得注意的是 GE2E 模型比 TE2E 少了大约60%的训练时间。

### 3.2. Text-Independent Speaker Verification

![图3](/assets/images/nlp/ge2e/fig3.png)

图3：训练TI-SV模型的批建造过程。

在 TI-SV 训练中，我们将训练话语分成更小的片段，我们称之为 partial uttenrances (部分话语)。虽然我们不要求所有的部分话语都具有相同的长度，但同一批次的所有部分话语必须具有相同的长度。因此，对于每批数据，我们在$[lb, ub] =[140, 180]$帧内随机选择一个时间长度$t$，并强制该批数据中所有的部分话语都是长度$t$(如图3所示)。

![图4](/assets/images/nlp/ge2e/fig4.png)

图4：用于TI-SV的滑动窗口。

在推理期间，我们对每个话语应用一个固定大小的滑动窗口$(lb + ub)/2 = 160$帧，重叠50%。我们计算每个窗口的 d-vector。按窗口的d-vectors 进行 L2 normalizing，然后按元素取平均值，生成最终的按话语的 d-vectors (如图4所示)。

我们的TI-SV模型从匿名日志中提取了18K个 speaker 发出的约36M个语音。对于评估，我们使用了额外的1000个speaker，平均每个speaker有6.3条注册话语和7.2条评价话语。表三给出了不同训练损失函数的性能比较。第一列是softmax，用于预测训练数据中所有speaker的 speaker 标签。第二列是用TE2E损失训练的模型。第三列是用GE2E损失训练的模型。如表所示，GE2E的性能比softmax和TE2E都好。EER提高10%以上。此外，我们还观察到GE2E训练比其他损失函数快3倍左右。

## 4. 结论

在本文中，我们提出了广义端到端(GE2E)损失函数来更有效地训练SV模型。理论和实验结果都验证了该损失函数的优越性。我们还引入了MultiReader技术来组合不同的数据源，使我们的模型能够支持多个关键字和多种语言。通过结合这两种技术，我们得到了更精确的SV模型。

## 5. 引用

1. Yury Pinsky, home now “Tomato, supports
tomahto. multiple google users," https://www.blog.google/products/assistant/tomato-tomahto-google-home-now-supports-multiple-users, 2017.
2. Mihai Matei, “Voice match will al- low google home to recognize your voice,” https://www.androidheadlines.com/2017/10/voice-match- will-allow-google-home-to-recognize-your-voice.html, 2017.
3. Tomi Kinnunen and Haizhou Li, “An overview of text- independent speaker recognition: From features to supervec- tors,” Speech communication, vol. 52, no. 1, pp. 12–40, 2010.
4. Fre ́de ́ric Bimbot, Jean-Franc ̧ois Bonastre, Corinne Fre- douille, Guillaume Gravier, Ivan Magrin-Chagnolleau, Syl- vain Meignier, Teva Merlin, Javier Ortega-Garc ́ıa, Dijana Petrovska-Delacre ́taz, and Douglas A Reynolds, “A tutorial on text-independent speaker verification,” EURASIP journal on applied signal processing, vol. 2004, pp. 430–451, 2004.
5. Guoguo Chen, Carolina Parada, and Georg Heigold, “Small- footprint keyword spotting using deep neural networks,” in Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014, pp. 4087– 4091.
6. Rohit Prabhavalkar, Raziel Alvarez, Carolina Parada, Preetum Nakkiran, and Tara N Sainath, “Automatic gain control and multi-style training for robust small-footprint keyword spotting with deep neural networks,” in Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on. IEEE, 2015, pp. 4704–4708.
7. Najim Dehak, Patrick J Kenny, Re ́da Dehak, Pierre Du- mouchel, and Pierre Ouellet, “Front-end factor analysis for speaker verification,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 4, pp. 788–798, 2011.
8. Ehsan Variani, Xin Lei, Erik McDermott, Ignacio Lopez Moreno, and Javier Gonzalez-Dominguez, “Deep neural net- works for small footprint text-dependent speaker verification,” in Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014, pp. 4052– 4056.
9. Yu-hsin Chen, Ignacio Lopez-Moreno, Tara N Sainath, Mirko ́ Visontai, Raziel Alvarez, and Carolina Parada, “Locally- connected and convolutional neural networks for small foot- print speaker recognition,” in Sixteenth Annual Conference of the International Speech Communication Association, 2015.
10. Chao Li, Xiaokong Ma, Bing Jiang, Xiangang Li, Xuewei Zhang, Xiao Liu, Ying Cao, Ajay Kannan, and Zhenyao Zhu, “Deep speaker: an end-to-end neural speaker embedding sys- tem,” CoRR, vol. abs/1705.02304, 2017.
11. Shi-Xiong Zhang, Zhuo Chen, Yong Zhao, Jinyu Li, and Yi- fan Gong, “End-to-end attention based text-dependent speaker verification,” CoRR, vol. abs/1701.00562, 2017.
12. Seyed Omid Sadjadi, Sriram Ganapathy, and Jason W. Pele- canos, “The IBM 2016 speaker recognition system,” CoRR, vol. abs/1602.07291, 2016.
13. Georg Heigold, Ignacio Moreno, Samy Bengio, and Noam Shazeer, “End-to-end text-dependent speaker verification,” in Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on. IEEE, 2016, pp. 5115– 5119.
14. Sepp Hochreiter and Ju ̈rgen Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.
15. Florian Schroff, Dmitry Kalenichenko, and James Philbin, “Facenet: A unified embedding for face recognition and clus- tering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 815–823.
16. Has ̧im Sak, Andrew Senior, and Franc ̧oise Beaufays, “Long short-term memory recurrent neural network architectures for large scale acoustic modeling,” in Fifteenth Annual Conference of the International Speech Communication Association, 2014.
17. Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio, “Un-derstanding the exploding gradient problem,” CoRR, vol. abs/1211.5063, 2012.
---
**参考**：
1. 论文：Li Wan, Quan Wang, Alan Papir, Ignacio Lopez Moreno [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)


