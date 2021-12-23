# 【译】WaveNet: 原始音频的生成模型（v2,2016.9.19）

* 作者：Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
* 论文：《WaveNet: A Generative Model for Raw Audio》
* 地址：https://arxiv.org/abs/1609.03499

---

## 个人总结

---

**摘要**: 本文介绍了一种用于生成原始音频波形的深度神经网络WaveNet。该模型是完全概率和自回归的，每个音频样本的先验分布取决于所有先前的样本;尽管如此，我们证明它可以有效地训练成千上万的样本每秒音频的数据。当它被应用于文本到语音的转换时，它产生了最先进的表现，人类听众认为它比最好的参数化和连接系统(英语和汉语)的声音更自然。单个WaveNet可以以相同的保真度捕获许多不同 speakers 的特性，并通过对 speaker 身份的调节在它们之间进行切换。当我们接受音乐模型的训练时，我们发现它会产生新奇的，通常是高度写实的音乐片段。结果表明，该方法可以作为一种判别模型，在音素识别方面取得了良好的效果。

## 1. 介绍

这项工作探索原始音频生成技术,灵感来自最新进展自回归生成模型,模型 complex 分布，如图像(van den Oord et al ., 2016; b)和文本(Jozefowicz et al ., 2016)。使用神经结构作为条件分布的产物，对像素或单词的联合概率进行建模，可以产生最先进的生成技术。

值得注意的是，这些架构能够对数千个随机变量(如PixelRNN中的64×64像素)的分布进行建模(van den Oord et al.， 2016a)。本文要解决的问题是，类似的方法是否能够成功地生成宽带原始音频波形，这些波形是具有很高的时间分辨率的信号，至少每秒16,000个样本(见图1)。

![图1](/assets/images/深度学习/WaveNet/fig1.png)

图1：产生的一秒的话语。

本文介绍了基于PixelCNN的音频生成模型WaveNet (van den Oord et al.， 2016a;b)架构。这项工作的主要贡献如下:

* 我们证明WaveNets可以生成具有主观自然度的原始语音信号，这在之前的文本-语音转换(TTS)领域中从未报道过，由人类评分员进行评估。
* 为了处理原始音频生成所需的长期时间依赖关系，我们开发了基于 dialated 因果卷积(dilated causal convolutions，有叫空洞卷积的，有叫膨胀卷积的，以空洞的叫法为多)的新架构，该架构显示出非常大的接受域。
* 我们证明，当以说话者身份为条件时，一个单一的模型可以用来生成不同的声音。
* 同样的架构在小的语音识别数据集上测试时显示出强大的结果，并且在用于生成其他音频模式(如音乐)时很有前途。

我们相信WaveNets提供了一个通用和灵活的框架来处理许多依赖于音频生成的应用程序(如TTS、音乐、语音增强、语音转换、源分离)。

## 2. WaveNet

本文介绍了一种直接对原始音频波形进行生成的新模型。波形$\mathrm{x}=\{x_1,\cdots,x_T\}$的联合概率按下式分解为条件概率的乘积:

$$
\tag{1}
p(\mathrm{x}) = \prod_{t=1}^T p(x_t|x_1,\cdots,x_{t-1})
$$

因此，每个音频样本$x_t$都以之前的所有时间步长为条件。

与PixelCNNs类似(van den Oord et al.， 2016a;b)，条件概率分布是由一堆卷积层建模的。网络中不存在池化层，模型的输出具有与输入相同的时间维数。该模型通过一个softmax层输出下一个值$x_t$的一个分类分布，并对其进行优化以使数据的对数似然最大化。因为对数似然是可处理的，我们在验证集上调整超参数，可以很容易地测量模型是否过拟合或欠拟合。

### 2.1 Dilated Causal Convolutions

![图2](/assets/images/深度学习/WaveNet/fig2.png)

WaveNet的主要成分是因果卷积。通过使用因果卷积，我们确保模型不会违反我们对数据建模的顺序:模型在时间步$t$发出的预测$p(x_{t+1}|x_1,\cdots,x_t)$不能依赖于未来的时间步$x_{t+1},x_{t+2},\cdots,x_T$,如图2所示。对于图像，与因果卷积等价的是一个 masked 卷积(van den Oord et al.，2016a)，可以通过构造一个 mask 张量，并在对其进行运算之前对该mask与卷积核进行元素相乘来实现。对于像音频这样的一维数据，通过将普通卷积的输出移动几个时间步可以更容易地实现这一点。

在训练时，所有时间步的条件预测可以并行进行，因为ground truth $\mathrm{x}$的所有时间步都是已知的。当使用模型生成时，预测是有意义的:每个样本被预测后，它被反馈回网络以预测下一个样本。

因为带有因果卷积的模型不具有递归连接，所以它们通常比 RNNs 训练得更快，尤其是在应用于非常长的序列时。因果卷积的问题之一是，它们需要许多层，或大型 filters 来增加接受域。例如，在图2中，接受域只有5个(= #layers + filter length - 1)。在本文中，我们使用 dialated 卷积将接受域增加几个数量级，而不会大幅增加计算成本。

![图3](/assets/images/深度学习/WaveNet/fig3.png)

dialated 卷积(也称为 trous, 或 convolution with holes)是一种通过跳过特定步骤的输入值来对大于其长度的区域进行滤波的卷积。它等价于用一个更大的滤波器进行卷积，这个滤波器是由原来的滤波器通过用0来展开而来的，但是它的效率要高得多。dilated 卷积有效地允许网络在一个比正常卷积更大的范围内运行。这类似于 pooling 或 strided convolutions，但是这里输出的大小与输入的大小相同。作为一个特例，dilation 为1的 dilated 卷积即为标准卷积。图3描述了1、2、4和8的 dilations 的因果卷积。dilated 卷积之前已经在不同的上下文中使用过，例如信号处理(Holschneider et al.， 1989;Dutilleux, 1989)和图像分割(Chen et al.， 2015;Yu & Koltun, 2016)。

堆叠 dilated 卷积使得网络可以有非常大的感受野，而只需要几个层，同时保持整个网络的输入分辨率和计算效率。在本文中，每一层的膨胀倍数达到一定的限度，然后重复如下步骤:

$$
1,2,4,\cdots,512,1,2,4,\cdots,512,1,2,4,\cdots,512
$$

这种配置背后的直觉是双重的。首先，随着深度的增加，膨胀因子呈指数增长，导致感受野呈指数增长(Yu & Koltun, 2016)。例如每个$1,2,4,\cdots,512$ block 的感受野大小为1024，可以看作是$1\times 1024$卷积的更高效、更具区别性(非线性)的对应物。其次，堆叠这些块进一步增加模型容量和感受野大小。

### 2.2 softmax 分布

对于单个音频样本,一种建模条件分布$p(x_t|x_1,\cdots,x_{t-1})$的方法是可以使用混合模型，如 mixture density network (混合密度网络 Bishop, 1994)或 mixture of conditional Gaussian scale mixtures (条件高斯尺度混合,MCGSM. Theis & Bethge, 2015)。然而，van den Oord等人(2016a)表明，即使数据是隐式连续的(如图像像素强度或音频样本值)，softmax分布往往工作得更好。其中一个原因是，分类分布更灵活，更容易对任意分布建模，因为它对它们的形状没有任何假设。

由于原始音频通常存储为16位整数值序列(每个时间步一个)，所以softmax层需要在每个时间步输出65,536个概率来建模所有可能的值。使这更容易处理,首先应用$\mu$-law压缩数据转换(ITU-T, 1988),然后数字转换到256个可能的值:

$$
f(x_t) = sign(x_t)\frac{ln(1+\mu|x_t|)}{ln(1+\mu)}
$$

其中$-1 < x_t < 1$ 和 $\mu=255$ 这种非线性量化比简单的线性量化方案产生了更好的重建效果。特别是在语音方面，我们发现量化后的重构信号听起来与原始信号非常相似。

### 2.3 门控激活单元

我们使用与门控PixelCNN相同的门控激活单元(van den Oord et al.， 2016b):

$$
\tag{2}
\mathrm{z} = \tanh(W_{f,k}*\mathrm{x})\odot \sigma(W_{g,k}*\mathrm{x})
$$

其中$*$表示卷积操作，$\odot$表示按元素乘操作，$\sigma(\cdot)$是sigmoid函数，$k$是层索引，$f$和$g$表示 filter 和 gate.， $W$是一个可学习的卷积 filter。在我们最初的实验中，我们观察到这种非线性在建模音频信号方面明显优于整流线性激活函数(Nair & Hinton, 2010)。

### 2.4 残差连接和跳跃连接

![图4](/assets/images/深度学习/WaveNet/fig4.png)

残差(He et al.， 2015)和参数化跳跃连接都用于整个网络，以加快收敛速度，并使更深层次模型的训练成为可能。在图4中，我们展示了我们的模型的一个残差块，它在网络中被多次堆叠。

### 2.5 条件 WAVENETS

给定一个额外的输入$h$, WaveNets可以对给定输入的音频的条件分布$p(\mathrm{x}|h)$进行建模。等式(1)现在变成：

$$
\tag{3}
p(\mathrm{x}|\mathrm{h}) = \prod_{t=1}^T p(x_t|x_1,\cdots,x_{t-1},\mathrm{h})
$$

通过在其他输入变量上对模型进行调节，我们可以引导WaveNet生成具有所需特性的音频。例如，在多 speaker 设置中，我们可以通过将 speaker 标识作为额外输入输入到模型中来选择 speaker。类似地，对于TTS，我们需要提供关于文本的信息作为额外的输入。

我们以两种不同的方式对模型的其他输入进行调节:全局调节和局部调节。全局条件作用的特征是单个潜在表征$\mathrm{h}$，它影响所有时间步的输出分布，例如，嵌入在TTS模型中的 speaker。此时Eq.(2)的激活函数为:

$$
\mathrm{z} = \tanh(W_{f,k}*\mathrm{x}+V_{f,k}^T\mathrm{h})\odot \sigma(W_{g,k}*\mathrm{x}+V_{g,k}^T\mathrm{h})
$$

其中$V_{*,k}$是一个可学习的线性投影，向量$V_{*,k}^T\mathrm{h}$是在时间维度进行了广播。

对于局部条件作用，我们有第二个时间序列$h_t$，可能比音频信号的采样频率更低，例如TTS模型中的语言特征。我们首先使用一个 transposed 的卷积网络(learning upsampling)对这个时间序列进行变换，将它映射到一个与音频信号分辨率相同的新的时间序列$y=f(\mathrm{h})$，然后在激活单元中使用如下:

$$
\mathrm{z} = \tanh(W_{f,k}*\mathrm{x}+V_{f,k}*\mathrm{y})\odot \sigma(W_{g,k}*\mathrm{x}+V_{g,k}*\mathrm{y})
$$

其中$V_{f,k}*\mathrm{y}$现在是一个$1\times 1$卷积，作为 transposed 卷积网络的替代，也可以使用$V_{f,k}*\mathrm{h}$并在不同时间重复这些值。在我们的实验中，我们发现这种方法的效果稍差。

### 2.6 上下文堆栈

我们已经提到了几种不同的方法来增加 WaveNet 的感受野大小:增加 dilation 阶段的数量，使用更多的层，更大的过滤器，更大的膨胀因子，或者是它们的组合。一种补充的方法是使用一个单独的、更小的上下文堆栈来处理音频信号的长部分，而在局部条件下使用一个更大的WaveNet来处理音频信号的一小部分(在末尾裁剪)。可以使用具有不同长度和数量隐藏单元的多个上下文堆栈。具有较大感受野的堆栈每层的单位数较少。上下文堆栈还可以使用池化层以较低的频率运行。这将计算需求保持在一个合理的水平上，并且与以下直觉相一致:在更长的时间尺度上，建模时间相关性所需的能力更少。

## 3. 实验

为了测量WaveNet的音频建模性能，我们用三个不同的任务来评估它:多speaker 语音生成(不以文本为条件)、TTS和音乐音频建模。我们在附带的网页上提供了来自WaveNet的用于这些实验的样本:https://www.deepmind.com/blog/wavenet-genermodel-raw-audio/.

### 3.1 多speaker 语音生成

在第一个实验中，我们观察了自由形式的语音生成(不以文本为条件)。我们使用来自CSTR语音克隆工具包(VCTK)的英语多 speaker 语料库(Yamagishi, 2012)和仅针对 speaker 的条件 WaveNet。通过将 speaker ID以 one-hot 向量的形式提供给模型，对其进行了调节。数据集包括来自109位不同演讲者的44个小时的数据。

由于该模型不以文本为条件，所以它以一种流畅的方式生成了不存在的但类似人类语言的单词，语调听起来也很真实。这类似于语言或图像的生成模型，其中的样本乍一看很真实，但仔细观察就会发现明显不自然。缺乏长距离的连贯性部分是由于模型的感受野有限(大约300毫秒)，这意味着它只能记住最后2-3个音素。

一个单独的WaveNet就可以通过对speaker进行one-hot编码来模拟来自任何 speaker 的语音。这证实了它的强大功能足以在一个模型中从数据集捕获所有109个 speakers 的特征。我们观察到，与只训练一个 speaker 相比，添加 speaker 可以获得更好的验证集性能。这表明WaveNet的内部表征在多个 speakers 之间共享。

最后，我们观察到，除了声音本身，模型还考虑了音频中的其他特性。例如，它还模仿了音响效果和录音质量，以及 speaker 的呼吸和嘴部运动。

### 3.2 TEXT-TO-SPEECH

在第二个实验中，我们观察了TTS。我们使用与谷歌北美英语和汉语普通话TTS系统相同的单speaker 语音数据库。北美英语数据集包含24.6小时语音数据，汉语普通话数据集包含34.8小时;这两种语言都是由职业女性 speakers 讲的。

在TTS任务中，WaveNets根据输入文本的语言特征进行局部条件设置。除了语言特征外，我们还根据对数基频($\log F_0$)值对 WaveNets 进行了训练。对每种语言的外部模型也进行了训练，以从语言特征预测 $\log F_0$值和 phone 时长。WaveNet 的感受野大小为240毫秒。以基于实例和基于模型的语音合成基线为基础，构建了隐马尔可夫模型(HMM)驱动的单元选择 concatenative (Gonzalvo et al.， 2016)和基于长短期记忆递归神经网络(LSTM-RNN)的统计参数(Zen et al.， 2016)语音合成器。由于使用相同的数据集和语言特征来训练基线和 WaveNets，因此可以对这些语音合成器进行比较。

为了评估WaveNets在TTS任务中的表现，我们进行了主观配对比较测试和平均意见评分(MOS)测试。在成对比较测试中，在听了每对样本后，受试者被要求选择他们更喜欢的样本，尽管如果他们没有任何偏好，他们可以选择“中性”。在MOS测试中，在听完每个刺激后，受试者被要求对刺激的自然程度进行 Likert 五分制评分(1:Bas，2:Poor，3:Fair，4:Good，5:Excellent)。详见附件B。

![图5](/assets/images/深度学习/WaveNet/fig5.png)

图5：语音样本的主观偏好评分(%)介于两个基线(上)、两个WaveNets(中)和最佳基线和WaveNet之间(下)。注意，LSTM和Concat对应于 LSTM-RNN-based 的统计参数和 HMM-driven (驱动)的单元选择连接基线合成器，而WaveNet (L)和WaveNet (L+F)对应于仅针对语言特征的WaveNet，且同时基于语言特征和$\log F_0$值。

图5为主观配对比较检验结果的选择(完整表见附录B)。从实验结果可以看出，WaveNet在两种语言中都优于基线统计参数和连接语音合成器。我们发现，基于语言特征的WaveNet可以合成具有自然节段性的语音样本，但有时由于在句子中重读错误的单词而产生不自然的韵律。这可能是由于F0轮廓线的长期依赖性造成的:WaveNet的感受野大小为240毫秒，不足以捕捉这种长期依赖性。基于语言特征和F0值的WaveNet没有这个问题:外部的F0预测模型运行在较低的频率(200hz)，因此它可以了解存在于F0等值线中的长期依赖关系。

表1为MOS测试结果。从表中可以看出，WaveNets的自然度达到了4.0以上的5级MOSs，明显好于基线系统。在这些训练数据集和测试语句中，它们是最高的MOS值。在美国英语中，最佳合成语音与天然语音的差距从0.69降至0.34(51%)，在普通话中，差距从0.42降至0.13(69%)。

### 3.3 音乐

在第三组实验中，我们训练WaveNets对两个音乐数据集进行建模:
* MagnaTagATune 数据集(Law&VonAhn,2009)，其中包含大约200小时的音乐音频。每一段29秒的音乐片段都用188个标签进行了注解，这些标签描述了音乐的体裁、乐器、节奏、音量和情绪。
* YouTube钢琴数据集，包括从YouTube视频中获得的约60小时的钢琴独奏音乐。因为它被限制在一个单一的仪器上，所以它的建模要容易得多。

虽然很难对这些模型进行定量评估，但是通过倾听它们所产生的样本，可以进行主观评估。我们发现，扩大接受范围对获得某些听起来像音乐的样本至关重要。即使接受时间只有几秒钟，这些模型也不能保证长时间的一致性，这就导致了音乐流派、乐器演奏、音量和音质的每秒变化。尽管如此，即使是由无条件的模型制作出来的样本，也常常是和谐的、美观的。

特别令人感兴趣的是条件音乐模型，它可以生成音乐给定一组标签，如流派或乐器。与条件语音模型类似，我们插入依赖于与每个训练片段相关的标记的二进制向量表示的偏差。这使得在采样时控制模型输出的各个方面成为可能，方法是输入一个二进制向量来编码样本的期望属性。我们已经在MagnaTagATune数据集上训练了这样的模型;尽管与数据集绑定的标记数据相对比较吵，有很多遗漏，但是通过合并相似的标记并删除那些关联剪辑太少的标记，我们发现这种方法工作得相当好。

### 3.4 语音识别

虽然WaveNet被设计成一个生成模型，但它可以直接应用于语音识别等非犯罪音频任务。

传统上，语音识别的研究主要集中在使用 mel-filterbank energies 或mel-频率倒谱系数(MFCCs)，但最近转向原始音频(Palaz et al.， 2013;Tuske et al., 2014;Hoshen等人，2015;Sainath 等人，2015)。LSTM-RNNs (Hochreiter & Schmidhuber, 1997)等递归神经网络已经成为这些新的语音分类管道中的一个关键组成部分，因为它们允许建立具有长范围上下文的模型。使用WaveNets，我们已经展示了扩展的卷积层允许感受野以比使用LSTM单元更便宜的方式增长。

最后一个实验是在TIMIT (Garofolo et al.， 1993)数据集上使用WaveNets进行语音识别。在这个任务中，我们在 dilated 卷积之后添加了一个均值池化层，dilated 卷积将激活扩展为更粗的帧，时间跨度为10毫秒(160倍下采样)。池化层之后是一些非因果卷积。我们用两个损失项训练WaveNet,一个损失预测下一个样本，另一个分类 frame ,比用一个损失,模型泛化更好,在测试集取得了18.8PER,这是我们从模型获得的最好的知识得分，在直接对原始音频TIMIT训练上。

## 4. 结论

本文提出了WaveNet，这是一种直接在波形级运行的音频数据的深度生成模型。wavenet是自回归的，它将因果filter 与 dilated 卷积相结合，使它们的感受野随深度呈指数增长，这对于建立音频信号的长期时间依赖关系非常重要。我们已经展示了WaveNets如何在全局(如 speaker身份)或局部(如语言特征)的其他输入上进行配置。当应用于TTS时，WaveNets生成的样本在主观自然性方面优于当前最好的TTS系统。最后，WaveNets在音乐音频建模和语音识别方面显示出了很好的应用前景。

## 感谢

作者们要感谢Lasse Espeholt, Jeffrey De Fauw和Grzegorz szcz的投入，Adam Cain, Max和Adrian Bolton对 artwork 的帮助，Helen King, Steven Gaffney和Steve Crossan帮助管理这个项目，Faith Mackinder帮助准备博客文章，James Besley提供法律支持，Demis Hassabis负责管理这个项目和他的 inputs。

## 引用

* Agiomyrgiannakis, Yannis. Vocaine the vocoder and applications is speech synthesis. In ICASSP, pp. 4230–4234, 2015.
* Bishop, Christopher M. Mixture density networks. Technical Report NCRG/94/004, Neural Com- puting Research Group, Aston University, 1994.
* Chen, Liang-Chieh, Papandreou, George, Kokkinos, Iasonas, Murphy, Kevin, and Yuille, Alan L. Semantic image segmentation with deep convolutional nets and fully connected CRFs. In ICLR, 2015. URL http://arxiv.org/abs/1412.7062.
* Chiba, Tsutomu and Kajiyama, Masato. The Vowel: Its Nature and Structure. Tokyo-Kaiseikan, 1942.
* Dudley, Homer. Remaking speech. The Journal of the Acoustical Society of America, 11(2):169– 177, 1939.
* Dutilleux, Pierre. An implementation of the “algorithme a` trous” to compute the wavelet transform. In Combes, Jean-Michel, Grossmann, Alexander, and Tchamitchian, Philippe (eds.), Wavelets: Time-Frequency Methods and Phase Space, pp. 298–304. Springer Berlin Heidelberg, 1989.
* Fan, Yuchen, Qian, Yao, and Xie, Feng-Long, Soong Frank K. TTS synthesis with bidirectional LSTM based recurrent neural networks. In Interspeech, pp. 1964–1968, 2014.
* Fant, Gunnar. Acoustic Theory of Speech Production. Mouton De Gruyter, 1970.
* Garofolo, John S., Lamel, Lori F., Fisher, William M., Fiscus, Jonathon G., and Pallett, David S. DARPA TIMIT acoustic-phonetic continuous speech corpus CD-ROM. NIST speech disc 1-1.1. NASA STI/Recon technical report, 93, 1993.
* Gonzalvo, Xavi, Tazari, Siamak, Chan, Chun-an, Becker, Markus, Gutkin, Alexander, and Silen, Hanna. Recent advances in Google real-time HMM-driven unit selection synthesizer. In Inter- speech, 2016. URL http://research.google.com/pubs/pub45564.html.
* He, Kaiming, Zhang, Xiangyu, Ren, Shaoqing, and Sun, Jian. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.
* Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Comput., 9(8):1735–1780, 1997.
* Holschneider, Matthias, Kronland-Martinet, Richard, Morlet, Jean, and Tchamitchian, Philippe. A real-time algorithm for signal analysis with the help of the wavelet transform. In Combes, Jean- Michel, Grossmann, Alexander, and Tchamitchian, Philippe (eds.), Wavelets: Time-Frequency Methods and Phase Space, pp. 286–297. Springer Berlin Heidelberg, 1989.
* Hoshen, Yedid, Weiss, Ron J., and Wilson, Kevin W. Speech acoustic modeling from raw multi- channel waveforms. In ICASSP, pp. 4624–4628. IEEE, 2015.
* Hunt, Andrew J. and Black, Alan W. Unit selection in a concatenative speech synthesis system using a large speech database. In ICASSP, pp. 373–376, 1996.
* Imai, Satoshi and Furuichi, Chieko. Unbiased estimation of log spectrum. In EURASIP, pp. 203– 206, 1988.
* Itakura, Fumitada. Line spectrum representation of linear predictor coefficients of speech signals. The Journal of the Acoust. Society of America, 57(S1):S35–S35, 1975.
* Itakura, Fumitada and Saito, Shuzo. A statistical method for estimation of speech spectral density and formant frequencies. Trans. IEICE, J53A:35–42, 1970.
* ITU-T. Recommendation G. 711. Pulse Code Modulation (PCM) of voice frequencies, 1988.
* Jo ́zefowicz, Rafal, Vinyals, Oriol, Schuster, Mike, Shazeer, Noam, and Wu, Yonghui. Exploring the limits of language modeling. CoRR, abs/1602.02410, 2016. URL http://arxiv.org/abs/ 1602.02410.
* Juang, Biing-Hwang and Rabiner, Lawrence. Mixture autoregressive hidden Markov models for speech signals. IEEE Trans. Acoust. Speech Signal Process., pp. 1404–1413, 1985.
* Kameoka, Hirokazu, Ohishi, Yasunori, Mochihashi, Daichi, and Le Roux, Jonathan. Speech anal- ysis with multi-kernel linear prediction. In Spring Conference of ASJ, pp. 499–502, 2010. (in Japanese).
* Karaali, Orhan, Corrigan, Gerald, Gerson, Ira, and Massey, Noel. Text-to-speech conversion with neural networks: A recurrent TDNN approach. In Eurospeech, pp. 561–564, 1997.
* Kawahara, Hideki, Masuda-Katsuse, Ikuyo, and de Cheveigne ́, Alain. Restructuring speech rep- resentations using a pitch-adaptive time-frequency smoothing and an instantaneous-frequency- based f0 extraction: possible role of a repetitive structure in sounds. Speech Commn., 27:187– 207, 1999.
* Kawahara, Hideki, Estill, Jo, and Fujimura, Osamu. Aperiodicity extraction and control using mixed mode excitation and group delay manipulation for a high quality speech analysis, modification and synthesis system STRAIGHT. In MAVEBA, pp. 13–15, 2001.
* Law, Edith and Von Ahn, Luis. Input-agreement: a new mechanism for collecting data using human computation games. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 1197–1206. ACM, 2009.
* Maia, Ranniery, Zen, Heiga, and Gales, Mark J. F. Statistical parametric speech synthesis with joint estimation of acoustic and excitation model parameters. In ISCA SSW7, pp. 88–93, 2010.
* Morise, Masanori, Yokomori, Fumiya, and Ozawa, Kenji. WORLD: A vocoder-based high-quality speech synthesis system for real-time applications. IEICE Trans. Inf. Syst., E99-D(7):1877–1884, 2016.
* Moulines, Eric and Charpentier, Francis. Pitch synchronous waveform processing techniques for text-to-speech synthesis using diphones. Speech Commn., 9:453–467, 1990.
* Muthukumar, P. and Black, Alan W. A deep learning approach to data-driven parameterizations for statistical parametric speech synthesis. arXiv:1409.8558, 2014.
* Nair, Vinod and Hinton, Geoffrey E. Rectified linear units improve restricted Boltzmann machines. In ICML, pp. 807–814, 2010.
* Nakamura, Kazuhiro, Hashimoto, Kei, Nankaku, Yoshihiko, and Tokuda, Keiichi. Integration of spectral feature extraction and modeling for HMM-based speech synthesis. IEICE Trans. Inf. Syst., E97-D(6):1438–1448, 2014.
* Palaz, Dimitri, Collobert, Ronan, and Magimai-Doss, Mathew. Estimating phoneme class condi- tional probabilities from raw speech signal using convolutional neural networks. In Interspeech, pp. 1766–1770, 2013.
* Peltonen, Sari, Gabbouj, Moncef, and Astola, Jaakko. Nonlinear filter design: methodologies and challenges. In IEEE ISPA, pp. 102–107, 2001.
* Poritz, Alan B. Linear predictive hidden Markov models and the speech signal. In ICASSP, pp. 1291–1294, 1982.
* Rabiner, Lawrence and Juang, Biing-Hwang. Fundamentals of Speech Recognition. PrenticeHall, 1993.
* Sagisaka, Yoshinori, Kaiki, Nobuyoshi, Iwahashi, Naoto, and Mimura, Katsuhiko. ATR ν-talk speech synthesis system. In ICSLP, pp. 483–486, 1992.
* Sainath, Tara N., Weiss, Ron J., Senior, Andrew, Wilson, Kevin W., and Vinyals, Oriol. Learning the speech front-end with raw waveform CLDNNs. In Interspeech, pp. 1–5, 2015.
* Takaki, Shinji and Yamagishi, Junichi. A deep auto-encoder based low-dimensional feature ex- traction from FFT spectral envelopes for statistical parametric speech synthesis. In ICASSP, pp. 5535–5539, 2016.
* Takamichi, Shinnosuke, Toda, Tomoki, Black, Alan W., Neubig, Graham, Sakriani, Sakti, and Naka- mura, Satoshi. Postfilters to modify the modulation spectrum for statistical parametric speech synthesis. IEEE/ACM Trans. Audio Speech Lang. Process., 24(4):755–767, 2016.
* Theis, Lucas and Bethge, Matthias. Generative image modeling using spatial LSTMs. In NIPS, pp. 1927–1935, 2015.
* Toda, Tomoki and Tokuda, Keiichi. A speech parameter generation algorithm considering global variance for HMM-based speech synthesis. IEICE Trans. Inf. Syst., E90-D(5):816–824, 2007.
* Toda, Tomoki and Tokuda, Keiichi. Statistical approach to vocal tract transfer function estimation based on factor analyzed trajectory hmm. In ICASSP, pp. 3925–3928, 2008.
* Tokuda, Keiichi. Speech synthesis as a statistical machine learning problem. http://www.sp. nitech.ac.jp/ ̃tokuda/tokuda_asru2011_for_pdf.pdf, 2011. Invited talk given at ASRU.
* Tokuda, Keiichi and Zen, Heiga. Directly modeling speech waveforms by neural networks for statistical parametric speech synthesis. In ICASSP, pp. 4215–4219, 2015.
* Tokuda, Keiichi and Zen, Heiga. Directly modeling voiced and unvoiced components in speech waveforms by neural networks. In ICASSP, pp. 5640–5644, 2016.
* Tuerk, Christine and Robinson, Tony. Speech synthesis using artificial neural networks trained on cepstral coefficients. In Proc. Eurospeech, pp. 1713–1716, 1993.
* Tu ̈ske, Zolta ́n, Golik, Pavel, Schlu ̈ter, Ralf, and Ney, Hermann. Acoustic modeling with deep neural networks using raw time signal for LVCSR. In Interspeech, pp. 890–894, 2014.
* Uria, Benigno, Murray, Iain, Renals, Steve, Valentini-Botinhao, Cassia, and Bridle, John. Modelling acoustic feature dependencies with artificial neural networks: Trajectory-RNADE. In ICASSP, pp. 4465–4469, 2015.
* van den Oord, Aa ̈ron, Kalchbrenner, Nal, and Kavukcuoglu, Koray. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, 2016a.
* van den Oord, Aa ̈ron, Kalchbrenner, Nal, Vinyals, Oriol, Espeholt, Lasse, Graves, Alex, and Kavukcuoglu, Koray. Conditional image generation with PixelCNN decoders. CoRR, abs/1606.05328, 2016b. URL http://arxiv.org/abs/1606.05328.
* Wu, Yi-Jian and Tokuda, Keiichi. Minimum generation error training with direct log spectral distor- tion on LSPs for HMM-based speech synthesis. In Interspeech, pp. 577–580, 2008.
* Yamagishi, Junichi. English multi-speaker corpus for CSTR voice cloning toolkit, 2012. URL http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html.
* Yoshimura, Takayoshi. Simultaneous modeling of phonetic and prosodic parameters, and char- acteristic conversion for HMM-based text-to-speech systems. PhD thesis, Nagoya Institute of Technology, 2002.
* Yu, Fisher and Koltun, Vladlen. Multi-scale context aggregation by dilated convolutions. In ICLR, 2016. URL http://arxiv.org/abs/1511.07122.
* Zen, Heiga. An example of context-dependent label format for HMM-based speech synthesis in English, 2006. URL http://hts.sp.nitech.ac.jp/?Download.
* Zen, Heiga, Tokuda, Keiichi, and Kitamura, Tadashi. Reformulating the HMM as a trajectory model by imposing explicit relationships between static and dynamic features. Comput. Speech Lang., 21(1):153–173, 2007.
* Zen, Heiga, Tokuda, Keiichi, and Black, Alan W. Statistical parametric speech synthesis. Speech Commn., 51(11):1039–1064, 2009.
* Zen, Heiga, Senior, Andrew, and Schuster, Mike. Statistical parametric speech synthesis using deep neural networks. In Proc. ICASSP, pp. 7962–7966, 2013.
* Zen, Heiga, Agiomyrgiannakis, Yannis, Egberts, Niels, Henderson, Fergus, and Szczepaniak, Prze- mysław. Fast, compact, and high quality LSTM-RNN based statistical parametric speech synthe- sizers for mobile devices. In Interspeech, 2016. URL https://arxiv.org/abs/1606. 06061.

## 附录

略


---
**参考**：
1. 论文：Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
