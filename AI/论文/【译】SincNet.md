---
typora-root-url: ../../../
---

# 【译】用 SincNet 从原始波形识别 speaker

* 作者：Mirco Ravanelli, Yoshua Bengio
* 论文：《SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET》
* 地址：https://arxiv.org/abs/1808.00158v3

---
## 个人总结：

在语音领域，以往多是在手工处理的特征基础上再进行更上层的处理。但手工设计的特征往往会丢失一些和任务有关的关键的特征。所以最新的研究倾向于直接在 spectrogram bins(声谱图箱)，甚至 raw waveforms (原始波形) 上应用神经网络。最流行的是采用CNNs来处理。本文认为基于波形的CNNs最关键的是第一层卷积。

于是作者设计了新的 filter,对 filter shape 增加了约束，通过一组参数化的 sinc 函数（辛格函数）实现 band-pass filters （带通滤波器就是让某个频率范围的波通过，不是这个范围的波衰减掉）

设计的 filter 具有参数量少，可解释性强，收敛快的特点

---

## 摘要

作为一种可行的替代 i-vector 的 specker 识别方法，深度学习正日益受到欢迎。利用卷积神经网络(CNNs)直接用原始语音样本进行fed，最近得到了一些有希望的结果。与使用标准的手工制作的功能不同，后来的 CNNs 从波形中学习低级的语音表征，潜在地允许网络更好地捕获重要的窄带 speaker 特性，如 pitch(音高)和formants(共振峰)。合理设计神经网络是实现这一目标的关键。

本文提出了一种新的CNN架构，称为SincNet，它鼓励第一个卷积层发现更有意义的过滤器。SincNet基于参数化sinc函数，实现 band-pass filters。与标准CNNs不同的是，标准CNNs学习每个 filters 的所有元素，该方法仅仅直接从数据中学习(low and high cutoff frequencies )低截止频率和高截止频率。这提供了一种非常简洁和有效的方法来派生专门针对所需应用进行 tuned (调整，调音)的自定义 filter bank。

我们的实验，在 speaker 识别和 speaker 验证的任务，表明提出的架构在原始波形上比标准 cnn 收敛更快,效果更好。

**Index Terms** —— speaker recognition, convolutional neural networks, raw samples.

## 1. 介绍

speaker recognition 是一个非常活跃的研究领域，在生物特征识别、取证、安全、语音识别和 speaker diarization 等领域有着不可比拟的应用，这使得人们对这门学科[1]产生了浓厚的兴趣。目前最先进的解决方案是基于语音段[2]的 i-vector 表征，这有助于显著改善之前的高斯混合模型-通用背景模型(GMM- UBMs)[3]。深度学习已经在许多语音任务中取得了显著的成功[4-8]，包括最近在speaker recognition 方面的研究[9,10]。深度神经网络(DNNs)已在i-vector框架内用于计算 Baum-Welch 统计[11]，或用于框架级特征抽取[12]。DNNs也被提议用于直接区分 speaker classification，最近关于这一主题的文献[13-16]证明了这一点。然而，过去的大多数尝试，采用手工制作的特征，如FBANK和MFCC系数[13,17,18]。这些经过设计的特性最初是根据感知的证据设计的，并且不能保证这些表征对于所有与语音相关的任务都是最优的。例如，标准特征使语音频谱平滑，这可能会妨碍提取诸如 pitch 和 formants 之类的重要的 narrow-band speaker 特征。为了减轻这一缺陷，最近的一些工作提出直接用 spectrogram bins(声谱图箱）[19-21]或甚至用 raw waveforms (原始波形)[22-34]来 fed 网络。CNNs是处理原始语音样本的最流行的架构，因为权重共享、local filters 和 polling 有助于发现鲁棒和不变的表征。

我们认为当前基于波形的CNNs最关键的部分之一是第一卷积层。这一层不仅处理高维输入，而且更容易受到消失的梯度问题的影响，特别是在使用非常深的架构时。CNN 学习的 filters 通常采用嘈杂且 incongruous (不协调)的 multi-band shapes (多频带形状)，特别是在可用的训练样本很少的情况下。这些过滤器对神经网络的工作当然有一定的意义，但对人类的直觉没有吸引力，也似乎不会导致语音信号的有效表达。

为了帮助CNNs在输入层中发现更有意义的 filters ，本文提出在 filters shape 上增加一些约束。与标准的CNNs相比，SincNet的 filter bank 特性依赖于几个参数(filter vector 的每个参数都是直接学习的)，SincNet通过一组参数化的sinc函数来实现 band-pass filters，从而卷积波形。低、高 cutoff 频率是 filter 从数据中学到的唯一参数。这个解决方案仍然提供了相当大的灵活性，但是迫使网络把重点放在对最终 filter 的 shape 和 bandwidth 有广泛影响的高级可调参数上。

我们的实验是在极具挑战性的条件下进行的，这种条件的特点是训练数据极少(即训练数据很少)。和简短的测试句(最后2- 6秒)。在各种数据集上取得的结果表明，本文提出的SincNet算法比更标准的CNN算法收敛速度更快，具有更好的末端任务性能。在考虑的实验设置下，我们的架构也优于一个更传统的基于 i-vector 的speaker recognition 。

论文的其余部分组织如下。SincNet体系结构在第2节中进行了描述。第3节讨论了与先前工作的关系。实验设置和结果分别在第4节和第5节中概述。最后，第6节讨论了我们的结论。

## 2. SincNet 架构

![图1](/assets/images/nlp/sincnet/fig1.png)

图1：SincNet 架构

标准CNN的第一层在输入波形和一些 Finite Impulse Response (有限脉冲响应,FIR)滤波器[35]之间执行一组时域卷积。每个卷积定义如下:

$$
\tag{1}
y[n] = x[n]*h[n] = \sum_{l=0}^{L-1} x[l]\cdot h[n-l]
$$

其中$x[n]$为语音信号块,$h[n]$为长度$L$的 filter,$y[n]$为滤波后的输出。在标准CNNs中，每个 filter 的所有$L$元素(elements,taps)都是从数据中学习的。相反,该SincNet(图1中所示)用一个预定义的函数$g$执行卷积，$g$只依赖几个可学习的参数$\theta$，在下面方程中凸显:

$$
\tag{2}
y[n]=x[n]*g[n,\theta]
$$

在数字信号处理中，受标准 filter 的启发，采用矩形 bandpass filters 组成的 filter-bank 来定义$g$是一种合理的选择。在频域上，一般 bandpass filter 的幅值可以表示为两个 low-pass filters 的差值:

$$
\tag{3}
G[f,f_1,f_2] = rect(\frac{f}{2f_2})-rect(\frac{f}{2f_1})
$$

---
个人补充：矩形函数的定义为

$$
rect(t) = \Pi(t) = 
\begin{cases}
    0 \quad if |t|> \frac{1}{2} \\\\
    \frac{1}{2}  \quad if |t|=\frac{1}{2} \\\\ 
    1  \quad if |t|<\frac{1}{2}
\end{cases}
$$

图像：

![rect](/assets/images/nlp/sincnet/rect.png)

所以$rect(\frac{f}{2f_2})-rect(\frac{f}{2f_1})$,取$f_2=3,f_1=2$时，图像为：

![rect3-2](/assets/images/nlp/sincnet/rect3-2.png)

考虑到频率不会小于0，应该只考虑大于0的范围

---

其中$f_1$和$f_2$分别为习得的低、高 cutoff frequencies，$rect(\cdot)$为 magnitude frequency domain (幅频域)中的矩形函数。返回时域后(使用傅里叶逆变换[35])，参考函数$g$为:

$$
\tag{4}
g[n,f_1,f_2] = 2f_2 sinc(2\pi f_2 n)-2f_1 sinc(2\pi f_1 n)
$$

其中 sinc 函数定义为$sinc(x)=sin(x)/x$。

cut-off 频率可以在$[0,f_s/2]$范围内随机初始化，其中$f_s$表示输入信号的采样频率。作为一种替代方法，filters 可以初始化为mel-scale filter-bank的 cutoff 频率，它的优点是可以直接在频谱较低的部分分配更多的滤波器，关于 speaker 身份的许多重要线索都在这里。为了保证$f_1\ge 0$和$f_2\ge f_1$，前一个方程实际上是由以下参数提供的:

$$
\tag{5}
f_1^{abs} = |f_1|
$$

$$
\tag{6}
f_2^{abs} = f_1 + |f_2 - f_1|
$$

请注意，我们并没有强制$f_2$小于 Nyquist frequency，因为我们观察到这个约束在训练期间自然地得到了满足。此外，每个 filter 的 gain (增益)不是在这个层次上学习的。这个参数计由后续的层来管理，它可以很容易地赋予每个 filter 的输出或多或少的重要性。

一个理想的 bandpass filter（即，passband 的 filter 是完全平坦和 stopband 的衰减是无限的）需要无限的元素$L$. $g$的任意截断从而不可避免地导致了逼近理想filter的特点——passband 呈波状,stopband 有限衰减。缓解这个问题的一个流行的解决方案是窗口化[35]。窗口化是将截断的函数$g$与窗口函数$w$相乘，其目的是平滑$g$末端的(the abrupt discontinuities)突变不连续点:

$$
\tag{7}
g_w[n,f_1,f_2] = g[n,f_1,f_2]\cdot w[n]
$$

本文采用流行的 Hamming 窗口[36]，定义如下:

$$
\tag{8}
w[n] = 0.54 - 0.46 \cdot cos(\frac{2\pi n}{L})
$$

Hamming 窗口特别适合实现高频率选择性[36]。但是，这里没有报告的结果显示，在采用其他 functions (如Hann、Blackman 和 Kaiser windows)时，没有显著的性能差异。还请注意，filters $g$是对称的，因此不会引入任何 phase distortions (相位畸变)。由于对称性，可以通过考虑 filter 的一边并继承另一半的结果来有效地计算 filters。

SincNet中涉及的所有操作都是完全可微的，滤波器的 cutoff frequencies 可以使用随机梯度下降法(SGD)或其他基于梯度的优化方法与其他CNN参数联合优化。如图1所示，第一个基于sincs的卷积后，可以使用标准的CNN管道(池化、归一化、激活、dropout)。多个标准的卷积层、全连接层或递归层[37-40]可以叠加在一起，最后使用softmax分类器进行 speaker 分类。

## 2.1 模型属性

提出的SincNet具有一些显著的性质:

* **快速收敛**: SincNet强制网络只考虑对性能有重大影响的 filter 参数。所提出的方法实际上补充了一种自然的 inductive bias (归纳偏差)，它利用了有关filter shape 的知识(类似于这项任务中通常使用的特征提取方法)，同时保留了适应数据的灵活性。这种先验知识使得学习 filter 特性变得更加容易，帮助SincNet更快地收敛到一个更好的解决方案。

* **参数量少**: SincNet 大大减少了第一个卷积层的参数数量。例如，如果我们考虑一个长度为$L$的$F$ filters 组成的层，一个标准的 CNN 使用$F\cdot L$参数，而 SincNet 考虑的是$2F$。如果$F = 80$,和 $L = 100$，我们对 CNN 使用 8k 参数，而对 SincNet 仅使用 160。此外,如果我们用 2倍的filter 长度,一个标准的CNN 也2倍其参数计算(例如,从 8k 到 16k),尽管SincNet 不改变参数量(只有两个参数是用于每个过滤器,不管它的长度$L$)。这提供了一种可能性，可以获得具有许多 taps 的非常有选择性的过滤器，而不需要在优化问题中实际添加参数。此外，SincNet 体系结构的紧凑性使其适合于少数样本的情况。

* **可解释性**: 与其他方法相比，在第一个卷积层中获得的SincNet特征映射显然更具有可移植性和可读性。事实上，filter bank 只依赖于具有明确物理意义的参数。

## 3. 相关工作

最近有几项研究探索了使用CNNs来处理 audio 和 speech 的 low-level speech 表征。之前的大多数尝试都利用了 magnitude(量级，震级) spectrogram（声谱图） 特征[19 - 21,41 - 43]。虽然声谱图比标准的手工特征保留了更多的信息，但它们的设计仍然需要仔细调整一些关键的超参数，比如帧窗口的持续时间、重叠和 typology(类型)，以及频率 bins 的数量。因此，最近的趋势是直接学习原始波形，从而完全避免任何特征提取步骤。该方法在语音[22-26]中显示了良好的前景，包括情绪任务[27]、speaker识别[32]、欺骗检测[31]和语音合成[28,29]。与SincNet类似，之前的一些工作也提出了对CNN filter 添加约束，例如强制它们在 specific bands (特定波段)上工作[41,42]。与提出的方法不同的是，后者的工作是根据声谱图特征进行操作，同时仍然学习 CNN 滤波器的所有$L$元素。在[43]中，使用了一组参数化高斯滤波器，探索了与所提方法相关的思想。该方法对谱图域进行处理，而SincNet直接考虑原始时域波形。

据我们所知，这项研究是第一次显示了使用卷积神经网络对原始波形进行时域音频处理的 sinc filters 的有效性。过去的一些研究主要针对语音识别，而我们的研究主要针对语音识别的应用。SincNet学习的 compact (紧凑) 过滤器特别适合于 speaker 识别任务，特别是在每个 speaker 的训练数据只有几秒钟和用于测试的短句的现实场景中。

## 4. 实验设置

建议的 SincNet 已经在不同的语料库上进行了评估，并与许多 speaker 识别基线进行了比较。本着可重复研究的精神，我们使用诸如 Librispeech 这样的公共数据进行大多数的实验，并在 GitHub(https://github.com/mravanelli/SincNet/) 上发布 SincNet 的代码。在下面的部分中，将提供实验设置的概述。

## 4.1 Corpora

为了对不同数量的 speakers 数据集提供实验证据，本文考虑了 TIMIT (462 spks, train chunk)[44]和 Librispeech (2484 spks)[45]语料库。去掉每个句子开头和结尾的非语音间隔。内部沉默超过125毫秒的 Librispeech 语句被分成多个块。为了解决文本无关的speaker识别，TIMIT 的校准语句(即，所有 speakers 有相同文本的话语)已被删除。对于后一个数据集，每个 speaker 使用5个句子进行训练，其余3个句子用于测试。在 Librispeech 语料库中，训练和测试材料被随机选择，利用每个speaker 12-15秒的训练材料，测试2-6秒的句子。

## 4.2 SincNet 设置

每个语音句子的波形被分割成200ms的块(有10ms的重叠)，并输入到SincNet体系结构中。第一层使用长度为$L=251$个样本的80个 filters，第2节中描述了基于sincn的卷积实施。然后，该架构使用两个标准的卷积层，都使用60个长度为5的过滤器。层归一化[46]用于输入样本和所有卷积层(包括SincNet输入层)。接下来，我们使用三个由2048个神经元组成的全连接层，并使用批归一化[47]进行归一化。所有的隐层使用 leaky-ReLU[48]非线性。使用mel-scale cutoff 频率初始化sincs层的参数，而使用众所周知的“Glorot”初始化方案[49]初始化网络的其余部分。通过使用softmax分类器获得 Frame-level speaker分类，提供了一组目标speakers的后验概率。一个句子级别的分类是简单地通过平均帧预测和投票给speaker而得到的，这样可以最大化平均后验。

用RMSprop优化器训练,学习速率 $lr = 0.001$,$\alpha = 0.95$,$\epsilon = 10^{−7}$,minibatches大小128。该架构的所有超参数在TIMIT上进行了调优，并在 Librispeech 上进行了继承。

speaker 验证系统是由 speaker-id 神经网络考虑两种可能的设置。首先，我们考虑$d-vector$框架[13,21]，它依赖于最后一个隐含层的输出，计算测试和声明的speaker $d-vectors$之间的余弦距离。作为另一种解决方案(下称DNN-class)，speaker验证系统可以直接取与声明身份对应的softmax后验分数。这两种方法将在第5节中进行比较。

从冒名顶替者中随机选出10个话语，每个句子都来自一个真正的演讲者。请注意，为了评估我们在标准的开放集speaker-id任务中的方法，所有的冒名顶替者都来自一个与用于训练 speaker-id DNN不同的 speaker pool。

### 4.3 Baseline 设置

我们比较了SincNet与几个备选系统。首先，我们考虑由原始波形提供的标准CNN。这个网络基于与SincNet相同的架构，但是用一个标准的来代替基于SincNet的卷积。

与流行的手工制作特征进行了比较。为此，我们使用 Kaldi toolkit[50]计算了39个 MFCCs (13个 static +$\Delta$+$\Delta$$\Delta$)和40个 FBANKs。这些特征每25毫秒计算一次，有10毫秒的重叠，收集起来形成一个约200毫秒的上下文窗口(即，与考虑的基于波形的神经网络的上下文相似)。对 FBANK 特征使用CNN，对 MFCC 使用多层感知器(MLP)，FBANK网络采用层归一化，MFCC网络采用批归一化。这些网络的超参数也使用上述方法进行了调整。

对于 speaker 验证实验，我们也考虑了 i-vector 基线。用 SIDEKIT 工具包[51]对i-vector系统进行了简化。在 Librispeech 数据(避免测试和 enrollment (登记)语句)上训练 GMM-UBM 模型、总变率(Total Variability,TV)矩阵和概率线性判别分析(PLDA)。GMM-UBM 由2048个高斯分量组成，TV 和 PLDA 特征语音矩阵(eigenvoice matrix)的秩为400。注册和测试阶段在Librispeech上进行，使用与DNN实验相同的一组语音段。

## 5. 结果

本节报告所提出的SincNet的实验验证。首先，我们将使用SincNet学习的filters与使用标准CNN学习的filters进行比较。然后，我们将我们的体系结构与其他竞争系统在speaker识别和验证任务方面进行比较。

### 5.1 Filter 分析

![图2](/assets/images/nlp/sincnet/fig2.png)

图2：由一个标准的CNN和由提议的SincNet(使用Librispeech语料库)学到的过滤器的例子。第一行报告时域中的滤波器，第二行显示它们的幅频响应。

检查学习过的filters是一种有价值的实践，可以洞察网络实际上正在学习什么。图2展示了一些使用Librispeech数据集（0-4kHZ的频率响应被绘制）的标准CNN(图2a)和SincNet(图2b)学习滤波器的示例。从图中可以看出，标准的CNN并不总是学习具有明确频率响应的滤波器。在某些情况下，频率响应看起来有噪声(见图2a的第一个滤波器)，而在另一些情况下，假设有 multi-band shapes(见CNN图的第三个滤波器)。而 SincNet，是专门设计来实现矩形 band-pass filters，导致更有意义的CNN滤波器。

![图3](/assets/images/nlp/sincnet/fig3.png)

图3：SincNet filters 的累积频率响应

![图4](/assets/images/nlp/sincnet/fig4.png)

图4：在不同的训练阶段，SincNet和CNN模型的帧错误率(%)。在TIMIT上报告结果。

除了定性的检查外，重要的是要在 highlight 下，所学的 filters 所涵盖的 frequency bands (频带)。图3为SincNet和CNN学习的滤波器的累积频率响应。有趣的是，在SincNet图中有三个明显突出的主要峰值(参见图中的红线)。第一个对应于pitch(音高)区域(男性的平均音高为133Hz，女性为234Hz)。第二个峰值(大约位于500hz)主要捕捉第一个共振峰，其在各种英语元音上的平均值确实是500Hz。最后，第三个峰(从900到1400赫兹)捕捉到一些重要的第二共振峰，如元音$/a/$的第二共振峰，平均位于1100Hz。这种 filter-bank 配置表明，SincNet已成功地适应了其特点，以解决 speaker 识别。相反，标准的CNN没有表现出这样一种有意义的模式:CNN过滤器倾向于正确地聚焦在频谱的较低部分，但是调谐到第一和第二共振峰的峰值并没有清晰地出现。从图3可以看出，CNN曲线位于SincNet曲线之上。实际上，SincNet学习的过滤器，平均来说，比CNN的选择性更强，可能更好地捕捉 narrow-band speaker 的线索。

### 5.2. Speaker 识别

![表1](/assets/images/nlp/sincnet/tab1.png)

表1：在TIMIT(462 spks)和Librispeech(2484 spks)数据集上训练的 speaker 识别系统的分类错误率。SincNets 比竞争对手的性能好。

与标准CNN相比，SincNet的学习曲线如图4所示。在TIMIT数据集上得到的这些结果突出了使用SincNet时 Frame Error Rate(FER%)的更快降低。此外，SincNet接近于更好的性能，导致一个33.0%的FER对CNN基线的37.7%的FER实现。

表1报告了实现的分类错误率(CER%)。该表显示，SincNet在TIMIT和Librispeech数据集上都优于其他系统。在TIMIT上，原始波形与标准CNN的差距特别大，这证实了SincNet在训练数据较少的情况下的有效性。虽然LibriSpeech的使用减少了这一差距，我们仍然观察到4%的相对改善，也获得了更快的收敛(1200对1800 epochs)。标准FBANKs只在TIMIT上提供了与SincNet相当的结果，但在使用Librispech时比我们的架构差得多。在训练数据很少的情况下，网络不能比 FBANKs 更好地发现 filters，但是在数据较多的情况下，可以学习和利用定制的 filter-bank 来提高性能。

### 5.3 Speaker 验证

![表2](/assets/images/nlp/sincnet/tab2.png)

表2：在不同的系统上，Librispeech数据集的 speaker 验证 Equal Error Rate(EER%)。SincNets 比竞争对手更有竞争力。

作为最后一个实验，我们将验证扩展到 speaker verification。表2报告了使用 Librispeech 语料库获得的 Equal Error Rate (EER%)。所有DNN模型都显示出良好的性能，导致所有cases下，EER均低于1%。该表还强调，SincNet优于其他模型，显示了一个相对标准CNN模型约11%的性能改进。DNN-class 模型的性能明显优于 d-vector。尽管后一种方法很有效，但是必须为每一个添加到pool[32]中的新speaker训练(或调整)一个新的DNN模型。这使得该方法的执行效果更好，但不如d-vector灵活。

为了完整起见，还对标准i-vector进行了实验。虽然对这种技术进行详细的比较超出了本文的范围，但值得注意的是，我们最好的i-vector系统实现了EER=1.1%，远远低于DNN系统。众所周知，在文献中，当每个speaker使用更多的训练材料和使用更长的测试语句时，i-vector可以提供有竞争的绩效[52-54]。在这项工作所面临的挑战条件下，神经网络可以实现更好的泛化。

## 6. 结论与未来工作

本文提出了一种直接处理波形音频的神经网络结构SincNet。我们的模型受到数字信号处理中滤波方式的启发，通过有效的参数化对 filter shapes 施加约束。SincNet已经广泛地评估了挑战性的 speaker 识别和验证任务，对所有考虑的语料库显示了性能效益。

除了性能上的改进，SincNet还显著提高了与标准CNN相比的收敛速度，并且由于利用了filter对称，计算效率更高。对SincNet滤波器的分析表明，所学习的filter-bank 被调优，以精确地提取一些已知的重要 speaker 特性，如音高和共振峰。在未来的工作中，我们将评估SincNet在其他流行的 speaker 识别任务，如VoxCeleb。虽然本研究只针对 speaker 辨识，但我们相信所提出的方法定义了处理时间序列的一般范例，并可应用于许多其他领域。因此，我们未来的努力将致力于扩展到其他任务，如语音识别、情感识别、语音分离和音乐处理。

## 鸣谢

We would like to thank Gautam Bhattacharya, Kyle Kastner, Titouan Parcollet, Dmitriy Serdyuk, Maurizio Omologo, and Renato De Mori for their helpful comments. This research was enabled in part by support provided by Calcul Que ́bec and Compute Canada.

## 7. 引用

1. H. Beigi, Fundamentals of Speaker Recognition, Springer, 2011.
2. N. Dehak, P. J. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, “Front-end factor analysis for speaker verifi- cation,” IEEE Transactions on Audio, Speech, and Lan- guage Processing, vol. 19, no. 4, pp. 788–798, 2011.
3. D. A. Reynolds, T. F. Quatieri, and R. B. Dunn, “Speaker verification using adapted Gaussian mixture models,” Digital Signal Processing, vol. 10, no. 1–3, pp. 19–41, 2000.
4. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, MIT Press, 2016.
5. D. Yu and L. Deng, Automatic Speech Recognition - A Deep Learning Approach, Springer, 2015.
6. G. Dahl, D. Yu, L. Deng, and A. Acero, “Context- dependent pre-trained deep neural networks for large vocabulary speech recognition,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 1, pp. 30–42, 2012.
7. M. Ravanelli, Deep learning for Distant Speech Recog- nition, PhD Thesis, Unitn, 2017.
8. M. Ravanelli, P. Brakel, M. Omologo, and Y. Bengio, “A network of deep neural networks for distant speech recognition,” in Proc. of ICASSP, 2017, pp. 4880–4884.
9. M. McLaren, Y. Lei, and L. Ferrer, “Advances in deep neural network approaches to speaker recognition,” in Proc. of ICASSP, 2015, pp. 4814–4818.
10. F. Richardson, D. Reynolds, and N. Dehak, “Deep neu- ral network approaches to speaker and language recog- nition,” IEEE Signal Processing Letters, vol. 22, no. 10, pp. 1671–1675, 2015.
11. P. Kenny, V. Gupta, T. Stafylakis, P. Ouellet, and J. Alam, “Deep neural networks for extracting baum- welch statistics for speaker recognition,” in Proc. of Speaker Odyssey, 2014.
12. S. Yaman, J. W. Pelecanos, and R. Sarikaya, “Bot- tleneck features for speaker recognition,” in Proc. of Speaker Odyssey, 2012, pp. 105–108.
13. E. Variani, X. Lei, E. McDermott, I. L. Moreno, and J. Gonzalez-Dominguez, “Deep neural networks for small footprint text-dependent speaker verification,” in Proc. of ICASSP, 2014, pp. 4052–4056.
14. G. Heigold, I. Moreno, S. Bengio, and N. Shazeer, “End-to-end text-dependent speaker verification,” in Proc. of ICASSP, 2016, pp. 5115–5119.
15. D. Snyder, P. Ghahremani, D. Povey, D. Romero, Y. Carmiel, and S. Khudanpur, “Deep neural network- based speaker embeddings for end-to-end speaker veri- fication,” in Proc. of SLT, 2016, pp. 165–170.
16. D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust dnn embeddings for speaker recognition,” in Proc. of ICASSP, 2018.
17. F. Richardson, D. A. Reynolds, and N. Dehak, “A unified deep neural network for speaker and language recognition,” in Proc. of Interspeech, 2015, pp. 1146– 1150.
18. D. Snyder, D. Garcia-Romero, D. Povey, and S. Khu- danpur, “Deep neural network embeddings for text- independent speaker verification,” in Proc. of Inter- speech, 2017, pp. 999–1003.
19. C. Zhang, K. Koishida, and J. Hansen, “Text- independent speaker verification based on triplet con- volutional neural network embeddings,” IEEE/ACM Trans. Audio, Speech and Lang. Proc., vol. 26, no. 9, pp. 1633–1644, 2018.
20. G. Bhattacharya, J. Alam, and P. Kenny, “Deep speaker embeddings for short-duration speaker verification,” in Proc. of Interspeech, 2017, pp. 1517–1521.
21. A. Nagrani, J. S. Chung, and A. Zisserman, “Voxceleb: a large-scale speaker identification dataset,” in Proc. of Interspech, 2017.
22. D. Palaz, M. Magimai-Doss, and R. Collobert, “Analy- sis of CNN-based speech recognition system using raw speech as input,” in Proc. of Interspeech, 2015.
23. T. N. Sainath, R. J. Weiss, A. W. Senior, K. W. Wilson, and O. Vinyals, “Learning the speech front-end with raw waveform CLDNNs,” in Proc. of Interspeech, 2015.
24. Y. Hoshen, R. Weiss, and K. W. Wilson, “Speech acous- tic modeling from raw multichannel waveforms,” in Proc. of ICASSP, 2015.
25. T. N. Sainath, R. J. Weiss, K. W. Wilson, A. Narayanan, M. Bacchiani, and A. Senior, “Speaker localization and microphone spacing invariant acoustic modeling from raw multichannel waveforms,” in Proc. of ASRU, 2015.
26. Z. Tu ̈ske, P. Golik, R. Schlu ̈ter, and H. Ney, “Acous- tic modeling with deep neural networks using raw time signal for LVCSR,” in Proc. of Interspeech, 2014.
27. G. Trigeorgis, F. Ringeval, R. Brueckner, E. Marchi, M. A. Nicolaou, B. Schuller, and S. Zafeiriou, “Adieu features? end-to-end speech emotion recognition using a deep convolutional recurrent network,” in Proc. of ICASSP, 2016, pp. 5200–5204.
28. A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” in Arxiv, 2016.
29. S. Mehri, K. Kumar, I. Gulrajani, R. Kumar, S. Jain, J. Sotelo, A. C. Courville, and Y. Bengio, “Samplernn: An unconditional end-to-end neural audio generation model,” CoRR, vol. abs/1612.07837, 2016.
30. P. Ghahremani, V. Manohar, D. Povey, and S. Khudan- pur, “Acoustic modelling from the signal domain using CNNs,” in Proc. of Interspeech, 2016.
31. H. Dinkel, N. Chen, Y. Qian, and K. Yu, “End-to- end spoofing detection with raw waveform CLDNNS,” Proc. of ICASSP, pp. 4860–4864, 2017.
32. H. Muckenhirn, M. Magimai-Doss, and S. Marcel, “To- wards directly modeling raw speech signal for speaker verification using CNNs,” in Proc. of ICASSP, 2018.
33. J.-W. Jung, H.-S. Heo, I.-H. Yang, H.-J. Shim, , and H.- J. Yu, “A complete end-to-end speaker verification sys- tem using deep neural networks: From raw signals to verification result,” in Proc. of ICASSP, 2018.
34. J.-W. Jung, H.-S. Heo, I.-H. Yang, H.-J. Shim, and H.-J. Yu, “Avoiding Speaker Overfitting in End-to- End DNNs using Raw Waveform for Text-Independent Speaker Verification,” in Proc. of Interspeech, 2018.
35. L. R. Rabiner and R. W. Schafer, Theory and Applica- tions of Digital Speech Processing, Prentice Hall, NJ, 2011.
36. S. K. Mitra, Digital Signal Processing, McGraw-Hill, 2005.
37. J. Chung, C ̧. Gu ̈lc ̧ehre, K. Cho, and Y. Bengio, “Em- pirical evaluation of gated recurrent neural networks on sequence modeling,” in Proc. of NIPS, 2014.
38. M. Ravanelli, P. Brakel, M. Omologo, and Y. Bengio, “Improving speech recognition by revising gated recur- rent units,” in Proc. of Interspeech, 2017.
39. M. Ravanelli, P. Brakel, M. Omologo, and Y. Ben- gio, “Light gated recurrent units for speech recogni- tion,” IEEE Transactions on Emerging Topics in Com- putational Intelligence, vol. 2, no. 2, pp. 92–102, April 2018.
40. M. Ravanelli, D. Serdyuk, and Y. Bengio, “Twin reg- ularization for online speech recognition,” in Proc. of Interspeech, 2018.
41. T. N. Sainath, B. Kingsbury, A. R. Mohamed, and B. Ramabhadran, “Learning filter banks within a deep neural network framework,” in Proc. of ASRU, 2013, pp. 297–302.
42. H. Yu, Z. H. Tan, Y. Zhang, Z. Ma, and J. Guo, “DNN Filter Bank Cepstral Coefficients for Spoofing Detec- tion,” IEEE Access, vol. 5, pp. 4779–4787, 2017.
43. H. Seki, K. Yamamoto, and S. Nakagawa, “A deep neural network integrated with filterbank learning for speech recognition,” in Proc. of ICASSP, 2017, pp. 5480–5484.
44. J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fis- cus, D. S. Pallett, and N. L. Dahlgren, “DARPA TIMIT Acoustic Phonetic Continuous Speech Corpus CDROM,” 1993.
45. V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An ASR corpus based on public domain audio books,” in Proc. of ICASSP, 2015, pp. 5206–5210.
46. J. Ba, R. Kiros, and G. E. Hinton, “Layer normaliza- tion,” CoRR, vol. abs/1607.06450, 2016.
47. S. Ioffe and C. Szegedy, “Batch normalization: Acceler- ating deep network training by reducing internal covari- ate shift,” in Proc. of ICML, 2015, pp. 448–456.
48. A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier nonlinearities improve neural network acoustic models,” in Proc. of ICML, 2013.
49. X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in Proc. of AISTATS, 2010, pp. 249–256.
50. D. Povey et al., “The Kaldi Speech Recognition Toolkit,” in Proc. of ASRU, 2011.
51. A. Larcher, K. A. Lee, and S. Meignier, “An extensi- ble speaker identification sidekit in python,” in Proc. of ICASSP, 2016, pp. 5095–5099.
52. A.K.Sarkar,DMatrouf,P.M.Bousquet,andJ.F.Bonas- tre, “Study of the effect of i-vector modeling on short and mismatch utterance duration for speaker verifica- tion,” in Proc. of Interspeech, 2012, pp. 2662–2665.
53. R. Travadi, M. Van Segbroeck, and S. Narayanan, “Modified-prior i-Vector Estimation for Language Iden- tification of Short Duration Utterances,” in Proc. of In- terspeech, 2014, pp. 3037–3041.
54. A. Kanagasundaram, R. Vogt, D. Dean, S. Sridharan, and M. Mason, “i-vector based speaker recognition on short utterances,” in Proc. of Interspeech, 2011, pp. 2341–2344.

---
**参考**：
1. 论文：Mirco Ravanelli, Yoshua Bengio [SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET](https://arxiv.org/abs/1808.00158v3)
