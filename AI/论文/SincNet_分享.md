---
typora-root-url: ../../../
---
# SincNet 分享
## 个人总结：

在语音领域，以往多是在手工处理的特征基础上再进行更上层的处理。但手工设计的特征往往会丢失一些和任务有关的关键的特征。所以最新的研究倾向于直接在 spectrogram bins(声谱图箱)，甚至 raw waveforms (原始波形) 上应用神经网络。最流行的是采用CNNs来处理。本文认为基于波形的CNNs最关键的是第一层卷积。

于是作者设计了新的 filter,对 filter shape 增加了约束，通过一组参数化的 sinc 函数（辛格函数）实现 band-pass filters （带通滤波器就是让某个频率范围的波通过，不是这个范围的波衰减掉）

设计的 filter 具有参数量少，可解释性强，收敛快的特点

## SincNet 架构

![图1](/assets/images/nlp/sincnet/fig1.png)

### 标准CNN

标准CNN执行的时域卷积

$$
\tag{1}
y[n] = x[n]*h[n] = \sum_{l=0}^{L-1} x[l]\cdot h[n-l]
$$

其中$x[n]$为语音信号块,$h[n]$为长度$L$的 filter,$y[n]$为滤波后的输出。在标准CNNs中，每个 filter 的所有$L$元素(elements,taps)都是从数据中学习的。相反,该SincNet(图1中所示)用一个预定义的函数$g$执行卷积，$g$只依赖几个可学习的参数$\theta$，在下面方程中凸显:

$$
\tag{2}
y[n]=x[n]*g[n,\theta]
$$

A reasonable choice, inspired by standard filtering in digital signal processing, is to define $g$ such that a filter-bank composed of rectangular bandpass filters is employed. In the frequency domain, the magnitude of a generic bandpass filter can be written as the difference between two low-pass filters:

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

where $f_1$ and $f_2$ are the learned low and high cutoff frequencies, and $rect(\cdot)$ is the rectangular function in the magnitude frequency domain. After returning to the time domain (using the inverse Fourier transform [35]), the reference function $g$ becomes:

$$
\tag{4}
g[n,f_1,f_2] = 2f_2 sinc(2\pi f_2 n)-2f_1 sinc(2\pi f_1 n)
$$

其中 sinc 函数定义为$sinc(x)=sin(x)/x$。

The cut-off frequencies can be initialized randomly in the range $[0,f_s/2]$, where $f_s$ represents the sampling frequency of the input signal. As an alternative, filters can be initialized with the cutoff frequencies of the mel-scale filter-bank, which has the advantage of directly allocating more filters in the lower part of the spectrum, where many crucial clues about the speaker identity are located. To ensure $f_1\ge 0$ and $f_2\ge f_1$, the previous equation is actually fed by the following parameters:

$$
\tag{5}
f_1^{abs} = |f_1|
$$

$$
\tag{6}
f_2^{abs} = f_1 + |f_2 - f_1|
$$

Note that no bounds have been imposed to force $f_2$ to be smaller than the Nyquist frequency, since we observed that this constraint is naturally fulfilled during training. Moreover, the gain of each filter is not learned at this level. This parameter is managed by the subsequent layers, which can easily attribute more or less importance to each filter output.

An ideal bandpass filter (i.e., a filter where the passband is perfectly flat and the attenuation in the stopband is infinite) requires an infinite number of elements $L$. Any truncation of $g$ thus inevitably leads to an approximation of the ideal filter, characterized by ripples in the passband and limited attenuation in the stopband. A popular solution to mitigate this issue is windowing [35]. Windowing is performed by multiplying the truncated function $g$ with a window function $w$, which aims to smooth out the abrupt discontinuities at the ends of $g$:

$$
\tag{7}
g_w[n,f_1,f_2] = g[n,f_1,f_2]\cdot w[n]
$$

This paper uses the popular Hamming window [36], defined as follows:

$$
\tag{8}
w[n] = 0.54 - 0.46 \cdot cos(\frac{2\pi n}{L})
$$

The Hamming window is particularly suitable to achieve high frequency selectivity [36]. However, results not reported here reveals no significant performance difference when adopting other functions, such as Hann, Blackman and Kaiser win- dows. Note also that the filters g are symmetric and thus do not introduce any phase distortions. Due to the symmetry, the filters can be computed efficiently by considering one side of the filter and inheriting the results for the other half.

All operations involved in SincNet are fully differentiable and the cutoff frequencies of the filters can be jointly opti- mized with other CNN parameters using Stochastic Gradi- ent Descent (SGD) or other gradient-based optimization rou- tines. As shown in Fig. 1, a standard CNN pipeline (pooling, normalization, activations, dropout) can be employed after the first sinc-based convolution. Multiple standard convolu- tional, fully-connected or recurrent layers [37–40] can then be stacked together to finally perform a speaker classification with a softmax classifier.

## 模型性质

1. 快速收敛
   
   SincNet强制网络只考虑对性能有重大影响的 filter 参数。

2. 参数量少
   
   SincNet 大大减少了第一个卷积层的参数数量。例如，如果我们考虑一个长度为$L$的$F$ filters 组成的层，一个标准的 CNN 使用$F\cdot L$参数，而 SincNet 考虑的是$2F$。
3. 可解释
   
   在第一个卷积层中获得的SincNet特征映射显然更具有可移植性和可读性。事实上，filter bank 只依赖于具有明确物理意义的参数。

## 模型效果

![表1](/assets/images/nlp/sincnet/tab1.png)

表1：在TIMIT(462 spks)和Librispeech(2484 spks)数据集上训练的 speaker 识别系统的分类错误率。SincNets 比竞争对手的性能好。

![表2](/assets/images/nlp/sincnet/tab2.png)

表2：在不同的系统上，Librispeech数据集的 speaker 验证 Equal Error Rate(EER%)。SincNets 比竞争对手更有竞争力。