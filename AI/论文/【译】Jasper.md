---
typora-root-url: ../../../
---

# 【译】Jasper:一个端到端的卷积神经声学模型

* 作者：Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gadde
* 论文：《Jasper: An End-to-End Convolutional Neural Acoustic Model》
* 地址：https://arxiv.org/abs/1904.03288

---
个人总结：

Jasper 虽然声明为端到端的卷积神经声学模型，但准确地说应该算端到端模型的其中一部分，如图：

![ctc_asr](/assets/images/nlp/jasper/ctc_asr.png)

该图来源于：https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition.html

只是理解Jasper，对于ASR的初学者，并不能明白整个端到端的过程。

Jasper的作者说灵感来自于 wav2letter，所以wav2letter需要去了解一下。另外预处理和最后的损失定义及解码器，jasper中都没有介绍，需要另外找资料了解。

Jasper 自己声明的主要贡献为：

1. 提出了一种计算效率高的端对端卷积神经网络声学模型。
2. 我们发现ReLU和批归一化超越了其他正则和归一化的组合，为了使训练收敛，残差连接是必须的。
3. 我们介绍了NovoGrad，它是Adam 优化器[15]的一个变种，内存占用更小。
4. 我们在 LibriSpeech test-clean上提升了 SOTA WER

Jasper 的结构: 一个 Jasper $B\times R$模型有$B$个blocks,每个block有 $R$个sub-blocks.每个 sub-block 应用以下操作:一维卷积、批归一化、ReLU和dropout。一个block中的所有sub-blocks具有相同数量的输出通道。

每个block输入通过残差连接直接连接到最后一个子块。残差连接首先通过1x1卷积进行投影，以解决输入和输出通道的不同数量，然后通过批归一化层。该批归一化层的输出被添加到最后一个子块的批归一化层的输出中。和的结果通过激活函数和dropout生成当前块的输出。

其中可借鉴的经验就是：深度堆叠+残差，ReLU+BN，更稳定的优化器NovoGrad,但该优化器的实现对小白来说没有现成的。

---

## 摘要

本文报道了在没有任何训练数据的情况下，端到端语音识别模型在LibriSpeech方面的最新研究成果。我们的模型Jasper只使用了1D卷积、批量归一化、ReLU、dropout和残差连接。为了改进训练，我们进一步引入了一种新的分层优化器NovoGrad。通过实验，我们证明了所提出的深度架构与更复杂的选择具有相同或更好的性能。我们最深的Jasper版本使用了54个卷积层。在LibriSpeech test-clean中，我们使用一个带有外部神经语言模型的波束搜索解码器(beam-search decoder)实现了2.95%的WER，使用一个贪婪解码器(greedy decoder)实现了3.86%的WER。我们还在《华尔街日报》和Hub500对话评估数据集上报告了竞争结果。

Index Terms: speech recognition, convolutional networks, time-delay neural networks

## 1. 介绍

常见的自动语音识别(ASR)系统包括几个独立的学习组件:一个声学模型来预测上下文相关的子音素状态(sub-phoneme states,senones)，一个图形结构来映射 senones 到音素，一个语音模型来映射音素到单词。混合系统将隐马尔可夫模型与神经网络相结合，建立状态依赖关系模型来预测状态[1,2,3,4]。较新的方法如端到端(E2E)系统降低了最终系统的整体复杂性。

我们的研究建立在利用时延神经网络(TDNN)、其他形式的卷积神经网络和连接时间分类(CTC)损失的前期工作的基础上[5,6,7]。我们的灵感来自 wav2letter[7]，它使用了一维卷积层。Liptchinsky等人通过将模型深度增加到19个卷积层，并添加门控线性单元(GLU)[9]、权重归一化[10]和dropout，对 wav2letter 进行了改进。

通过建立一个更深入和更大的容量网络，我们的目标是证明我们可以匹配或超越LibriSpeech和 2000hr Fisher+Switchboard 任务的非端到端模型。与wav2letter一样，我们的架构Jasper使用了一组一维卷积层的堆叠，但是使用了ReLU和批归一化[11]。我们发现，ReLU和批归一化优于我们在卷积ASR中测试的其他激活和归一化方案。因此，Jasper的架构只包含1D卷积层、批归一化层、ReLU层和dropout层，这些操作都是针对gpu的训练和推理进行了高度优化的。

可以通过堆叠这些操作来增加Jasper模型的容量。我们最大的版本使用54卷积层(333M参数)，而我们的小模型使用34层(201M参数)。我们使用残差连接让这个深度级别可用。我们研究了许多残差选项，并提出了一种新的残差连接拓扑，我们称之为密集残差(Dense Residual,DR)。

将我们最好的声学模型与Transformer-XL[12]语言模型相集成，可以使我们在LibriSpeech[13]测试中获得2.95% WER的最新水平(SOTA)结果，并在LibriSpeech test-other端到端模型中获得SOTA结果。我们在华尔街日报(Wall Street Journal,WSJ)和2000hr Fisher+Switchboard(F+S)上展示了具有竞争力的结果。在LibriSpeech test-clean中，只使用贪心解码而不使用语言模型，得到3.86% WER 的测试结果。

本文的贡献如下:
1. 提出了一种计算效率高的端对端卷积神经网络声学模型。
2. 我们发现ReLU和批归一化超越了其他正则和归一化的组合，为了使训练收敛，残差连接是必须的。
3. 我们介绍了NovoGrad，它是Adam 优化器[15]的一个变种，内存占用更小。
4. 我们在 LibriSpeech test-clean上提升了 SOTA WER

## 2. Jasper 架构

Jasper 是一个端到端的ASR模型家族，它用卷积神经网络取代了 acoustic 和 pronunciation 模型。Jasper使用 mel-filterbank 功能，从20ms的窗口计算出10ms的重叠，并输出一个每帧对字符的贡献的概率分布。**Jasper 有一个block结构: 一个 Jasper $B\times R$模型有$B$个blocks,每个block有 $R$个sub-blocks.每个 sub-block 应用以下操作:一维卷积、批归一化、ReLU和dropout。一个block中的所有sub-blocks具有相同数量的输出通道。**

**每个block输入通过残差连接直接连接到最后一个子块。残差连接首先通过1x1卷积进行投影，以解决输入和输出通道的不同数量，然后通过批归一化层。该批归一化层的输出被添加到最后一个子块的批归一化层的输出中。和的结果通过激活函数和dropout生成当前块的输出。**

Jasper的sub-block架构设计，实现了快速GPU推理。每个sub-block都可以融合到一个单独的GPU内核中:在推理时不使用dropout，dropout被消除了，批归一化与之前的卷积融合，ReLU截断结果，在融合操作中，残差求和作为一个修改过的偏置项。

所有的Jasper模型都有四个额外的卷积块:一个 pre-processing 块和三个 post-processing 块。详见图1和表1。

![图1](/assets/images/nlp/jasper/fig1.png)

图1：Jasper $B\times R$模型：$B$个 blocks,$R$个 sub-blocks

![表1](/assets/images/nlp/jasper/tab1.png)

表1：Jasper $10\times 5$,10个blocks,每个由5个1D卷积 sub-blocks 组成，加4个 additional blocks

我们还构建了一个变种的Jasper，Jasper Dense Residual (DR)。Jasper DR遵循DenseNet[16]和DenseRNet[17]，但不是在一个块中有密集的连接，而是将一个卷积块的输出添加到所有后续块的输入中。DenseNet和DenseRNet将不同层的输出连接起来，Jasper DR以与ResNet中添加残差相同的方式添加它们。正如下面所解释的，我们发现加法和拼接一样有效。

![图2](/assets/images/nlp/jasper/fig2.png)

图2：Jasper Dense Residual

### 2.1 归一化与激活

在我们的研究中，我们使用以下方法来评估模型的性能:
* 3类归一化:批归一化[11]，权重归一化[10]，层归一化[18]
* 3种类型的整流线性单元（rectified linear units）:ReLU, clipped ReLU (cReLU)，和leaky ReLU (lReLU)
* 2类门控单元:门控线性单元(GLU)[9]和门控激活单元(GAU) [19]

实验结果如表2所示。我们首先使用一个较小的Jasper5x3模型挑选前3个设置，然后在较大的Jasper模型上进行训练。我们发现，**在小模型上，带GAU的层归一化表现最好**。在我们的测试中，带ReLU的层归一化和带ReLU的批归一化分别排在第二位和第三位。利用这3个，我们在一个更大的Jasper10x4上进行了进一步的实验。**对于较大的型号，我们注意到带ReLU的批归一化优于其他选择。因此，我们决定为我们的架构使用批归一化和ReLU。**

在批处理过程中，填充所有序列以匹配最长的序列。这些填充的值会导致在执行层归一化时出现问题。**我们应用了一个序列遮蔽来排除平均值和方差计算中的填充值**。此外，我们还计算了时间维度和通道上的均值和方差，类似于Laurent等人提出的按序列归一化方法。**除了遮蔽层归一化外，我们还在卷积运算之前附加了遮蔽，并在批归一化中遮蔽平均值和方差的计算**。这些结果如表3所示。有趣的是，我们发现**在卷积之前使用遮蔽会得到更低的WER**，而在卷积和批归一化中使用遮蔽会导致更差的性能。

最后，我们发现权重归一化的训练是非常不稳定的，会导致爆炸性激活。

![表2](/assets/images/nlp/jasper/tab2.png)

表2：归一化和激活：50 epochs 后的 Greedy WER, LibriSpeech 

![表3](/assets/images/nlp/jasper/tab3.png)

表3：序列遮蔽：归一化和激活：50 epochs 后 Jasper $10\times 4$ 的 Greedy WER, LibriSpeech 

### 2.2 残差连接

对于比Jasper 5x3更深的模型，我们一致地观察到，残差连接对于训练的收敛是必要的。除了上述简单的残差和密集残差模型外，我们还研究了Jasper的DenseNet[16]和DenseRNet[17]变种。两者都将每个sub-block的输出连接到一个block中的后续sub-blocks的输入。与Dense Residual 类似，DenseRNet将每个block的输出连接到所有后续blocks的输入。DenseNet和DenseRNet使用拼接的方式合并残差连接，而 Residual 和 Dense Residual 使用加法。我们发现 Dense Residual 和 DenseRNet 在 LibriSpeech 的特定子集上表现相似，并且每一个都表现得更好。我们决定在后续的实验中使用 Dense Residual。主要原因是由于拼接作用，DenseNet和DenseRNet的 growth factor 需要针对更深层次的模型进行调优，而 Dense Residual 没有growth factor。

### 2.3 语言模型

语言模型(LM)是任意符号序列$P(w_1,\cdots,w_n)$上的概率分布，使得更可能的序列具有更高的概率。LMs常用于条件波束搜索(condition beam search)。在解码过程中，使用声学评分和LM评分对候选对象进行评估。在最近的研究中，传统的 N-gram LMs被神经LMs增强了[21,22,23]。

![表4](/assets/images/nlp/jasper/tab4.png)

表4：残差连接:Greedy WER,LibriSpeech,在Jasper $10\times 3$经过 400个 epochs 之后。所有模型的大小都具有大致相同的参数计数。

我们用统计的N-gram语言模型[24]和神经 Transformer-XL[12]模型进行了实验。我们的最佳结果使用声学和单词级的N-gram语言模型，使用宽度为2048的波束搜索(beam search)生成候选列表。接下来，外部 Transformer-XL LM重核最后的名单。所有的LMs都是在数据集上独立于声学模型进行训练的。我们在结果部分使用神经LM显示结果。我们观察到神经LM的质量(用 perplexity 测量)与WER之间有很强的相关性，如图3所示。

![图3](/assets/images/nlp/jasper/fig3.png)

图3：LM perplexity vs WER。LibriSpeech dev-other。不同的 perplexity 是通过在训练中拍摄较早或较晚的快照来实现的。

### 2.4 NovoGrad

对于训练，我们要么使用带有动量的随机梯度下降(SGD)，要么使用我们自己的NovoGrad，这是一种类似于Adam[15]的优化器，只是它的第二个 moments 是按每层计算的，而不是按每个权重计算的。与Adam相比，它减少了内存消耗，我们发现它在数值上更稳定。

在每个步骤$t$中，NovoGrad按照常规的前后向遍历计算随机梯度$g_t^l$。然后计算与ND-Adam[29]相似的各层$l$的 second-order moment $v_t^l$:

$$
v_t^l = \beta_2 \cdot v_{t-1}^l + (1-\beta_2) \cdot||g_t^l||^2
$$

second-order moment $v_t^l$ 用于在计算 first-order moment $m_t^l$ 之前重新缩放梯度$g_t^l$:

$$
m_t^l = \beta_1 \cdot m_{t-1}^l + \frac{g_t^l}{\sqrt{v_t^l + \epsilon}}
$$

![表5](/assets/images/nlp/jasper/tab5.png)

表5：LibriSpeech,WER(%)

如果使用L2正则化，则在重新缩放的梯度上添加一个权重衰减$d\cdot w_t$(如AdamW [30]):

$$
m_t^l = \beta_1 \cdot m_{t-1}^l + \frac{g_t^l}{\sqrt{v_t^l + \epsilon}} + d\cdot w_t
$$

最后,使用学习速率$\alpha_t$计算新的权重:

$$
w_{t+1} = w_t - \alpha_t \cdot m_t
$$

使用 NovoGrad 而不是带动量的SGD，我们降低了dev-clean 上 LibriSpeech的WER从4.00%到3.64%，Jasper DR 10x5相对改善9%。关于NovoGrad的更多细节和实验结果，请参见[31]。

## 3. 结果

我们通过不同领域的大量数据集对Jasper进行了评估。在所有的实验中，我们都使用dropout和权值衰减作为正则化。在训练时，我们使用固定 +/-10%[32] 的 3-fold speed perturbation(速度扰动)。对于《华尔街日报》和Hub500，我们使用一个范围在[-10%，10%]的随机速度扰动因子，作用在每个话语上，输入到模型中。所有的模型都在NVIDIA DGX-1上使用OpenSeq2Seq[34]进行混合精度[33]的训练。预训练的模型和训练配置是有用的，可查看“https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition.html "

### 3.1 Read Speech(朗读语料)

我们评估了Jasper在两个read speech 数据集上的表现:LibriSpeech 和 Wall Street Journal (WSJ)。对于LibriSpeech，我们使用我们的NovoGrad 优化器对Jasper DR 10x5进行了400个epoch的训练。在test-clean子集上实现了SOTA性能，在test-other端到端语音识别模型上实现了SOTA性能。

我们使用带有动量优化器的SGD对一个组合的WSJ数据集(80小时)上的400个epoch训练了一个更小的Jasper 10x3模型:LDC93S6A (WSJ0)和LDC94S13A (WSJ1)。结果如表6所示。

### 3.2 Conversational Speech(会话语料)

我们还在一个会话型英语语料库上对Jasper模型的表现进行了评价。Hub5 Year 2000 (Hub500)评价方法(LDC2002S09, LDC2002T43)在学术界得到了广泛的应用。它被分为两个子集:Switchboard(SWB)和Callhome(CHM)。声学和语言模型的培训数据包括2000hr Fisher+Switchboard训练数据(LDC2004S13, LDC2005S13, LDC97S62)。Jasper DR 10x5使用带动量的SGD进行了50个epoch的训练。我们与使用相同数据训练的其他模型进行比较，并在表7中报告Hub500结果。

![表6](/assets/images/nlp/jasper/tab6.png)

表6：WSJ 端到端模型，WER(%)

![表7](/assets/images/nlp/jasper/tab7.png)

表7：Hub500，WER(%)

我们得到了很好的结果。然而，对于像CHM这样的更困难的任务，我们还有很多工作要做。

## 4. 结论

提出了一种用于端到端语音识别的神经网络结构。受wav2letter卷积方法的启发，我们构建了一个深度的、可扩展的模型，它需要设计良好的残差拓扑、有效的正则化和强大的优化器。正如我们的体系结构研究所述，标准组件的组合导致了SOTA在LibriSpeech上的结果和在其他基准上的竞争结果。我们的Jasper架构在训练和推理方面非常高效，是一个很好的基线方法，在此基础上可以探索更复杂的正则化、数据迁移、损失函数、语言模型和优化策略。我们有兴趣看看我们的方法是否可以继续扩展到更深的模型和更大的数据集。

## 5. 引用
1. A.Waibel,T.Hanazawa,G.Hinton,K.Shirano,andK.Lang,“A time-delay neural network architecture for isolated word recogni- tion,” IEEE Trans. on Acoustics, Speech and Signal Processing, 1989.
2. Y.Bengio,R.DeMori,G.Flammia,andR.Kompe,“Globalopti- mization of a neural network-hidden markov model hybrid,” IEEE Transactions on Neural Networks, 3(2), 252259, 1992.
3. A. Graves and J. Schmidhuber, “Framewise phoneme classifica- tion with bidirectional lstm and other neural network architec- tures,” Neural Networks, vol. 18, pp. 602–610, 2005.
4. G.Hintonetal.,“Deepneuralnetworksforacousticmodelingin speech recognition,” IEEE Signal Processing Magazine, 2012.
5. A. Graves, S. Ferna ́ndez, F. Gomez, and J. Schmidhuber, “Con- nectionist temporal classification: labelling unsegmented se- quence data with recurrent neural networks,” in Proceedings of the 23rd international conference on Machine learning. ACM, 2006, pp. 369–376.
6. Y. Zhang et al., “Towards end-to-end speech recognition with deep convolutional neural networks,” in Interspeech 2016, 2016, pp. 410–414.
7. R. Collobert, C. Puhrsch, and G. Synnaeve, “Wav2letter: an end- to-end convnet-based speech recognition system,” arXiv preprint arXiv:1609.03193, 2016.
8. V. Liptchinsky, G. Synnaeve, and R. Collobert, “Letter- based speech recognition with gated convnets,” arXiv preprint arXiv:1712.09444, 2017.
9. Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language modeling with gated convolutional networks,” in Proceedings of the 34th International Conference on Machine Learning - Volume 70, ser. ICML’17. JMLR.org, 2017, pp. 933–941.
10. T. Salimans and D. P. Kingma, “Weight normalization: A simple reparameterization to accelerate training of deep neural networks,” in Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, Eds. Curran Associates, Inc., 2016, pp. 901–909.
11. S.IoffeandC.Szegedy,“Batchnormalization:Acceleratingdeep network training by reducing internal covariate shift,” CoRR, vol. abs/1502.03167, 2015.
12. Z. Dai et al., “Transformer-xl: Language modeling with longer-term dependency,” CoRR, vol. abs/1901.02860, 2018.
13. V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Lib- rispeech: an asr corpus based on public domain audio books,” in Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on. IEEE, 2015, pp. 5206–5210.
14. H.Hadian,H.Sameti,D.Povey,andS.Khudanpur,“End-to-end speech recognition using lattice-free mmi,” in Proc. Interspeech 2018, 2018, pp. 12–16.
15. D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” CoRR, vol. abs/1412.6980, 2014.
16. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” arXiv preprint arXiv:1608.06993, 2016.
17. J.Tang,Y.Song,L.Dai,andI.McLoughlin,“Acousticmodeling with densely connected residual network for multichannel speech recognition,” in Proc. Interspeech 2018, 2018, pp. 1783–1787.
18. J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” CoRR, vol. abs/1607.06450, 2016.
19. A. van den Oord et al., “Conditional image generation with pixelcnn decoders,” in Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, Eds. Curran Associates, Inc., 2016, pp. 4790–4798.
20. C. Laurent, G. Pereyra, P. Brakel, Y. Zhang, and Y. Bengio, “Batch normalized recurrent neural networks,” in 2016 IEEE Interna- tional Conference on Acoustics, Speech and Signal Processing (ICASSP), March 2016, pp. 2657–2661.
21. A. Zeyer, K. Irie, R. Schlter, and H. Ney, “Improved training of end-to-end attention models for speech recognition,” in Proc. Interspeech 2018, 2018, pp. 7–11.
22. D. Povey et al., “Semi-orthogonal low-rank matrix factorization for deep neural networks,” in Interspeech, 2018.
23. K. J. Han, A. Chandrashekaran, J. Kim, and I. R. Lane, “The CAPIO 2017 conversational speech recognition system,” CoRR, vol. abs/1801.00059, 2018.
24. K.Heafield,“Kenlm:Fasterandsmallerlanguagemodelqueries,” in Proceedings of the sixth workshop on statistical machine trans- lation. Association for Computational Linguistics, 2011, pp. 187–197.
25. X. Yang, J. Li, and X. Zhou, “A novel pyramidal-fsmn architecture with lattice-free MMI for speech recognition,” CoRR, vol. abs/1810.11352, 2018.
26. D. Amodei et al., “Deep speech 2: End-to-end speech recognition in english and mandarin,” in Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ser. ICML’16. JMLR.org, 2016, pp. 173–182.
27. N. Zeghidour et al., “Fully convolutional speech recognition,” CoRR, vol. abs/1812.06864, 2018.
28. D. S. Park et al., “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,” arXiv e-prints, 2019.
29. Z. Zhang, L. Ma, Z. Li, and C. Wu, “Normalized direction- preserving adam,” arXiv e-prints arXiv:1709.04546, 2017.
30. I.LoshchilovandF.Hutter,“Decoupledweightdecayregulariza- tion,” in International Conference on Learning Representations, 2019.
31. B. Ginsburg et al., “Stochastic Gradient Methods with Layer- wise Adaptive Moments for Training of Deep Networks,” arXiv e-prints, 2019.
32. K. Tom, P. Vijayaditya, P. Daniel, and K. Sanjeev, “Audio aug- mentation for speech recognition,” Interspeech 2015, 2015.
33. P. Micikevicius et al., “Mixed precision training,” arXiv preprint arXiv:1710.03740, 2017.
34. O. Kuchaiev et al., “Openseq2seq: extensible toolkit for dis- tributed and mixed precision training of sequence-to-sequence models,” , 2018.
35. Y. Zhang, W. Chan, and N. Jaitly, “Very deep convolutional net- works for end-to-end speech recognition,” in Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Con- ference on. IEEE, 2017.
36. C.Wengetal.,“Improvingattentionbasedsequence-to-sequence models for end-to-end english conversational speech recognition,” in Proc. Interspeech 2018, 2018, pp. 761–765.
37. E.Battenbergetal.,“Exploringneuraltransducersforend-to-end speech recognition,” in 2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), Dec 2017, pp. 206–213.

---
**参考**：
1. 论文：Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gadde [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/abs/1904.03288)
