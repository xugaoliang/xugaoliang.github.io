---
typora-root-url: ../../../
---
## Jasper 分享
## 个人总结：

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

## 论文介绍

## contributions

1. We present a computationally efficient end-to-end convolutional neural network acoustic model.
2. We show ReLU and batch norm outperform other combi-nations for regularization and normalization, and residual connections are necessary for training to converge.
3. We introduce NovoGrad, a variant of the Adam optimizer with a smaller memory footprint.
4. We improve the SOTA WER on LibriSpeech test-clean.

## Jasper Architecture

Jasper has a block architecture: a Jasper BxR model has B blocks, each with R sub-blocks. Each sub-block applies the following operations: a 1D- convolution, batch norm, ReLU, and dropout. All sub-blocks in a block have the same number of output channels.

Each block input is connected directly into the last sub-block via a residual connection. The residual connection is first projected through a 1x1 convolution to account for different numbers of input and output channels, then through a batch norm layer. The output of this batch norm layer is added to the output of the batch norm layer in the last sub-block. The result of this sum is passed through the activation function and dropout to produce the output of the current block.

![图1](/assets/images/nlp/jasper/fig1.png)


![表1](/assets/images/nlp/jasper/tab1.png)

We also build a variant of Jasper, Jasper Dense Residual (DR). Jasper DR follows DenseNet and DenseRNet, but instead of having dense connections within a block, the out- put of a convolution block is added to the inputs of all the fol- lowing blocks. While DenseNet and DenseRNet concatenates the outputs of different layers, Jasper DR adds them in the same way that residuals are added in ResNet. As explained below, we find addition to be as effective as concatenation.

![图2](/assets/images/nlp/jasper/fig2.png)

## NormalizationandActivation

In our study, we evaluate performance of models with:
* 3 types of normalization: batch norm [11], weight norm [10], and layer norm [18]
* 3 types of rectified linear units: ReLU, clipped ReLU (cReLU), and leaky ReLU (lReLU)
* 2 types of gated units: gated linear units (GLU) [9], and gated activation units (GAU) [19]

We first experimented with a smaller Jasper5x33 model to pick the top 3 settings before training on larger Jasper models. We found that layer norm with GAU performed the best on the smaller model.

For larger models, we noticed that batch norm with ReLU outperformed other choices. Thus, leading us to decide on batch normalization and ReLU for our architecture.

During batching, all sequences are padded to match the longest sequence. These padded values caused issues when us- ing layer norm. We applied a sequence mask to exclude padding values from the mean and variance calculation. 

In addition to masking layer norm, we additionally applied masking prior to the convolution operation, and masking the mean and variance calculations in batch norm. These results are shown in Table 3. Interestingly, we found that while masking before convolution gives a lower WER, using masks for both convolutions and batch norm results in worse performance.


![表3](/assets/images/nlp/jasper/tab3.png)

## NovoGrad

For training, we use either Stochastic Gradient Descent (SGD) with momentum or our own NovoGrad, an optimizer similar to Adam [15], except that its second moments are computed per layer instead of per weight. Compared to Adam, it reduces memory consumption and we find it to be more numerically stable.


At each step $t$，NovoGrad computes the stochastic gradi- ent gtl following the regular forward-backward pass. Then the second-order moment $v_t^l$ is computed for each layer $l$ similar to ND-Adam:

$$
v_t^l = \beta_2 \cdot v_{t-1}^l + (1-\beta_2) \cdot||g_t^l||^2
$$

second-order moment $v_t^l$ is used to re-scale gradients  $g_t^l$ before calculating the first-order moment $m_t^l$:

$$
m_t^l = \beta_1 \cdot m_{t-1}^l + \frac{g_t^l}{\sqrt{v_t^l + \epsilon}}
$$

![表5](/assets/images/nlp/jasper/tab5.png)

表5：LibriSpeech,WER(%)

If L2-regularization is used, a weight decay $d\cdot w_t$ is added to the re-scaled gradient (as in AdamW [30]):

$$
m_t^l = \beta_1 \cdot m_{t-1}^l + \frac{g_t^l}{\sqrt{v_t^l + \epsilon}} + d\cdot w_t
$$

Finally, new weights are computed using the learning rate  $\alpha_t$ :

$$
w_{t+1} = w_t - \alpha_t \cdot m_t
$$
Using NovoGrad instead of SGD with momentum, we de- creased the WER on dev-clean LibriSpeech from 4.00% to 3.64%, a relative improvement of 9% for Jasper DR 10x5.