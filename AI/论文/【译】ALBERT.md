---
typora-root-url: ../../../
---

# 【译】ALBERT: 一个用于语言表征自监督学习的轻量BERT

* 作者：Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut
* 论文：《ALBERT: A Lite BERT for Self-supervised Learning of Language Representations》
* 地址：https://arxiv.org/abs/1909.11942

---
个人总结：

ALBERT主要有3个贡献：
1. 因式分解词嵌入矩阵，将词汇先映射到低维空间，再映射到高维空间，以此减少参数。因为词嵌入是把词汇含义合理的聚集，100维的空间，全是01，就能表达$2^{100}$多个词汇，所以更高的空间不是必需的。
2. 跨层权重共享，进一步减少参数。这里有注意力层权重共享和前馈层权重共享。共享注意力层性能下降的少，甚至不下降，共享前馈层，性能下降的相对多些。个人认为不同注意力层所完成的任务是相同的，因此各层基本一样，共享不会让性能下降太多，而前馈层主要是做空间变换，输入不同，空间变换的结果也会不同，因为前馈层共享权重，导致性能下降的多些。总之，某种机制和输入无关，则可以共享，若和输入有关，则不能共享。前馈层比注意力层更依赖输入。真的是这样吗？应该有一定这样的原因吧。
3. 将下句预测任务（NSP)改为句子顺序预测任务（SOP)。提高句间含义的理解，因为NSP预测暗含了主题预测，NSP正确率高，大概率是因为两个句子描述的内容本身就是不同的主题。
---

## 摘要

在对自然语言表征进行预处理时，增加模型大小通常可以改善下游任务的性能。然而，由于GPU/TPU内存的限制、较长的训练时间和意料之外的模型退化，在某些情况下，进一步的模型增长会变得更加困难。为了解决这些问题，我们提出了两种参数约简技术（parameter-reduction techniques）来降低内存消耗和提高BERT的训练速度。综合的经验证据表明，我们提出的方法导致模型规模比原来的 BERT 更好。我们还使用了一种自监督的损失，它侧重于对句子间的连贯性进行建模，并表明它始终有助于下游任务的多句输入。因此，我们的最佳模型在GLUE、RACE和SQuAD基准上建立了新的最先进的结果，同时与 BERT-large 相比参数更少。代码和预培训的模型可以在 https://github.com/googl-research/albert 找到。

## 1. 介绍

全网络预训练(Dai & Le, 2015; Radford等人，2018; Devlin等人，2019; Howard & Ruder,2018)在语言表征学习方面取得了一系列突破。许多非凡的NLP任务，包括那些训练数据有限的任务，都从这些预训练的模型中获益良多。这些突破中最引人注目的一个是，在中国为初中和高中英语考试设计的阅读理解任务上，机器性能的演变。RACE (Reading Comprehension dataset collected from English Examinations) 测试(Lai 等, 2017):最初的论文描述了建模的挑战任务,报告了先进的机器精度为44.1%;最新公布的结果显示，他们的模型性能为83.2% (Liu 等, 2019);我们在这里展示的工作将其进一步提高到89.4%，45.3%的惊人改进，这主要归功于我们当前构建高性能预训练语言表征的能力。

来自这些改进的证据表明，大型网络对于实现最先进的性能至关重要(Devlin 等，2019; Radford 等，2019)。预训练大型模型并将其提炼为小型模型已成为普遍做法(Sun 等， 2019;Turc 等，2019)。考虑到模型大小的重要性，我们问:拥有更好的NLP模型和拥有更大的模型一样容易吗?

回答这个问题的一个障碍是可用硬件的内存限制。考虑到目前最先进的模型通常有数亿甚至数十亿的参数，当我们试图扩展我们的模型时，很容易突破这些限制。在分布式训练中，训练速度也会受到很大的影响，因为通信开销与模型中参数的数量成正比。我们还观察到，简单地增加一个模型的隐层维度大小，比如 BERT-large (Devlin 等， 2019)，会导致更糟糕的性能。表1和图1给出了一个典型的例子，在这个例子中，我们简单地将 BERT-large的隐藏大小增加2倍，而使用这个 BERT-xlarge 模型得到的结果更糟。

![图1](/assets/images/nlp/albert/fig1.png)

图1：BERT-large 和 BERT-xlarge(在隐层维度大小方面比 BERT-large 大2倍)的训练损失(左)和验证 MLM 正确率。较大的模型具有较低的 MLM 准确率，但没有明显的过拟合迹象。

![表1](/assets/images/nlp/albert/tab1.png)

表1： 提升 BERT-large 隐层维度大小导致了在 RACE 上更糟的性能

针对上述问题的现有解决方案包括模型并行化(Shoeybi 等， 2019)和智能内存管理(Chen 等，2016; Gomez 等，2017)。这些解决方案解决了内存限制问题，但没有解决通信开销和模型退化问题。在本文中，我们通过设计比传统 BERT 体系结构参数少得多的 A Lite BERT (ALBERT)体系结构来解决所有上述问题。

ALBERT 采用了两种参数约简技术，消除了在缩放预训练模型时的主要障碍。第一个是因式分解 embedding 参数。通过将大的词嵌入矩阵分解成两个小的矩阵，将隐藏层的大小与词嵌入的大小分离开来。这种分离使得在不显著增加词汇表嵌入的参数大小的情况下更容易地增加隐藏维度的大小。第二种技术是跨层参数共享。这种技术可以防止参数随着网络的深度而增长。这两种方法在不严重影响性能的前提下，显著地减少了BERT的参数数量，从而提高了参数效率。类似于 BERT-large 的 ALBERT 配置参数少了18倍，训练速度快了1.7倍。参数约简技术也作为正则化的一种形式，稳定了训练并有助于泛化。

为了进一步提高ALBERT的性能，我们还引入了一个用于句子顺序预测的自监督损失模型(SOP, sentence-order prediction)。SOP 主要关注句子间的连贯，旨在解决原 BERT 中提出的下句预测(NSP)损失(Yang 等， 2019;Liu 等，2019)的无效性。

作为这些设计决策的结果，我们能够扩展到更大的 ALBERT 配置，这些配置的参数仍然比 BERT-large 少，但是可以获得更好的性能。我们使自然语言理解在著名的GLUE、SQuAD 和 RACE 基准上建立了新的最先进的结果。具体来说，我们将 RACE 的正确率提高到89.4%，GLUE 基准提高到 89.4,SQuAD的F1提高到92.2。

## 2. 相关工作
### 2.1 扩大自然语言的表征学习

自然语言中学习表征已被证明对广泛的NLP任务有用，并被广泛采用(Mikolov 等， 2013;Le & Mikolov, 2014;Dai & Le，2015;Peters 等，2018;Devlin 人，2019;Radford 等，2018;2019)。过去两年最显著的变化之一是预训练词嵌入的转变，是否标准(Mikolov 等，2013;Pennington 等，2014）或者语境化(McCann 等，2017;Peters 等，2018)，然后是全网络预训练，然后是具体任务的微调(Dai & Le, 2015;Radford等人，2018;Devlin等人，2019)。在这一行工作中，经常可以看到较大的模型大小可以提高性能。例如，Devlin等人(2019)表明，在三个选定的自然语言理解任务中，使用更大的隐藏维度大小、更多的隐藏层和更多的注意力头总是会带来更好的性能。但是，它们的隐藏大小为1024。我们发现，在相同的设置下，将隐藏大小增加到2048会导致模型性能下降，从而导致性能下降。因此，扩大自然语言的表征学习并不像简单地增加模型大小那么容易。

此外，由于计算方面的限制，特别是在GPU/TPU内存方面的限制，大型模型很难进行实验。考虑到目前最先进的模型通常有数亿甚至数十亿的参数，我们很容易突破内存限制。为了解决这个问题，Chen等人(2016)提出了一种称为梯度检查点的方法，以牺牲额外的前向传递为代价，将内存需求降低为次线性。Gomez等人(2017)提出了一种从下一层重建每一层激活的方法，这样它们就不需要存储中间激活。这两种方法都以牺牲速度为代价来减少内存消耗。相比之下，我们的参数约简技术减少了内存消耗，提高了训练速度。

### 2.2 跨层参数共享

跨层共享参数的想法之前已经在 Transformer 架构中进行了探索(Vaswani等人，2017)，但是之前的工作主要针对标准的编码器和解码器任务进行训练，而不是针对预训练/微调设置的。与我们的观察不同，Dehghani等人(2018)表明，具有跨层参数共享(Universal Transformer, UT)的网络在语言建模和主动词一致方面比标准 Transformer 有更好的性能。最近，Bai等人(2019)提出了一种转换网络的深度均衡模型(Deep Equilibrium Model, DQE)，并证明了DQE可以达到一个平衡点，在这个平衡点上，某个层的输入嵌入和输出嵌入保持不变。我们的观察表明，嵌入是振荡的，而不是收敛的。Hao等人(2019)将参数共享 transformer 与标准 transformer 相结合，进一步增加了标准 transformer 的参数数量。

### 2.3 句子排序目标

ALBERT 使用了一个基于预测两个连续文本片段排序的预训练损失。一些研究人员已经尝试了与话语连贯相关的预训练目标。话语中的连贯和衔接已被广泛研究，并发现了许多连接相邻语段的现象((Hobbs, 1979; Halliday & Hasan, 1976; Grosz 等, 1995)。在实践中发现大多数有效的目标都很简单。Skip-thought (Kiros 等， 2015)和 FastSent  (Hill 等， 2016)通过对句子进行编码来预测相邻句子中的单词来学习句子嵌入。句子嵌入学习的其他目标包括预测将来的句子而不仅仅是邻居(Gan等，2017)和预测显性话语标记(Jernite等，2017;Nie 等人，2019)。我们的损失与Jernite等人(2017)的句子排序目标最为相似，他们通过学习句子嵌入来确定两个连续句子的排序。但是，与上面的大多数工作不同，我们的损失是在文本片段而不是句子中定义的。BERT (Devlin 等， 2019)使用基于预测一对句子中第二段是否与另一个文档中的一段进行了交换的损失。我们在实验中比较了这一损失，发现句子排序是一项更具挑战性的预训练任务，对某些下游任务更有用。在我们的工作同时，Wang等人(2019)也尝试预测两个连续的文本片段的顺序，但他们将其与原始的下句预测结合在一个三分类任务中，而不是从经验上比较两者。

## 3. ALBERT 的元素

在本节中，我们将为 ALBERT 提供设计决策，并与原始 BERT 架构的相应配置进行量化比较(Devlin 等，2019)。

### 3.1 模型架构选择

ALBERT 架构的主干与 BERT 类似，它使用了一个 transformer 编码器(Vaswani 等， 2017)和 GELU 非线性(Hendrycks & Gimpel, 2016)。我们遵循 BERT 符号约定,定义词嵌入大小为 $E$、编码器层的数量为 $L$,和隐藏维度大小为 $H$，遵循 Devlin 等人(2019),我们设置了前馈/过滤器大小为 $4H$,注意力头数为 $H/64$。

ALBERT对BERT的设计的改变上做出了三个主要的贡献。

**因式分解 embedding 参数** 在BERT中，以及随后的建模改进，如 XLNet (Yang 等， 2019)和 RoBERTa (Liu 等， 2019)，词片（WordPiece）嵌入大小 $E$ 与隐层维度大小$H$绑定，即 $E \equiv H$。 对于建模和实际原因，这个决策看起来都不是最优的，如下所示。

从建模的角度来看，词片嵌入意味着学习上下文无关的表征，而隐藏层嵌入意味着学习上下文相关的表征。根据上下文长度的实验表明(Liu 等， 2019)，BERT-like 表征的力量来自于使用上下文为学习这种上下文相关的表征提供信号。因此,解开词片嵌入大小$E$和隐藏层大小$H$,允许我们根据建模需要做出更高效的按需使用的总模型参数,这决定了 $H \gg E$。

从实用的角度来看，自然语言处理通常需要词汇量$V$是大的，如果$E \equiv H$，则增加 $H$ 就增加了具有大小的嵌入矩阵的大小，将具有 $V \times E$ 的大小。这很容易产生一个拥有数十亿参数的模型，其中大多数参数在训练期间只进行少量更新。

因此，对于ALBERT，我们使用嵌入参数的因式分解，将它们分解成两个更小的矩阵。我们不直接将一个 one-hot 向量投影到大小为$H$的隐藏空间中，而是先将其投影到大小为$E$的低维嵌入空间中，然后再将其投影到隐藏空间中。通过使用这种分解,我们将嵌入参数从$O(V \times H)$减少到了 $O(V\times E+E\times H)$。当$H\gg E$时，这个参数显著减少，我们选择对所有的词片使用相同的$E$,因为与整词嵌入相比他们更均匀分布在文档中,对于不同的单词来说有不同的嵌入大小(Grave 等。(2017);Baevski & Auli (2018);Dai 等.(2019))是重要的。

**跨层参数共享** 对于 ALBERT，我们提出了一种跨层参数共享的方法来提高参数效率。共享参数的方法有多种，例如只跨层共享前馈网络(FFN)参数，或只共享注意力参数。ALBERT的默认决策是跨层共享所有参数。除非另有说明，我们所有的实验都使用这个默认决策。

Dehghani 等 (2018) (Universal Transformer, UT)和 Bai 等. (2019) (Deep Equilibrium Models, DQE)对 Transformer 网络也探索了类似的策略。与我们的观察不同，Dehghani等人(2018)表明UT性能优于香草 Transformer。Bai等(2019)的研究表明，他们的DQEs达到了一个平衡点，在这个平衡点上，某一层的输入和输出嵌入保持不变。我们对 L2 距离和余弦相似度的测量表明，我们的嵌入是振荡的，而不是收敛的。

![图2](/assets/images/nlp/albert/fig2.png)

图2： BERT-large 和 ALBERT-large 各层嵌入的输入和输出的L2距离和余弦相似度(以度表示)。

图2显示了使用 BERT-large 和 ALBERT-large 配置的每一层的输入和输出嵌入的L2距离和余弦相似度(见表2)。结果表明，权值共享对网络参数的稳定有一定的影响。尽管与BERT相比，这两个指标都有下降，但即使在24层之后，它们也不会收敛到0。这说明ALBERT参数的解空间与DQE的解空间有很大的不同。

**句间连贯性损失** 除了遮蔽语言建模(MLM)损失(Devlin 等， 2019)之外，BERT还使用了一个称为下句预测(NSP)的额外损失。NSP是预测两个片段在原文中是否连续出现的二分类损失，具体如下:从训练语料库中提取连续片段，生成正例;负例是由来自不同文档的片段配对产生的;正、负样本的抽样概率相等。NSP的目标是为了提高下游任务的性能，比如自然语言推理，这需要对句子对之间的关系进行推理。然而，后续研究(Yang 等，2019;Liu等人2019)发现NSP的影响不可靠，并决定消除它，这一决定得到了多个任务的下游任务性能改进的支持。

我们推测，NSP之所以无效的主要原因是它作为一项任务缺乏难度，与MLM相比。NSP将主题预测和相干性预测合并在一起作为单一的任务。然而，与一致性预测相比，主题预测更容易学习，并且与使用MLM损失所学的内容有更多的重叠。

我们认为句间建模是语言理解的一个重要方面，但我们提出了一个主要基于连贯的损失。也就是说，对于ALBERT来说，我们使用了一个句子顺序预测损失(SOP)，它避免了主题预测，而是专注于对句子间连贯性进行建模。SOP损失使用与BERT相同的技术作为正例(同一文档中的两个连续片段)，而作为负例使用相同的两个连续段，但顺序互换。这就迫使模型学习关于话语级连贯性属性的更细粒度的区别。如第4.6节所示，NSP根本无法解决SOP任务(即，它最终学习更容易的主题预测信号，并在SOP任务上执行随机基线水平)，而SOP可以在分析未对齐的相干线索的基础上，预先将NSP任务解决到一个合理的程度。因此，ALBERT模型不断地改进多句编码任务的下游任务性能。

### 3.2 模型设置

表2给出了 BERT 和 ALBERT 模型在超参数设置上的差异。由于以上讨论的设计选择，ALBERT 模型的参数尺寸比相应的 BERT 模型要小得多。

![表2](/assets/images/nlp/albert/tab2.png)

表2：本文对主要的BERT和ALBERT模型的结构进行分析的配置。

例如，与 BERT-large 相比，ALBERT-large少了大约18倍的参数，ALBERT-large 是18M，BERT-large是334M。如果我们将 BERT 设置为超大规模，$H=2048$,我们最终得到一个拥有12.7亿的模型参数和低性能(图1)。相比之下,一个 ALBERT-xlarge 配置$H=2048$只有60M参数,而一个 ALBERT-xxlarge 配置$H =4096$有 233M 参数,即约70%的 BERT-large 参数。注意，对于 ALBERT-xxlarge，我们主要在12层网络上报告结果，因为24层网络(具有相同的配置)可以获得类似的结果，但在计算上更昂贵。

这种参数效率的提高是ALBERT设计选择的最重要的优势。在我们能够量化这一优势之前，我们需要更详细地介绍我们的实验设置。

## 4 实验结果
### 4.1 实验设置

为了使比较尽可能有意义，我们遵循 BERT (Devlin 等，2019)设置，使用 BookCorpus 语料库(Zhu 等，2015)和英语维基百科(Devlin 等，2019)进行预训练基线模型。这两个语料库由大约16GB的未压缩文本组成。我们的格式化我们的输入为“[CLS] $x_1$[SEP]$x_2$[SEP]”，其中$x_1=x_{1,1},x_{1,2} \cdots$,$x_2=x_{2,1},x_{2,2}\cdots$是两段。我们总是限制最大输入长度为512，以10%的概率随机产生输入序列小于512的句子。和BERT一样，我们使用了30000个词汇量，用句片来标记(Kudo & Richardson, 2018)，比如XLNet (Yang 等， 2019)。

我们使用 n-gram 遮蔽(Joshi 等，2019)为MLM目标生成遮蔽输入，随机选择每个n-gram掩码的长度。长度为$n$的概率由下式给出：

$$
p(n)=\frac{1/n}{\sum_{k=1}^N 1/k}
$$

我们设置了n-gram的最大长度(即，n)为3(即，MLM目标可以由一个3-gram完整的单词，比如“White House correspondents”)。

所有的模型更新都使用批大小为4096和学习率为0.00176的LAMB优化器(You 等，2019)。除非另有说明，我们对所有型号进行125,000步的训练。训练是在 Cloud TPU V3上完成的。根据模型的大小，用于训练的TPUs的数量从64到1024不等。

本节中描述的实验设置用于我们自己的 BERT 和 ALBERT 模型的所有版本，除非另有说明。

### 4.2 评价基准

#### 4.2.1 内在(固有)评估

为了监控训练进度，我们使用与第4.1节相同的步骤，创建了一个基于来自 SQuAD 和 RACE 的开发集的开发集。我们报告准确的MLM和句子分类任务。注意，我们只使用这个集合来检查模型是如何收敛的;它的使用方式不会影响任何下游评估的性能，例如通过模型选择。

#### 4.2.2 下游评价

在Yang等人(2019)和Liu等人(2019)之后，我们在三个流行的基准上评估我们的模型:通用语言理解评估(GLUE)基准(Wang等人，2018)、两个版本的斯坦福问答数据集(SQuAD;Rajpurkar等人，2016;2018)以及来自阅读理解考试(RACE)的数据集(Lai 等，2017)。为了完整性，我们在附录A.1中提供了对这些基准的描述。与(Liu等人，2019年)一样，我们在开发集上执行早期停止，除了基于任务排行榜的最终比较之外，我们报告所有的比较，我们还报告测试集结果。

### 4.3 BERT和ALBERT的总体比较

我们现在准备量化第3节中描述的设计选择的影响，特别是关于参数效率的选择。参数性能的改善表明 ALBERT 设计选择的最重要的优势,如表3所示:只有 BERT-large 大约70%的参数,ALBERT-xxlarge 就达到了更显著的改善,以验证集分数区分几个代表性的下游任务：SQuAD v1.1 (+1.9%),SQuAD v2.0 (+3.1%), MNLI (+1.4%), SST-2 (+2.2%), 和 RACE (+8.4%).

我们还观察到，在所有指标上，BERT-xlarge 得到的结果明显比 BERT-base 差。这表明，像 BERT-xlarge 这样的模型比那些参数大小较小的模型更难训练。另一个有趣的发现是，在相同的训练配置(相同数量的TPUs)下，训练时的数据吞吐量的速度。由于 ALBERT 模型通信少、计算量小，与对应的 BERT 模型相比，ALBERT 模型具有更高的数据吞吐量。最慢的是BERT-xlarge模型，我们使用它作为基线。随着模型的增大，BERT和ALBERT模型之间的差异也越来越大，例如 ALBERT-xlarge 训练速度是 BERT-xlarge 的2.4倍。

![表3](/assets/images/nlp/albert/tab3.png)

表3：开发人员通过 BOOKCORPUS 和 Wikipedia 为模型设定了125k步的预训练结果。在这里和其他地方，Avg列的计算方法是对其左侧的下游任务的得分进行平均(每个 SQuAD 的 F1 和 EM 两个数字首先进行平均)。

接下来，我们进行消融实验（Ablation experiment），量化每个设计选择对ALBERT的个人贡献。

### 4.4 因式分解嵌入参数

表4显示了使用 ALBERT-base 配置设置更改词汇表嵌入大小$E$的效果(参见表2)，使用的是同一组具有代表性的下游任务。在非共享条件下( BERT 风格)，更大的嵌入大小提供了更好的性能，但不是很多。在全共享条件下(ALBERT 风格)，大小为128的嵌入似乎是最好的。基于这些结果，我们在以后的所有设置中使用嵌入大小$E = 128$，作为进一步扩展的必要步骤。

![表4](/assets/images/nlp/albert/tab4.png)

表4:词汇表嵌入大小对ALBERT-base性能的影响。

### 4.5 跨层参数共享

表5展示了各种跨层参数共享策略的实验，使用ALBERT-base配置(表2)，有两个嵌入大小($E = 768$和$E = 128$)。我们比较了全共享策略(ALBERT风格)、非共享策略(BERT风格)和中间策略，其中只有注意力参数是共享的(但不包括FNN参数)或只有FFN参数是共享的(但不包括注意力参数)。

![表5](/assets/images/nlp/albert/tab5.png)

表5:跨层参数共享策略的效果，基于 ALBERT 的配置。

在两种情况下，全共享策略都会影响性能，但对于$E = 128$ (Avg上为-1.5)和$E = 768$ (Avg上为-2.5)来说，这种影响没有那么严重。此外，性能下降主要来自共享 FFN 层参数，而共享注意力参数在$E = 128$ (Avg +0.1)时没有下降，在$E = 768$ (Avg -0.7)时略有下降。

还有其他跨层共享参数的策略。例如，我们可以将L层划分为$N$个大小为$M$的组，每个大小为$M$的组共享参数。总的来说，我们的实验结果表明，群体规模$M$越小，我们的表现越好。然而，减小组大小$M$也会显著增加总体参数的数量。我们选择共享策略作为默认选择。

### 4.6 句子顺序预测(SOP)

我们使用 ALBERT-base 配置比较了三个额外的句子间损失的实验条件:none (XLNet 和 RoBERTa 风格)、NSP (BERT 风格)和SOP (ALBERT 风格)。结果如表6所示，包括内在的(MLM、NSP和SOP任务的准确性)和下游任务。

![表6](/assets/images/nlp/albert/tab6.png)

表6:句子预测损失、NSP和SOP对内在(intrinsic)和下游任务的影响。

内在任务的结果表明，NSP损失对SOP任务没有区分能力(正确率为52.0%，与“无”条件下的随机猜测性能相近)。这使我们可以得出结论，NSP最终只建模主题转移。相比之下，SOP损失较好地解决了NSP任务(78.9%的准确率)，SOP损失更佳(86.5%的准确率)。更重要的是，对于多句编码任务(对于SQuAD1.1，大约+1%;SQuAD2.0，大约+2%;RACE大约+1.7%)，对于平均成绩提高大约+1%，SOP损失似乎持续改善了下游任务性能。

### 4.7 网络深度和宽度的影响

在本节中，我们将检查深度(层数)和宽度(隐藏大小)如何影响ALBERT的性能。表7显示了使用不同层数的ALBERT-large配置的性能(参见表2)。具有3层或3层以上的网络使用之前的深度参数进行微调(例如，12层网络参数从6层网络参数的检查点进行微调)来训练，类似的技术已经在Gong等人(2019)中使用。如果我们将三层ALBERT模型与一层ALBERT模型进行比较，尽管它们具有相同数量的参数，但是性能显著提高。但是，当继续增加层数时，会产生递减收益:12层网络的结果与24层网络的结果比较接近，而48层网络的性能出现下降。

![表7](/assets/images/nlp/albert/tab7.png)

表7：增加ALBERT-large层数的效果。

类似的现象(这次是宽度)可以在表8中看到3层ALBERT-large配置。当我们增加隐藏的大小时，我们得到了性能的提高，但收益却在减少。在隐藏大小为6144时，性能似乎明显下降。我们注意到，这些模型似乎都没有对训练数据进行过度拟合，而且与性能最佳的ALBERT配置相比，它们都有更高的训练和验证损失。

![表8](/assets/images/nlp/albert/tab8.png)

表8：增加ALBERT-large 3层的隐藏层size的效果。

### 4.8 如果我们训练的时间一样会如何？

表3中的加速结果表明，与ALBERT-xxlarge相比，BERT-large的数据吞吐量大约高出3.17倍。由于较长时间的训练通常会带来更好的性能，所以我们进行了一个比较，在这个比较中，我们不是控制数据吞吐量(训练步骤的数量)，而是控制实际的训练时间(即训练时间)。让模型训练相同的时间)。在表9中，我们比较了一个BERT-large模型在400k训练步骤之后(训练34h之后)的性能，大致相当于用125k训练步骤(训练32h)训练一个ALBERT-xxlarge模型所需的时间。

![表9](/assets/images/nlp/albert/tab9.png)

表9：控制训练时间的效果，BERT-large vs ALBERT-xxlarge 配置。

经过大致相同时间的训练后，ALBERT-xxlarge明显优于BERT-large:平均成绩提高1.5%，RACE差异高达+5.2%。

### 4.9 非常宽的 ALBERT 模型也需要深吗?

在4.7节中，我们展示了对于ALBERT-large (H=1024)， 12层配置和24层配置之间的差异很小。这个结果是否仍然适用于更广泛的ALBERT配置，比如ALBERT-xxlarge (H=4096)?

![表10](/assets/images/nlp/albert/tab10.png)

表10：使用 ALBERT-xxlarge 配置的深层网络的效果。

答案由表10的结果给出。在下游精度方面，12层和24层ALBERT-xxlarge配置之间的差异可以忽略不计，而Avg分数是相同的。我们得出结论，当共享所有跨层参数(ALBERT风格)时，不需要比12层配置更深的模型。

### 4.10 额外的训练数据和 Dropout 的影响

到目前为止，所做的实验仅使用了Wikipedia和BOOKCORPUS数据集，如(Devlin 等， 2019)。在本节中，我们报告XLNet (Yang 等， 2019)和RoBERTa (Liu 等， 2019)使用的附加数据的影响的测量结果。

![图3](/assets/images/nlp/albert/fig3.png)

图3：在训练中添加数据和删除dropout的效果。

图3a描绘了两种情况下验证集MLM的准确性，没有和有额外的数据，后一种情况提供了一个显著的提升。我们还观察到表11中下游任务的性能改进，除了SQuAD基准测试(基于维基百科，因此受到领域外训练材料的负面影响)。

![表11](/assets/images/nlp/albert/tab11.png)

表11：使用ALBERT-base配置的附加训练数据的效果。

我们还注意到，即使在训练了100万步之后，我们最大的模型仍然没有对训练数据进行过度拟合。因此，我们决定删除dropout，以进一步增加我们的模型容量。从图3b的图中可以看出，去掉dropout可以显著提高MLM的精度。在大约1M的训练步骤中，对ALBERT-xxlarge的中间评估(表12)也证实了删除dropout有助于下游任务。有实证(Szegedy 等， 2017)和理论(Li 等， 2019)的证据表明，卷积神经网络中的批处理归一化和dropout可能会产生有害的结果。据我们所知，我们是第一个发现dropout会影响大型基于Transformer的模型的性能的。然而，ALBERT的底层网络结构是Transformer的一个特例，需要进一步的实验来观察这种现象是否出现在其他基于Transformer的架构中。

![表12](/assets/images/nlp/albert/tab12.png)

表12：去除dropout的效果，测量为 ALBERT-xxlarge 配置。

### 4.11 NLU任务的最新技术

本节报告的结果利用了Devlin等人(2019)使用的训练数据，以及Liu等人(2019)和Yang等人(2019)使用的附加数据。我们报告了在两种情况下进行微调的最新结果:单模式和集成。在这两种设置中，我们只进行单任务微调。在Liu等人(2019)之后，在验证集上，我们报告了5次运行的中间结果。

![表13](/assets/images/nlp/albert/tab13.png)

表13：GLUE基准测试的最新结果。对于单任务单模型结果，我们报告ALBERT在1M步(与RoBERTa相当)和在1.5M步。ALBERT套装使用的是经过1M、1.5M和其他步数训练的模型。

![表14](/assets/images/nlp/albert/tab14.png)

表14：SQuAD和RACE benchmarks 上的最优结果

单模型ALBERT配置合并了讨论过的最佳性能设置:ALBERT-xxlarge(表2)，使用了MLM和SOP损失，没有dropout。根据验证集的性能选择有助于最终集成模型的检查点;根据任务的不同，这个选择所考虑的检查点的数量从6个到17个不等。对于GLUE(表13)和RACE(表14)基准测试，我们对集成模型的模型预测进行平均，其中候选测试通过使用12层和24层架构的不同训练步骤进行微调。对于SQuAD(表14)，我们对具有多个概率的跨度的预测分数进行平均;我们还对“无法回答的”决策的得分进行平均。

单模型和整体测试结果都表明，ALBERT在所有三个基准测试中都显著提高了最先进的信号水平，得到了89.4的GLUE分数、92.2的SQuAD2.0 F1测试分数和89.4的RACE测试精度。后者似乎特别强烈的改善,一个跳跃+17.4%的绝对点BERT(Devlin 等, 2019), +7.6% XLNet(Yang 等, 2019), +6.2% RoBERTa(Liu 等, 2019),和5.3% DCMI+ (Zhang 等, 2019),多个模型的一个专门为阅读理解任务。我们的单一模型的准确率达到了86.5%，仍然比最先进的集成模型的准确率高了2.4%。

## 5. 讨论

虽然ALBERT-xxlarge的参数比BERT-large少，并且得到了更好的结果，但是由于它的结构更大，计算上更昂贵。下一步的重点是通过稀疏注意(Child 等， 2019)和块注意(Shen 等， 2018)等方法加快ALBERT的训练和推理速度。可以提供额外表示能力的正交研究线（orthogonal line of research）包括硬示例挖掘(hard example mining)(Mikolov 等， 2013)和更有效的语言建模训练(Yang 等， 2019)。此外,尽管我们有令人信服的证据,句子顺序预测始终如一地有用来学习任务导致更好的语言表征,我们假设可能有更多的维度没有被当前自监督训练损失捕获，这些维度可能对结果表征创建出额外的表征能力。

## 鸣谢

作者感谢Beer Changpinyo、Nan Ding、Noam Shazeer和Tomer Levinboim对项目的讨论和提供有用的反馈;Omer Levy和Naman Goyal为RoBERTa澄清实验设置;Zihang Dai澄清XLNet;Brandon Norick, Emma Strubell, Shaojie Bai，和Sachin Mehta为论文提供了有用的反馈;Liang Xu, Chenjie Cao和CLUE community提供的ALBERT模型中文版的训练数据和评估benechmark。

## 参考文献

* Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. arXiv preprint arXiv:1809.10853, 2018.
* Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Neural Information Processing Systems (NeurIPS), 2019.
* Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. In Proceedings of the second PASCAL challenges workshop on recognising textual entailment, volume 6, pp. 6–4. Venice, 2006.
* Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth PASCAL recognizing textual entailment challenge. In TAC, 2009.
* Daniel Cer, Mona Diab, Eneko Agirre, In ̃igo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 1–14, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/S17-2001. URL https://www.aclweb.org/anthology/S17-2001.
* Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.
* Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
* Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges Workshop, pp. 177–190. Springer, 2005.
* Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. In Advances in neural infor- mation processing systems, pp. 3079–3087, 2015.
* Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.
* Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https: //www.aclweb.org/anthology/N19-1423.
* William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005. URL https://www.aclweb.org/anthology/I05-5002.
* Zhe Gan, Yunchen Pu, Ricardo Henao, Chunyuan Li, Xiaodong He, and Lawrence Carin. Learn- ing generic sentence representations using convolutional neural networks. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2390–2400, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1254. URL https://www.aclweb.org/anthology/D17-1254.
* Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL Workshop on Textual Entail- ment and Paraphrasing, pp. 1–9, Prague, June 2007. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/W07-1401.
* Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual net- work: Backpropagation without storing activations. In Advances in neural information processing systems, pp. 2214–2224, 2017.
* Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of bert by progressively stacking. In International Conference on Machine Learning, pp. 2337–2346, 2019.
* Edouard Grave, Armand Joulin, Moustapha Cisse ́, Herve ́ Je ́gou, et al. Efficient softmax approxima- tion for gpus. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1302–1310. JMLR. org, 2017.
* Barbara J. Grosz, Aravind K. Joshi, and Scott Weinstein. Centering: A framework for modeling the local coherence of discourse. Computational Linguistics, 21(2):203–225, 1995. URL https: //www.aclweb.org/anthology/J95-2003.
* M.A.K. Halliday and Ruqaiya Hasan. Cohesion in English. Routledge, 1976.
* Jie Hao, Xing Wang, Baosong Yang, Longyue Wang, Jinfeng Zhang, and Zhaopeng Tu. Modeling recurrence for transformer. Proceedings of the 2019 Conference of the North, 2019. doi: 10. 18653/v1/n19-1122. URL http://dx.doi.org/10.18653/v1/n19-1122.
* Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415, 2016.
* Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1367–1377. Association for Computational Linguistics, 2016. doi: 10.18653/v1/N16-1162. URL http: //aclweb.org/anthology/N16-1162.
* Jerry R. Hobbs. Coherence and coreference. Cognitive Science, 3(1):67–90, 1979.
* Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.
* Shankar Iyer, Nikhil Dandekar, and Kornl Csernai. First quora dataset release: Ques- tion pairs, January 2017. URL https://www.quora.com/q/quoradata/ First-Quora-Dataset-Release-Question-Pairs.
* Yacine Jernite, Samuel R Bowman, and David Sontag. Discourse-based objectives for fast unsuper- vised sentence representation learning. arXiv preprint arXiv:1705.00557, 2017.
* Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. SpanBERT: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529, 2019.
* Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Ur- tasun, and Sanja Fidler. Skip-thought vectors. In Proceedings of the 28th International Con- ference on Neural Information Processing Systems - Volume 2, NIPS’15, pp. 3294–3302, Cam- bridge, MA, USA, 2015. MIT Press. URL http://dl.acm.org/citation.cfm?id= 2969442.2969607.
* Taku Kudo and John Richardson. SentencePiece: A simple and language independent sub- word tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Con- ference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URL https://www.aclweb.org/anthology/D18-2012.
* Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://www. aclweb.org/anthology/D17-1082.
* Quoc Le and Tomas Mikolov. Distributed representations of sentences and documents. In Proceed- ings of the 31st ICML, Beijing, China, 2014.
* Hector Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In Thir- teenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012.
* Xiang Li, Shuo Chen, Xiaolin Hu, and Jian Yang. Understanding the disharmony between dropout and batch normalization by variance shift. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2682–2690, 2019.
* Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pre- training approach. arXiv preprint arXiv:1907.11692, 2019.
* Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 6294–6305. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/ 7209-learned-in-translation-contextualized-word-vectors.pdf.
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed represen- tations of words and phrases and their compositionality. In Advances in neural information pro- cessing systems, pp. 3111–3119, 2013.
* Allen Nie, Erin Bennett, and Noah Goodman. DisSent: Learning sentence representations from ex- plicit discourse relations. In Proceedings of the 57th Annual Meeting of the Association for Com- putational Linguistics, pp. 4497–4510, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1442. URL https://www.aclweb.org/anthology/ P19-1442.
* Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word rep- resentation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532–1543, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1162. URL https://www.aclweb.org/anthology/ D14-1162.
* Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Con- ference of the North American Chapter of the Association for Computational Linguistics: Hu- man Language Technologies, Volume 1 (Long Papers), pp. 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1202. URL https://www.aclweb.org/anthology/N18-1202.
* Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. https://s3-us-west-2.amazonaws.com/ openai-assets/research-covers/language-unsupervised/language_ understanding_paper.pdf, 2018.
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 2019.
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URL https://www.aclweb. org/anthology/D16-1264.
* Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 784–789, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124. URL https://www.aclweb. org/anthology/P18-2124.
* Tao Shen, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. Bi-directional block self- attention for fast and memory-efficient sequence modeling. arXiv preprint arXiv:1804.00857, 2018.
* Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model par- allelism, 2019.
* Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642, Seattle, Washington, USA, October 2013. Association for Computa- tional Linguistics. URL https://www.aclweb.org/anthology/D13-1170.
* Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for BERT model compression. arXiv preprint arXiv:1908.09355, 2019.
* Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-First AAAI Confer- ence on Artificial Intelligence, 2017.
* Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: The impact of student initialization on knowledge distillation. arXiv preprint arXiv:1908.08962, 2019.
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998–6008, 2017.
* Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceed- ings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 353–355, Brussels, Belgium, November 2018. Association for Computational Lin- guistics. doi: 10.18653/v1/W18-5446. URL https://www.aclweb.org/anthology/ W18-5446.
* Wei Wang, Bin Bi, Ming Yan, Chen Wu, Zuyi Bao, Liwei Peng, and Luo Si. StructBERT: Incor- porating language structures into pre-training for deep language understanding. arXiv preprint arXiv:1908.04577, 2019.
* Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018.
* Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sen- tence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technolo- gies, Volume 1 (Long Papers), pp. 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1101. URL https://www.aclweb. org/anthology/N18-1101.
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019.
* Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. Reducing BERT pre-training time from 3 days to 76 minutes. arXiv preprint arXiv:1904.00962, 2019.
* Shuailiang Zhang, Hai Zhao, Yuwei Wu, Zhuosheng Zhang, Xi Zhou, and Xiang Zhou. DCMN+: Dual co-matching network for multi-choice reading comprehension. arXiv preprint arXiv:1908.11511, 2019.
* Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp. 19–27, 2015.

## A. 附录

### A.1 下游评估任务

**GLUE**由9个任务组成，即语言可接受性语料库(Corpus of Linguistic Acceptability,CoLA;Warstadt 等，2018)，斯坦福情感树库(SST;Socher 等，2013)，微软研究释义语料库(MRPC;Dolan&Brockett，2005)，语义文本相似标准(STS;Cer 等，2017），Quora问题对(QQP;Iyer等人，2017)，Multi-GenreNLI (MNLI;Williams等，2018)，Question NLI (QNLI;Rajpurkar等人，2016)，识别文本蕴涵(RTE;Dagan等，2005;Bar-Haim等人，2006;Giampiccolo等，2007;Bentivogli 等， 2009)和Winograd NLI (WNLI;Levesque等，2012)。它侧重于评估自然语言理解的模型能力。在报告MNLI结果时，我们只报告“匹配”（match）条件(MNLI-m)。我们遵循之前工作中的微调程序(Devlin等人，2019年;Liu等，2019年;Yang 等，2019)，并报告从GLUE提交获得的剩余测试集性能。对于提交的测试集，我们按照Liu等人(2019)和Yang等人(2019)的描述，对WNLI和QNLI执行特定于任务的修改。

**SQuAD**是一个从维基百科中提取问题答案的数据集。答案是来自上下文段落的片段，任务是预测答案的范围。我们在两个版本的SQuAD上评估我们的模型:v1.1和v2.0。SQuAD v1.1拥有100,000个带人类注释的问题/答案对。SQuAD v2.0版增加了5万个无法回答的问题。对于SQuAD v1.1，我们使用与BERT相同的训练过程，而对于SQuAD v2.0，我们联合使用一个span extraction loss和一个额外的分类器来预测答案来训练模型(Yang 等，2019;Liu 等，2019)。我们报告了验证集和测试集的性能。

**RACE**是一个大型的多选题阅读理解数据库，收集了中国英语考试中近10万个问题。RACE中的每个实例有4个候选答案。后续工作(Yang 等， 2019；Liu 等， 2019)，我们使用文章、问题和每个候选答案的连接作为模型的输入。然后，我们使用来自“[CLS]”令牌的表征来预测每个答案的概率。数据集由两个域组成:初中和高中。我们在这两个领域训练我们的模型，并报告验证集和测试集的准确性。

### A.2 超参数

下游任务的超参数如表15所示。我们采用了Liu等人(2019)、Devlin等人(2019)和Yang等人(2019)的超参数。

![表15](/assets/images/nlp/albert/tab15.png)

表15：下游任务中的超参数。LR:学习率。BSZ:批量大小。DR:dropout 率。TS:训练步数。WS:Warmup步数。MSL：最大序列长度。

---
**参考**：
1. 论文：Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)（2019.10.23）
