

# 支持向量机（SVM）

## 简介

支持向量机（Support Vector Machine, SVM）本身是一个**二元分类算法**，是感知器算法模型的一种扩展，现在的SVM算法支持**线性分类**和**非线性分类**的分类应用，并且也能够直接将SVM应用于**回归应用**中，同时通过OvR或者OvO的方式 我们也可以将SVM应用在**多元分类**领域中。在不考虑集成学习算法，不考虑特定 的数据集的时候，在分类算法中SVM可以说是特别优秀的。<img src="/Users/xugaoliang/Documents/xu-blog/assets/images/机器学习/svm/感知机.png" alt="感知机" style="zoom:25%;" />

在感知器模型中，算法是在数据中找出一个划分超平面，让尽可能多的数据分布在这个平面的两侧，从而达到分类的效果，但是在实际数据中这个符合我们要求的超平面可能有无穷多个。![感知机](/Users/xugaoliang/Documents/xu-blog/docs/机器学习/aaa/感知机.png)

![感知机](/assets/images/机器学习/svm/感知机.png)
<center>感知机</center>

但是实际上离超平面足够远的点基本上都是被正确分类的，所以这个是没有意义的;反而更需要关心那些离超平面很近的点，这些点比较容易分错。所以说我们只要**让离超平面比较近的点尽可能的远离这个超平面**，那么我们的模型分类效果应该就会比较不错了。SVM其实就是这个思想。


支持向量机学习方法由简到繁的模型可分为：
* **线性可分支持向量机**（硬间隔支持向量机）
* **线性支持向量机**（近似线性可分，软间隔支持向量机）
* **非线性支持向量机**（核技巧）

对于分类问题，我们有
* **输入空间**：欧氏空间或离散集合
* **特征空间**：欧氏空间或希尔伯特空间

对于**线性可分支持向量机**和**线性支持向量机**，输入空间和特征空间元素一一对应，将输入空间的输入映射到特征空间，**非线性支持向量机**通过非线性映射，将输入映射到特征空间。支持向量机的学习是在特征空间进行的。


## 一些概念和定义

### 数据集
$$
T = \{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}
$$

其中 $x_i \in \mathcal{X}=\mathrm{R}^n,y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$,$y_i$为$x_i$的类标记，当$y_i=+1$时，$x_i$为正例，当$y_i=-1$时，$x_i$为负例，$(x_i,y_i)$为样本点。

### 分割超平面

支持向量机的学习目标就是在特征空间中找到一个分离超平面，能将实例分到不同的类，分离超平面对应方程$w\cdot x+b=0$,由法向量$w$和截距$b$决定，可用$(w,b)$来表示。分离超平面将特征空间划分为两部分，一部分是正类，一部分是负类。法向量指向的一侧为正类，另一侧为负类。

### 线性可分支持向量机

定义：给定线性可分训练数据集，通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面为

$$
w^* \cdot x + b^*=0 
$$
以及相应的分类决策函数

$$
f(x)=sign(w^*\cdot x+b^*)
$$
称为线性可分支持向量机。

### 函数间隔

一般来说，一个点距离分割超平面的远近可以表示分类预测的确信程度。在超平面 $w\cdot x+b=0$确定的情况下，$|w\cdot x+b|$能够相对地表示点$x$距离超平面的远近。而$w\cdot x+b$的符号与类标记$y$的符号是否一致能表示分类是否正确。所有可以用$y(w\cdot x+b)$来表示分类的正确性及确信度，这就是函数间隔(functional margin)的概念

定义（函数间隔）：对于给定的训练数据集$T$和超平面$(w,b)$，定义超平面(w,b)关于样本点$(x_i,y_i)$的函数间隔为

$$
\hat{\gamma}_i = y_i(w\cdot x_i+b)
$$

定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔的最小值，即

$$
\hat{\gamma} = \mathop{\min}_{i=1,\cdots,N} \hat{\gamma}_i
$$

函数间隔可以表示分类预测的正确性及确信度。但选择分离超平面时，只有函数间隔还不够，因为只要成比例缩放$w$和$b$,例如将他们改为$2w$和$2b$,超平面并没有改变，但函数间隔却成为原来的2倍。这一事实启示我们，可以对分离超平面的法向量$w$加上某些约束，如规范化，$||w||=1$,使得间隔是确定的，这时函数间隔成为几何间隔(geometric margin)

### 几何间隔

定义（几何间隔）：对于给定的训练数据集$T$和超平面$(w,b)$，定义超平面(w,b)关于样本点$(x_i,y_i)$的几何间隔为

$$
\gamma_i = y_i(\frac{w}{||w||} \cdot x_i+\frac{b}{||w||})
$$

其中$||w||$为$w$的$L_2$范数。

定义超平面$(w,b)$关于训练数据集$T$的几何间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的几何间隔的最小值，即

$$
\gamma = \mathop{\min}_{i=1,\cdots,N} \gamma_i
$$

几何间隔其实就是实例点到超平面的带符号的距离（signed distance），当样本点被超平面正确分类时就是实例点到超屏幕的距离。

函数间隔与几何间隔的关系是：

$$
\gamma_i = \frac{\hat{\gamma}_i}{||w||}
$$

$$
\gamma = \frac{\hat{\gamma}}{||w||}
$$

如果$||w||=1$，那么函数间隔和几何间隔相等。超平面参数$w$和$b$成比例地改变（超平面没有改变），函数间隔也按此比例改变，而几何间隔不变。

### 凸优化问题

凸优化问题是指约束最优化问题

$$
\begin{aligned}
    
\mathop{\max}_{w} \quad&f(w) \\\\
s.t.\quad &g_i(w) \le 0,\quad i=1,2,\cdots,k \\\\
&h_i(w)=0,\quad i=1,2,\cdots,l

\end{aligned}
$$

其中，目标函数$f(w)$和约束函数$g_i(w)$都是$\mathrm{R}^n$上的连续可微凸函数，约束函数$h_i(w)$是$\mathrm{R}^n$上的仿射函数。

当目标函数$f(w)$是二次函数且约束函数$g_i(w)$是仿射函数时，上述凸最优化问题成为凸二次规划问题。

### 间隔最大化

支持向量机的目的就是找到让几何间隔最大的分离超平面，不仅将正负实例点分开，而且对最难分的点，也让他们有足够大的确信度将其分开，以便具有更好的泛化能力。

![间隔最大化](/assets/images/机器学习/svm/间隔最大化.png)
<center>间隔最大化</center>

这个问题可以表示为约束最优化问题：

$$
\begin{aligned}
    
&\mathop{\max}_{w,b} \gamma \\\\
&s.t.\quad y_i\left(\frac{w}{||w||}\cdot x_i + \frac{b}{||w||}\right) \ge \gamma, i=1,2,\cdots,N

\end{aligned}
$$
约束条件表示的是超平面$(w,b)$关于每个训练样本点的几何间隔至少是$\gamma$

考虑到几何间隔和函数间隔的关系，问题可改写为

$$
\begin{aligned}
    
&\mathop{\max}_{w,b} \frac{\hat{\gamma}}{||w||} \\\\
&s.t.\quad y_i(w\cdot x_i + b) \ge \hat{\gamma}, i=1,2,\cdots,N

\end{aligned}
$$
因为函数间隔可通过同比例缩放$w$和$b$而改变，且对最优化问题的不等式约束没有影响，对目标函数的优化也没有影响，所以，可取$\hat{\gamma}=1$，又因为最大化$\frac{1}{||w||}$和最小化$\frac{1}{2}{||w||}^2$等价（这样变换的目标是为了后面对目标函数求导方便），最终得到线性可分支持向量机学习的最优化问题为

$$
\begin{aligned}
    
&\mathop{\min}_{w,b} \frac{1}{2}{||w||}^2 \\\\
&s.t.\quad y_i(w\cdot x_i + b) -1\ge 0, \quad i=1,2,\cdots,N 

\end{aligned}
$$

这是一个凸二次规划问题。

线性可分训练数据集的最大间隔分类超平面是存在且唯一的，且最优解$(w^*,b^*)$必满足$w^* \ne 0$。证明详见李航《统计学习方法》100-101页

### 支持向量和间隔边界

线性可分情况下，训练数据集的样本点中与分离超平面距离最近的样本点的实例称为支持向量，支持向量是使约束条件成立的点，即

$$
y_i(w\cdot x+b)-1=0
$$

对于$y_i=+1$的正例点，支持向量在超平面

$$
H_1:w\cdot x+b=1
$$
上，对$y_i=-1$的负例点，支持向量在超平面

$$
H_2:w\cdot x+b=-1
$$
上。如图，在$H_1$和$H_2$上的点就是支持向量


![支持向量](/assets/images/机器学习/svm/支持向量.png)
<center>支持向量（李航老师的图）</center>


![支持向量与间隔](/assets/images/机器学习/svm/支持向量与间隔.png)
<center>支持向量与间隔（周志华老师的图）</center>

$H_1$与$H_2$之间的距离称为间隔（margin），$H_1$与$H_2$称为间隔边界。决定分离超平面时只有支持向量起作用，所以这种分类模型称为支持向量机。

## 线性可分支持向量机

对于约束问题
$$
\begin{aligned}
    
&\mathop{\min}_{w,b} \frac{1}{2}{||w||}^2 \\\\
&s.t.\quad y_i(w\cdot x_i + b) -1\ge 0, \quad i=1,2,\cdots,N 

\end{aligned}
$$

即

$$
\tag{1}
\begin{aligned}
    
&\mathop{\min}_{w,b} \frac{1}{2}{||w||}^2 \\\\
&s.t.\quad 1-y_i(w\cdot x_i + b)\le 0, \quad i=1,2,\cdots,N 

\end{aligned}
$$

通过引入拉格朗日乘子$\alpha_i \ge 0,i=1,2,\cdots,N$,构建出拉格朗日函数：

$$
L(w,b,\alpha) = \frac{1}{2}{||w||}^2+\sum_{i=1}^N \alpha_i (1-y_i(w\cdot x_i+b))
$$

其中，$\alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^\top$为拉格朗日乘子向量。

而原始问题(1)等价于拉格朗日函数的极小极大问题：

$$
\mathop{\min}_{w,b} \mathop{\max}_\alpha L(w,b,\alpha)
$$
当满足一定条件时，拉格朗日函数的极小极大问题的解等于拉格朗日函数的极大极小问题的解：

$$
\mathop{\max}_\alpha \mathop{\min}_{w,b} L(w,b,\alpha)
$$

拉格朗日函数的极大极小问题表示为约束问题的话，即：

$$
\begin{aligned}
    
&\mathop{\max}_{\alpha} \theta_D(\alpha) \\\\
&s.t.\quad \alpha_i \ge 0, i=1,2,\cdots,N

\end{aligned}
$$
称为原问题的对偶问题，其中$\theta_D(\alpha) =\mathop{\min}_{w,b} L(w,b,\alpha)$

所以，原始问题可以认为是极小极大问题，对偶问题是一个极大极小问题。

我们通过求解对偶问题得到原始问题的最优解，这样做的优点是：
1. 对偶问题往往更容易求解
2. 自然引入核函数，进而推广到非线性分类问题

为了得到对偶问题的解，需要先求$L(w,b,\alpha)$对$w,b$的极小，再求对$\alpha$的极大

（1）求$\mathop{\min}_{w,b} L(w,b,\alpha)$

将拉格朗日函数$L(w,b,\alpha)$分别对$w,b$求偏导数，并令其等于0.

$$
\begin{aligned}
&\triangledown_w L(w,b,\alpha) = w-\sum_{i=1}^N \alpha_i y_i x_i =0 \\\\
&\triangledown_b L(w,b,\alpha) = -\sum_{i=1}^N \alpha_i y_i = 0
\end{aligned}
$$
得

$$
\begin{aligned}
&w = \sum_{i=1}^N \alpha_i y_i x_i \\\\
&\sum_{i=1}^N \alpha_i y_i = 0
\end{aligned}
$$

代入拉格朗日函数，得到

$$
\begin{aligned}
    
L(w,b,\alpha) & = \frac{1}{2}{||w||}^2+\sum_{i=1}^N \alpha_i (1-y_i(w\cdot x_i+b)) \\\\
& = \frac{1}{2}{||w||}^2 - \sum_{i=1}^N \alpha_i y_i(w\cdot x_i+b) + \sum_{i=1}^N \alpha_i\\\\
& = \frac{1}{2}w w - \sum_{i=1}^N \alpha_i y_i w\cdot x_i - \sum_{i=1}^N \alpha_i y_ib + \sum_{i=1}^N \alpha_i\\\\
& = \frac{1}{2}w \sum_{i=1}^N \alpha_i y_i x_i - w \sum_{i=1}^N \alpha_i y_i\cdot x_i - b\sum_{i=1}^N \alpha_i y_i + \sum_{i=1}^N \alpha_i\\\\
& = -\frac{1}{2}w \sum_{i=1}^N \alpha_i y_i x_i - b\sum_{i=1}^N \alpha_i y_i + \sum_{i=1}^N \alpha_i\\\\
& = -\frac{1}{2}w \sum_{i=1}^N \alpha_i y_i x_i +\sum_{i=1}^N \alpha_i\\\\

& = -\frac{1}{2}\sum_{i=1}^N \alpha_i y_i x_i \sum_{i=1}^N \alpha_i y_i x_i +\sum_{i=1}^N \alpha_i\\\\
& = -\frac{1}{2}\sum_{i=1}^N \alpha_i y_i x_i \sum_{i=1}^N \alpha_i y_i x_i + \sum_{i=1}^N \alpha_i\\\\
& = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i  x_j +\sum_{i=1}^N \alpha_i\\\\
& = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i\\\\
\end{aligned}
$$

即：

$$
\mathop{\min}_{w,b} L(w,b,\alpha) =  -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i
$$

(2) 求$\mathop{\min}_{w,b} L(w,b,\alpha)$对$\alpha$的极大，即是对偶问题


$$
\begin{aligned}
    
\mathop{\max}_{\alpha}  &-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&\alpha_i \ge 0, i=1,2,\cdots,N

\end{aligned}
$$

等价于

$$
\tag{2}
\begin{aligned}
    
\mathop{\min}_{\alpha}  &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) -\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&\alpha_i \ge 0, i=1,2,\cdots,N

\end{aligned}
$$

由于原始最优化问题(1)满足定理C.2(详见：李航《统计学习方法》227 页，即存在$x$让目标函数的不等式约束严格可行，即不等式约束严格小于0)，所以存在$w^*,\alpha^*,\beta^*$,使$w^*$是原始问题的解，$\alpha^*,\beta^*$是对偶问题的解。这意味着求解原始问题可以转换为求解对偶问题.

对线性可分训练数据集，假设对偶最优化问题（2）对$\alpha$的解为$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^\top$,可由$\alpha^*$得到原始最优化问题（1）对$(w,b)$的解$w^*,b^*$,

根据定理C.3(详见：李航《统计学习方法》227 页，可能李老师多写了2个条件）,KKT条件成立，即：

1. 平稳性：$\triangledown_x L(x^*,\alpha^*,\beta^*) = 0$
2. 初始约束：
   * $c_i(x^*) \le 0, \quad i=1,2,\cdots,k$
   * $h_j(x^*)=0, \quad i=1,2,\cdots,l$
3. 双重约束：
   * $\alpha_i^* \ge 0, \quad i=1,2,\cdots,k$
4. 松弛互补条件：
   * $\alpha_i^* c_i(x^*) = 0, \quad i=1,2,\cdots,k$

因为我们的问题中没有等式约束，也自然没有对等式拉格朗日乘子的偏导，因此实际KKT条件为：

$$
\begin{aligned}
&\triangledown_w L(w^*,b^*,\alpha^*) = w^*-\sum_{i=1}^N \alpha_i^*y_i x_i=0 \\\\
&\triangledown_b L(w^*,b^*,\alpha^*) = -\sum_{i=1}^N \alpha_i^*y_i=0 \\\\
& 1-y_i(w^*\cdot x_i + b^*)\le 0, \quad i=1,2,\cdots,N \\\\
&\alpha_i^* \ge 0, i=1,2,\cdots,N \\\\
&\alpha_i^* (1-y_i(w^*\cdot x_i+b^*))=0 , \quad i=1,2,\cdots,N\\\\
\end{aligned}
$$

由此得：

$$
w^* = \sum_{i=1}^N \alpha_i^*y_i x_i
$$

其中至少有一个$\alpha_j^* > 0$(反证：假设$\alpha^*=0$,则$w^*=0$,而$w^*=0$不是原始最优化问题的解，矛盾)，对于此$j$，有

$$
1-y_j(w^*\cdot x_j+b^*) = 0
$$

可得：

$$
y_j(\sum_{i=1}^N \alpha_i^*y_i x_i x_j+b^*)=1=y_j^2 \\\\

\sum_{i=1}^N \alpha_i^*y_i x_i x_j+b^* = y_j \\\\

b^* = y_j - \sum_{i=1}^N \alpha_i^*y_i (x_i x_j)
$$

分离超平面为：

$$
w^*\cdot x+b^*=0 \\\\
\sum_{i=1}^N \alpha_i^*y_i (x\cdot x_i) + b^* =0 
$$

分类决策函数为：

$$
f(x) = sign\left(\sum_{i=1}^N \alpha_i^*y_i (x\cdot x_i) + b^* \right)
$$

由此可看出，分类决策函数只依赖于输入$x$和训练样本输入的内积。


### 线性可分支持向量机学习算法

输入：线性可分训练集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$,其中 $x_i \in \mathcal{X}=\mathrm{R}^n,y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$;

输出：分离超平面和分类决策函数。

（1）构造并求解约束最优化问题

$$
\begin{aligned}
    
\mathop{\min}_{\alpha}  &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) -\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&\alpha_i \ge 0, i=1,2,\cdots,N

\end{aligned}
$$
求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^\top$。

（2）计算

$$
w^* = \sum_{i=1}^N \alpha_i^*y_i x_i
$$

并选择$\alpha^*$的一个正分量$\alpha_j^* > 0$,计算

$$
b^* = y_j - \sum_{i=1}^N \alpha_i^*y_i (x_i x_j)
$$

（3）求得分离超平面：

$$
w^*\cdot x+b^*=0 
$$
分类决策函数：

$$
f(x) = sign\left(w^*\cdot x + b^* \right)
$$

由上面的式子可以看出，$w^*$和$b^*$只依赖于训练数据中对应于$\alpha_i^*>0$的样本点，而其他样本点对$w^*$和$b^*$没有影响。训练数据中对应于$\alpha_i^*>0$的实例点$x_i \in \mathrm{R}^n$称为支持向量。

## 线性支持向量机

线性可分SVM中要求数据必须是线性可分的，才可以找到分类的超平面，但是有的时候线性数据集中存在少量的异常点，由于这些异常点导致了数据集不能够线性划分;直白来讲就是:**正常数据本身是线性可分的，但是由于存在异常点数据，导致数据集不能够线性可分;**，此时我们可以通过引入软间隔的概念来解决这个问题;

具体做法是，对每个样本点$(x_i,y_i)$引入一个松弛变量$\xi \ge 0$,使函数间隔加上松弛变量大于等于1。此时约束条件变为

$$
y_i(w\cdot x_i+b)\ge 1-\xi_i
$$

同时，对每个松弛变量$\xi_i$，支付一个代价$\xi_i$,目标函数变为

$$
\frac{1}{2}{||w||}^2 + C\sum_{i=1}^N\xi_i
$$

这里，$C > 0$为惩罚参数，是一个超参数，$C$越大对误分类的惩罚越大。新的目标函数包含了两层含义：使$\frac{1}{2}{||w||}^2$尽量小，即间隔尽量大，同时使误分类点的个数尽量小，$C$是调和二者的系数。

此时，线性不可分的线性支持向量机的学习问题变为如下凸二次规划问题（原始问题）：

$$
\begin{aligned}
    
\mathop{\min}_{w,b} &\frac{1}{2}{||w||}^2 + C\sum_{i=1}^N\xi_i\\\\
s.t.\quad &y_i(w\cdot x_i + b) \ge 1-\xi_i, \quad i=1,2,\cdots,N  \\\\
& \xi_i \ge 0, \quad i=1,2,\cdots,N 

\end{aligned}
$$

即：


$$
\begin{aligned}
    
\mathop{\min}_{w,b} &\frac{1}{2}{||w||}^2 + C\sum_{i=1}^N\xi_i\\\\
s.t.\quad &1-\xi_i - y_i(w\cdot x_i + b) \le 0, \quad i=1,2,\cdots,N  \\\\
& -\xi_i \le 0, \quad i=1,2,\cdots,N 

\end{aligned}
$$


通过引入拉格朗日乘子$\alpha_i \ge 0,\mu_i \ge 0,i=1,2,\cdots,N$,构建出拉格朗日函数：

$$
L(w,b,\xi,\alpha,\mu) = \frac{1}{2}{||w||}^2+ C\sum_{i=1}^N\xi_i + \sum_{i=1}^N \alpha_i (1- \xi_i -y_i(w\cdot x_i+b)) + \sum_{i=1}^N -\mu_i\xi_i
$$

对偶问题是拉格朗日函数的极大极小问题，首先求$L(w,b,\xi,\alpha,\mu)$对$w,b,\xi$的极小，让偏导为0

$$
\begin{aligned}
&\triangledown_w L(w,b,\xi,\alpha,\mu) = w-\sum_{i=1}^N \alpha_i y_i x_i =0 \\\\
&\triangledown_b L(w,b,\xi,\alpha,\mu) = -\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&\triangledown_{\xi_i} L(w,b,\xi,\alpha,\mu) = C-\alpha_i-\mu_i = 0, \quad i=1,2,\cdots,N
\end{aligned}
$$

得

$$
\begin{aligned}
& w =\sum_{i=1}^N \alpha_i y_i x_i\\\\
&\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&C-\alpha_i-\mu_i = 0, \quad i=1,2,\cdots,N
\end{aligned}
$$

代入拉格朗日函数，得：

$$
\begin{aligned}
    
L(w,b,\xi,\alpha,\mu) & = \frac{1}{2}{||w||}^2+ C\sum_{i=1}^N\xi_i + \sum_{i=1}^N \alpha_i (1- \xi_i -y_i(w\cdot x_i+b)) + \sum_{i=1}^N -\mu_i\xi_i \\\\
& = \frac{1}{2}{||w||}^2 -\sum_{i=1}^N \alpha_i (y_i(w\cdot x_i+b)-1+ \xi_i ) + \sum_{i=1}^N (C-\mu_i)\xi_i \\\\
& = \frac{1}{2}{||w||}^2 -\sum_{i=1}^N \alpha_i (y_i(w\cdot x_i+b)-1)  - \sum_{i=1}^N\alpha_i \xi_i + \sum_{i=1}^N (C-\mu_i)\xi_i \\\\
& = \frac{1}{2}{||w||}^2 -\sum_{i=1}^N \alpha_i (y_i(w\cdot x_i+b)-1)  + \sum_{i=1}^N (C-\mu_i- \alpha_i)\xi_i \\\\
& = \frac{1}{2}{||w||}^2 -\sum_{i=1}^N \alpha_i (y_i(w\cdot x_i+b)-1) \\\\

\end{aligned}
$$
接下来的变换和线性可分支持向量机里的变换一样，最终

$$
L(w,b,\xi,\alpha,\mu) = -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i
$$

即

$$
\mathop{\min}_{w,b,\xi} L(w,b,\xi,\alpha,\mu) =  -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i
$$

再对$\mathop{\min}_{w,b,\xi} L(w,b,\xi,\alpha,\mu)$求$\alpha$的极大，即对偶问题：

$$
\begin{aligned}
    
\mathop{\max}_{\alpha} & -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) +\sum_{i=1}^N \alpha_i \\\\
s.t. \quad & \sum_{i=1}^N \alpha_i y_i = 0 \\\\
&C-\alpha_i-\mu_i = 0, \quad i=1,2,\cdots,N \\\\
& \alpha_i \ge 0 , \quad i=1,2,\cdots,N \\\\
& \mu_i \ge 0, \quad i=1,2,\cdots,N 
\end{aligned}
$$

利用约束中的等式，消去$\mu_i$,得到最终的对偶问题：

$$
\begin{aligned}
    
\mathop{\min}_{\alpha} & \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) -\sum_{i=1}^N \alpha_i \\\\
s.t. \quad & \sum_{i=1}^N \alpha_i y_i = 0 \\\\
& 0 \le \alpha_i \le C , \quad i=1,2,\cdots,N
\end{aligned}
$$

解满足KKT条件，即：

$$
\begin{aligned}
&\triangledown_w L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = w^*-\sum_{i=1}^N \alpha_i^*y_i x_i=0 \\\\
&\triangledown_b L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = -\sum_{i=1}^N \alpha_i^*y_i=0 \\\\
&\triangledown_\xi L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = C-\alpha^*-\mu^*=0 \\\\
&1-\xi_i^* - y_i(w^*\cdot x_i + b^*) \le 0 , \quad i=1,2,\cdots,N\\\\
& -\xi_i^* \le 0 , \quad i=1,2,\cdots,N\\\\
&\alpha_i^* \ge 0, \quad i=1,2,\cdots,N \\\\
&\mu_i^* \ge 0, \quad i=1,2,\cdots,N \\\\
&\alpha_i^* (1- \xi_i^* -y_i(w^*\cdot x_i+b^*)) =0, \quad i=1,2,\cdots,N\\\\
& -\mu_i^*\xi_i^*=0, \quad i=1,2,\cdots,N

\end{aligned}
$$

由以上条件可得：

$$
w^* = \sum_{i=1}^N \alpha_i^*y_i x_i
$$

若存在 $\alpha_j^*$,使得$0<\alpha_j^*<C$,则对应的$\mu_j^*>0$,所以对应的$\xi_j^*=0$,因此有

$$
\begin{aligned}
    
&y_j(w^*\cdot x_j+b^*)-1=0 \\\\

&y_j(\sum_{i=1}^N \alpha_i^*y_i x_i\cdot x_j+b^*)=1=y_j^2 \\\\
&\sum_{i=1}^N \alpha_i^*y_i (x_i x_j)+b^* = y_j \\\\
&b^* = y_j - \sum_{i=1}^N \alpha_i^*y_i (x_i x_j)

\end{aligned}
$$

进而得到分割超平面和决策函数

### 线性支持向量机学习算法

输入：训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$,其中 $x_i \in \mathcal{X}=\mathrm{R}^n,y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$;

输出：分离超平面和分类决策函数。

（1）选择惩罚参数$C>0$,构造并求解凸二次规划问题

$$
\begin{aligned}
    
\mathop{\min}_{\alpha}  &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i  x_j) -\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&0 \le \alpha_i \le C, i=1,2,\cdots,N

\end{aligned}
$$
求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^\top$。

（2）计算

$$
w^* = \sum_{i=1}^N \alpha_i^*y_i x_i
$$

并选择$\alpha^*$的一个分量$\alpha_j^*$适合条件$0<\alpha_j^* < C$,计算

$$
b^* = y_j - \sum_{i=1}^N \alpha_i^*y_i (x_i x_j)
$$
对任一适合条件$0<\alpha_j^* < C$的$\alpha_j^*$,都可求出$b^*$,但是由于原始问题对$b$的解并不唯一，所以实际计算时可以取在所有符合条件的样本点上的平均值。即：

$$
b^* = \frac{1}{J}\sum_{j=1}^J \left(y_j - \sum_{i=1}^N \alpha_i^*y_i (x_i x_j) \right)
$$
其中$x_j,y_j$为对应条件$0<\alpha_j^* < C$的样本点，$J$为符合条件的样本点个数。


（3）求得分离超平面：

$$
w^*\cdot x+b^*=0 
$$
分类决策函数：

$$
f(x) = sign\left(w^*\cdot x + b^* \right)
$$


### 支持向量

在线性不可分的情况下，对偶问题的解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^\top$中对应于$\alpha_i^*>0$的样本点$(x_i,y_i)$的实例$x_i$称为支持向量（软间隔的支持向量）


![软间隔的支持向量](/assets/images/机器学习/svm/软间隔的支持向量.png)
<center>软间隔的支持向量</center>

软间隔的支持向量有4种情况：

1. $\alpha_i^*<C$,则$\xi_i=0$,支持向量$x_i$恰好在间隔边界上；
2. $\alpha_i^*=C,0<\xi_i<1$,则分类正确，$x_i$在间隔边界与分离超平面之间；
3. $\alpha_i^*=C,\xi_i=1$,则$x_i$在分离超平面上；
4. $\alpha_i^*=C,\xi_i>1$,则$x_i$位于分离超平面误分一侧；

## 非线性支持向量机

非线性分类问题是指通过利用非线性模型才能很好地进行分类的问题。如下图左侧部分，我们无法用直线（线性模型）将正负实例正确分开，但可以用一条椭圆曲线（非线性模型）将他们正确分开。

![非线性分类问题](/assets/images/机器学习/svm/非线性分类问题.png)
<center>非线性分类问题</center>

非线性问题往往不好求解，但我们可以用一个非线性变换，将非线性问题变换为线性问题，通过解变换后的线性问题来解原来的非线性问题，如上图，将空间变换后，原空间的点相应地变换为新空间中的点，原空间的椭圆也变换为新空间中的直线，从而把原空间中的非线性分类问题变换为新空间中的线性分类问题。

上例说明，用线性分类方法求解非线性分类问题分为两步：
1. 使用一个变换将原空间数据映射到新空间
2. 在新空间用线性分类学习方法从训练数据中学习分类模型。

核技巧就属于这样的方法。

定义（核函数）设$\mathcal{X}$是输入空间（欧氏空间$\mathrm{R}^n$的子集或离散集合）,又设$\mathcal{H}$为特征空间（希尔伯特空间），如果存在一个从$\mathcal{X}$到$\mathcal{H}$的映射

$$
\phi(x):\mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有$x,z \in \mathcal{X}$,函数$K(x,z)$满足条件

$$
K(x,z)=\phi(x)\cdot\phi(z)
$$
则称$K(x,z)$为核函数，$\phi(x)$为映射函数，式中$\phi(x)\cdot\phi(z)$为$\phi(x)$和$\phi(z)$的内积。

核技巧的想法是，在学习与预测中只定义核函数$K(x,z)$，而不显式地定义映射函数$\phi$。通常，直接计算$K(x,z)$比较容易，而通过$\phi(x)$和$\phi(z)$计算$K(x,z)$并不容易。注意，$\phi$是输入空间$\mathrm{R}^n$到特征空间$\mathcal{H}$的映射，特征空间$\mathcal{H}$一般是高维的，甚至是无穷维的。对于给定的核函数$K(x,z)$，特征空间$\mathcal{H}$和映射函数$\phi$的取法并不唯一。可以取不同的特征空间，即便是在同一特征空间里也可以取不同的映射。

举例：输入空间$\mathrm{R}^2$，核函数$K(x,z)=(x\cdot z)^2$。特征空间和映射函数可以有多种
1. $\mathcal{H}=\mathrm{R}^3$,$\phi(x)=(x_1^2,\sqrt{2}x_1x_2,x_2^2)^\top$
2. $\mathcal{H}=\mathrm{R}^3$,$\phi(x)=\frac{1}{\sqrt{2}}(x_1^2-x_2^2,2 x_1x_2,x_1^2+x_2^2)^\top$
3. $\mathcal{H}=\mathrm{R}^4$,$\phi(x)=(x_1^2,x_1x_2,x_1x_2,x_2^2)^\top$

可验证在新空间中的$\phi(x)$和$\phi(z)$的内积$\phi(x)\cdot \phi(z)=K(x,z)$

注意到在线性支持向量机的对偶问题中，无论是目标函数还是决策函数（分离超平面）都只涉及输入实例与实例之间的内积。在对偶问题的目标函数中的内积$x_i\cdot x_j$可以用核函数$K(x_i,x_j)$来代替，此时，对偶问题目标函数为：

$$
\mathop{\min}_{\alpha}  \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) -\sum_{i=1}^N \alpha_i 
$$

同样，分类决策函数中的内积也可以用核函数代替，变为：

$$
\begin{aligned}
f(x) &= sign\left( \sum_{i=1}^N \alpha_i^*y_i \phi(x_i)\cdot \phi(x) + b^* \right) \\\\
&=sign\left( \sum_{i=1}^N \alpha_i^*y_i K(x_i,x) + b^* \right)
\end{aligned}
$$

这意味着，在核函数$K(x,z)$给定的条件下，可以利用解线性分类问题的方法求解非线性分类问题的支持向量机。学习时隐式地在特征空间进行的，不需要显式地定义特征空间和映射函数。这样的技巧称为核技巧。在实际应用中，往往依赖领域知识直接选择核函数，核函数选择的有效性需要通过实验验证。

核函数总结：
1. 核函数可以自定义；核函数必须是正定核函数，即Gram矩阵是半正定矩阵；
2. 核函数的价值在于它虽然也是将特征进行从低维到高维的转换，但核函数它事先在低维上进行计算，而将实质上的分类效果表现在了高维上，避免了直接在高维空间中的复杂计算;
3. 通过核函数，可以将非线性可分的数据转换为线性可分数据;



Gram矩阵定义：
$n$维欧式空间中任意$k(k\le n)$个向量$\alpha_1,\alpha_2,\cdots,\alpha_k$的内积所组成的矩阵

$$
\begin{matrix}
(\alpha_1,\alpha_1) & (\alpha_1,\alpha_2) & \cdots &(\alpha_1,\alpha_k) \\\\
(\alpha_2,\alpha_1) & (\alpha_2,\alpha_2) & \cdots &(\alpha_2,\alpha_k) \\\\
\vdots & \vdots & \ddots &\vdots \\\\
(\alpha_k,\alpha_1) & (\alpha_k,\alpha_2) & \cdots &(\alpha_k,\alpha_k) \\\\
\end{matrix}
$$
称为$k$个向量$\alpha_1,\alpha_2,\cdots,\alpha_k$的格拉姆矩阵（Gram矩阵）

定义（正定核）设$\mathcal{X}\subset \mathrm{R}^n$,$K(x,z)$是定义在$\mathcal{X}\times \mathcal{X}$上的对称函数，如果对于任意$x_i \in \mathcal{X},i=1,2,\cdots,m$,$K(x,z)$对应的Gram矩阵：

$$
\begin{matrix}
K(x_1,x_1) & K(x_1,x_2) & \cdots &K(x_1,x_m) \\\\
K(x_2,x_1) & K(x_2,x_2) & \cdots &K(x_2,x_m) \\\\
\vdots & \vdots & \ddots &\vdots \\\\
K(x_m,x_1) & K(x_m,x_2) & \cdots &K(x_k,x_m) \\\\
\end{matrix}
$$
是半正定矩阵，则称$K(x,z)$是正定核。

这一定义在构造核函数时很有用。但对于一个具体函数$K(x,z)$来说，检验它是否为正定核函数并不容易，因为要求对任意有限输入集$\{x_1,x_2,\cdots,x_m\}$验证$K$对应的Gram矩阵是否为半正定的。在实际问题中往往应用已有的核函数。

常用核函数：
1. 线性核函数：$K(x,z)=x\cdot z$
1. 多项式核函数： $K(x,z)=(\gamma x\cdot z+r)^d$,其中$\gamma,r,d$为超参数
2. 高斯核函数：$K(x,z)=\exp\left(-\frac{||x-z||^2}{2\sigma^2}\right)=e^{-\gamma||x-z||^2}$,其中$r>0$，为超参数
3. Sigmoid核函数：$K(x,z)=\tanh(\gamma x \cdot z+r)$,其中$\gamma,r$为超参数


![核函数](/assets/images/机器学习/svm/核函数.png)
<center>核函数</center>

### 非线性支持向量机学习算法

输入：训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$,其中 $x_i \in \mathcal{X}=\mathrm{R}^n,y_i \in \mathcal{Y}=\{+1,-1\},i=1,2,\cdots,N$;

输出：分类决策函数。

（1）选择适当的核函数$K(x,z)$和适当的惩罚参数$C>0$,构造并求解最优化问题

$$
\begin{aligned}
    
\mathop{\min}_{\alpha}  &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) -\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&0 \le \alpha_i \le C, i=1,2,\cdots,N

\end{aligned}
$$
求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^\top$。

（2）选择$\alpha^*$的一个分量$\alpha_j^*$适合条件$0<\alpha_j^* < C$,计算

$$
b^* = y_j - \sum_{i=1}^N \alpha_i^*y_i K(x_i,x_j)
$$

（3）求得分类决策函数：

$$
f(x) = sign\left(\sum_{i=1}^N \alpha_i^*y_i K(x\cdot x_i) + b^* \right)
$$

当$K(x,z)$是正定核函数时，上述问题是凸二次规划问题，解是存在的。

## SMO （序列最小最优化算法，sequential minimal optimization)

根据前面的介绍，我们发现，问题最终归结为求解对偶问题的最优解，即：

$$
\begin{aligned}
    
\mathop{\min}_{\alpha}  &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) -\sum_{i=1}^N \alpha_i\\\\
s.t.\quad &\sum_{i=1}^N \alpha_i y_i = 0 \\\\
&0 \le \alpha_i \le C, i=1,2,\cdots,N

\end{aligned}
$$

这是一个凸二次规划问题。凸二次规划问题具有全局最优解，有许多最优化算法可以用来求解该问题。但当训练样本容量很大时，这些算法往往变得很低效，以致无法使用，而序列最小最优化（SMO）算法即是一种快速的求解算法。

SMO 算法是一种启发式算法，基本思路是：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。可以想象，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题，其最优解也会让原始问题的目标函数变得更小。重要的是，这时子问题可以通过解析方法求解，大大提高计算速度。子问题有两个变量，一个是违反KKT条件最严重的，另一个由等式约束条件自动确定。如此，SMO算法将原问题不断分解为子问题求解，进而达到求解原问题的目的。

子问题虽然有两个变量，但因为有等式约束,所以自由变量相当于只有一个，因此，子问题中同时更新了两个变量。

整个SMO算法包括两部分：
1. 求解两个变量的二次规划的解析方法
2. 选择变量的启发式方法。

### 两个变量的二次规划的求解方法

不失一般性，假设选择的两个变量是$\alpha_1,\alpha_2$，其他变量$\alpha_i(i=3,4,\cdots,N)$固定。对偶优化问题的子问题变为：

$$
\begin{aligned}

\mathop{\min}_{\alpha_1,\alpha_2}  W(\alpha_1,\alpha_2)&=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) -\sum_{i=1}^N \alpha_i\\\\
&=\frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 + y_1y_2K_{12}\alpha_1\alpha_2 \\\\
&\qquad + y_1\alpha_1\sum_{i=3}^N y_i\alpha_i K_{i1} + y_2\alpha_2\sum_{i=3}^N y_i\alpha_i K_{i2} \\\\
&\qquad + \frac{1}{2}\sum_{i=3}^N\sum_{j=3}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) -(\alpha_1+\alpha_2)-\sum_{i=3}^N \alpha_i \\\\

s.t.\quad & \alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N y_i\alpha_i=\varsigma
\\\\
&0 \le \alpha_i \le C, i=1,2
\end{aligned}
$$
其中，$K_{ij}=K(x_i,x_j),i,j=1,2,\cdots,N$,$\varsigma$为常数，去掉目标函数中的常数项，最终问题为：


$$
\begin{aligned}

\mathop{\min}_{\alpha_1,\alpha_2}  W(\alpha_1,\alpha_2)&=\frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 + y_1y_2K_{12}\alpha_1\alpha_2 \\\\
&\qquad + y_1\alpha_1\sum_{i=3}^N y_i\alpha_i K_{i1} + y_2\alpha_2\sum_{i=3}^N y_i\alpha_i K_{i2} \\\\
&\qquad -(\alpha_1+\alpha_2) \\\\

s.t.\quad & \alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N y_i\alpha_i=\varsigma
\\\\
&0 \le \alpha_i \le C, i=1,2
\end{aligned}
$$


首先分析约束条件，在约束条件下求极小。考虑到$y_1,y_2$只能取$\pm 1$,所以$\alpha_1,\alpha_2$的等式约束是斜率为$\pm 1$的线性关系。而不等式约束则是把他们限制在$[0,C]\times [0,C]$的盒子里，我们可以用二维空间中的图像表示约束，横轴表示$\alpha_1$，纵轴表示$\alpha_2$。


![smo约束1](/assets/images/机器学习/svm/smo约束1.png)
<center>smo约束1</center>

![smo约束2](/assets/images/机器学习/svm/smo约束2.png)
<center>smo约束2</center>

因为等式约束的原因，最优化问题实质上是单变量的最优化问题，不妨考虑为变量$\alpha_2$的最优化问题。假设问题初始可行解为$\alpha_1^{\mathrm{old}},\alpha_2^{\mathrm{old}}$,最优解为$\alpha_1^{\mathrm{new}},\alpha_2^{\mathrm{new}}$,并假设只有等式约束时的最优解（沿着约束方向未经剪辑时）为$\alpha_2^{\mathrm{new,unc}}$

由于$\alpha_2^{\mathrm{new}}$需要满足不等式约束，记最优解$\alpha_2^{\mathrm{new}}$的取值范围最小为$L$,最大为$H$,由图可以看出$\alpha_2^{\mathrm{new}}$的取值范围为：

当$y_1 \ne y_2$时：
$$
\begin{aligned}
L &= \max(0,-k) =\max(0,\alpha_2^{\mathrm{old}}-\alpha_1^{\mathrm{old}})\\\\
H &= \min(C,C-k) = \min(C,C+\alpha_2^{\mathrm{old}}-\alpha_1^{\mathrm{old}})
\end{aligned}
$$

当$y_1 = y_2$时：
$$
\begin{aligned}
L &= \max(0,k-C) = \max(0,\alpha_2^{\mathrm{old}}+\alpha_1^{\mathrm{old}}-C) \\\\
H &= \min(C,k)= \min(C,\alpha_2^{\mathrm{old}}+\alpha_1^{\mathrm{old}})
\end{aligned}
$$

下面，我们首先求解等式约束下$\alpha_2$的最优解：$\alpha_2^{\mathrm{new,unc}}$,然后再求剪辑后的解$\alpha_2^{\mathrm{new}}$。

引入记号$v_i=\sum_{i=3}^Ny_i\alpha_iK_{ij},i=1,2$.则目标函数可写成：

$$
\begin{aligned}
    
W(\alpha_1,\alpha_2) = &\frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 + y_1y_2K_{12}\alpha_1\alpha_2 \\\\
 &-(\alpha_1+\alpha_2)+y_1v_1\alpha_1+y_2v_2\alpha_2
\end{aligned}
$$
由$\alpha_1y_1 = \varsigma-\alpha_2y_2$及$y_i^2=1$,可得：

$$
\alpha_1y_1y_1 = (\varsigma-\alpha_2y_2)y_1 \\\\
\alpha_1 = (\varsigma-\alpha_2y_2)y_1 
$$
则

$$
\begin{aligned}
    
W(\alpha_2) = &\frac{1}{2}K_{11}(\varsigma-\alpha_2y_2)^2 + \frac{1}{2}K_{22}\alpha_2^2 + y_2K_{12}(\varsigma-\alpha_2y_2)\alpha_2 \\\\
&-(\varsigma-\alpha_2y_2)y_1-\alpha_2+v_1(\varsigma-\alpha_2y_2)+y_2v_2\alpha_2
\end{aligned}
$$

对$\alpha_2$求导

$$
\begin{aligned}
    
\frac{\partial W}{\partial \alpha_2} = & K_{11}(\varsigma-\alpha_2y_2)(-y_2)+ K_{22}\alpha_2\\\\
& +y_2K_{12}(-y_2)\alpha_2 +  y_2K_{12}(\varsigma-\alpha_2y_2) \\\\
& + y_1y_2-1-v_1y_2+y_2v_2 \\\\
=& -K_{11}\varsigma y_2 + K_{11}\alpha_2+ K_{22}\alpha_2 \\\\
& -K_{12}\alpha_2 + y_2K_{12}\varsigma - K_{12}\alpha_2 \\\\
& + y_1y_2-1-v_1y_2+y_2v_2 \\\\
=& K_{11}\alpha_2+ K_{22}\alpha_2-2K_{12}\alpha_2 \\\\
& -K_{11}\varsigma y_2 + K_{12}\varsigma y_2 + y_1y_2-1-v_1y_2+y_2v_2
\end{aligned}
$$

令其为0，得到

$$
\begin{aligned}
(K_{11}+ K_{22}-2K_{12})\alpha_2 = & 1-y_1y_2+K_{11}\varsigma y_2-K_{12}\varsigma y_2 +v_1y_2-y_2v_2 \\\\
=&y_2(y_2-y_1+K_{11}\varsigma -K_{12}\varsigma +v_1-v_2) 
\end{aligned}
$$

我们记分离超平面：

$$
g(x) = \sum_{j=1}^N \alpha_jy_jK(x_j,x)+b
$$
令

$$
E_i=g(x_i)-y_i = \sum_{j=1}^N \alpha_jy_jK(x_j,x_i)+b-y_i,\qquad i=1,2
$$

当$i=1,2$时，$E_i$为函数$g(x)$对输入$x_i$的预测值与真实输出$y_i$之差。

而

$$
v_i=\sum_{i=3}^Ny_i\alpha_iK_{ij} = g(x_i)-\sum_{j=1}^2 \alpha_jy_jK(x_j,x_i) - b,\qquad i=1,2
$$

则：

$$
\begin{aligned}
(K_{11}+ K_{22}-2K_{12})\alpha_2 = &y_2(y_2-y_1+K_{11}\varsigma -K_{12}\varsigma +v_1-v_2) \\\\
=&y_2[ y_2-y_1+K_{11}\varsigma-K_{12}\varsigma \\\\
&+ (g(x_1)-\sum_{j=1}^2 \alpha_j^{\mathrm{old}}y_jK_{1j} - b) \\\\
&-(g(x_2)-\sum_{j=1}^2 \alpha_j^{\mathrm{old}}y_jK_{2j} - b)]
\end{aligned}
$$

因为

$$
\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N y_i\alpha_i=\varsigma
$$
所以

$$
\varsigma = \alpha_1^{\mathrm{old}}y_1+\alpha_2^{\mathrm{old}}y_2
$$

因此：

$$
\begin{aligned}
(K_{11}+ K_{22}-2K_{12})\alpha_2^{\mathrm{new,unc}} = &y_2[ y_2-y_1+K_{11}\varsigma-K_{12}\varsigma \\\\
&+ (g(x_1)-\sum_{j=1}^2 \alpha_j^{\mathrm{old}}y_jK_{1j} - b) \\\\
&-(g(x_2)-\sum_{j=1}^2 \alpha_j^{\mathrm{old}}y_jK_{2j} - b)] \\\\
=&y_2[ y_2-y_1+K_{11}(\alpha_1^{\mathrm{old}}y_1+\alpha_2^{\mathrm{old}}y_2)-K_{12}(\alpha_1^{\mathrm{old}}y_1+\alpha_2^{\mathrm{old}}y_2) \\\\
&+ (g(x_1)-\alpha_1^{\mathrm{old}}y_1K_{11}-\alpha_2^{\mathrm{old}}y_2K_{12} - b) \\\\
&-(g(x_2)- \alpha_1^{\mathrm{old}}y_1K_{21}-\alpha_2^{\mathrm{old}}y_2K_{22} - b)] \\\\
=&y_2[ y_2-y_1+\alpha_1^{\mathrm{old}}y_1K_{11}+\alpha_2^{\mathrm{old}}y_2K_{11}-\alpha_1^{\mathrm{old}}y_1K_{12}-\alpha_2^{\mathrm{old}}y_2K_{12} \\\\
&+ g(x_1)-\alpha_1^{\mathrm{old}}y_1K_{11}-\alpha_2^{\mathrm{old}}y_2K_{12} - b \\\\
&-g(x_2)+ \alpha_1^{\mathrm{old}}y_1K_{21}+\alpha_2^{\mathrm{old}}y_2K_{22} + b] \\\\
=&y_2[ y_2-y_1+\alpha_2^{\mathrm{old}}y_2K_{11}-\alpha_2^{\mathrm{old}}y_2K_{12} \\\\
&+ g(x_1)-\alpha_2^{\mathrm{old}}y_2K_{12} - b \\\\
&-g(x_2)+\alpha_2^{\mathrm{old}}y_2K_{22} + b] \\\\
=&y_2[ \alpha_2^{\mathrm{old}}y_2(K_{11}+K_{22}-2K_{12})+y_2-y_1+ g(x_1)-g(x_2)] \\\\
=&y_2[ \alpha_2^{\mathrm{old}}y_2(K_{11}+K_{22}-2K_{12})+E_1-E_2] \\\\
=&\alpha_2^{\mathrm{old}}(K_{11}+K_{22}-2K_{12})+y_2(E_1-E_2)
\end{aligned}
$$

令$\eta=K_{11}+K_{22}-2K_{12}$,代入上式，得到：

$$
\eta \alpha_2^{\mathrm{new,unc}} = \eta\alpha_2^{\mathrm{old}} + y_2(E_1-E_2) \\\\
\alpha_2^{\mathrm{new,unc}} = \alpha_2^{\mathrm{old}} + \frac{y_2(E_1-E_2)}{\eta }
$$

将其限制在区间$[L,H]$内，得到：

$$
\alpha_2^{\mathrm{new}}  = 
\begin{cases}
\begin{aligned}
&H, &\alpha_2^{\mathrm{new,unc}} >H \\\\
&\alpha_2^{\mathrm{new,unc}} ,&L \le \alpha_2^{\mathrm{new,unc}}  \le H \\\\
&L,&\alpha_2^{\mathrm{new,unc}}  < L
\end{aligned}
\end{cases}
$$

由$\alpha_2^{\mathrm{new,unc}}$求得

$$
\begin{aligned}
    
&\alpha_1^{\mathrm{new}}y_1 + \alpha_2^{\mathrm{new}}y_2 = \varsigma = \alpha_1^{\mathrm{old}}y_1 + \alpha_2^{\mathrm{old}}y_2 \\\\ 

&\alpha_1^{\mathrm{new}} + \alpha_2^{\mathrm{new}}y_1y_2 = \alpha_1^{\mathrm{old}} + \alpha_2^{\mathrm{old}}y_1y_2 \\\\ 
&\alpha_1^{\mathrm{new}} = \alpha_1^{\mathrm{old}} + y_1y_2(\alpha_2^{\mathrm{old}}- \alpha_2^{\mathrm{new}}) \\\\ 
\end{aligned}
$$

### 变量的选择方法

SMO 算法在每个子问题中需要选择两个变量优化，第一个变量的选择，需要选取训练集上违反KKT条件最严重的样本点。一般情况下，先选择$0<\alpha_i<C$的样本点（即支持向量），只有当所有的支持向量都满足KKT条件的时候，才会选择其他样本点。因为此时违反KKT条件越严重，在经过一次优化后，会让变量$\alpha_i$尽可能的发生变化，从而可以以更少的迭代次数让模型达到$g(x)$目标条件。具体地，检验训练样本点$(x_i,y_i)$是否满足KKT条件。

回顾KKT的条件：

$$
\begin{aligned}
&\triangledown_w L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = w^*-\sum_{i=1}^N \alpha_i^*y_i x_i=0 \\\\
&\triangledown_b L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = -\sum_{i=1}^N \alpha_i^*y_i=0 \\\\
&\triangledown_\xi L(w^*,b^*,\xi^*,\alpha^*,\mu^*) = C-\alpha^*-\mu^*=0 \\\\
&1-\xi_i^* - y_i(w^*\cdot x_i + b^*) \le 0 , \quad i=1,2,\cdots,N\\\\
& -\xi_i^* \le 0 , \quad i=1,2,\cdots,N\\\\
&\alpha_i^* \ge 0, \quad i=1,2,\cdots,N \\\\
&\mu_i^* \ge 0, \quad i=1,2,\cdots,N \\\\
&\alpha_i^* (1- \xi_i^* -y_i(w^*\cdot x_i+b^*)) =0, \quad i=1,2,\cdots,N\\\\
& -\mu_i^*\xi_i^*=0, \quad i=1,2,\cdots,N

\end{aligned}
$$

注意KKT条件中的松弛互补条件：

$$
\begin{aligned}
&\alpha_i^* (1- \xi_i^* -y_i(w^*\cdot x_i+b^*)) =0, \quad i=1,2,\cdots,N\\\\
& -\mu_i^*\xi_i^*=0, \quad i=1,2,\cdots,N
\end{aligned}
$$
和$\triangledown\xi=0$的条件：

$$
C-\alpha^*-\mu^*=0
$$

由此，我们有如下关系：

$$
\begin{aligned}
\alpha_i=0 \Rightarrow \mu_i>0 \Rightarrow \xi_i=0 \Rightarrow y_i(w\cdot x_i+b) \ge 1 \Rightarrow y_ig(x_i) \ge 1 \\\\

0 < \alpha_i<C \Rightarrow \mu_i>0 \Rightarrow \xi_i=0 \Rightarrow y_i(w\cdot x_i+b) = 1 \Rightarrow y_ig(x_i) = 1 \\\\

\alpha_i=C \Rightarrow \mu_i=0 \Rightarrow \xi_i \ge 0 \Rightarrow y_i(w\cdot x_i+b) \le 1 \Rightarrow y_ig(x_i) \le 1 \\\\
    
\end{aligned}
$$
其中$g(x_i) = \sum_{j=1}^N \alpha_jy_jK(x_j,x_i)+b$

我们检验是否满足KKT，即检验是否满足条件：：

$$
\begin{aligned}
\alpha_i=0 \Rightarrow y_ig(x_i) \ge 1 \\\\

0 < \alpha_i<C  \Rightarrow y_ig(x_i) = 1 \\\\

\alpha_i=C  \Rightarrow y_ig(x_i) \le 1 \\\\
    
\end{aligned}
$$

注意，检验是在$\varepsilon$范围内进行的。

第2个变量的选择的标准是希望能使$\alpha_2$有足够大的变化。因为

$$
\alpha_2^{\mathrm{new,unc}} = \alpha_2^{\mathrm{old}} + \frac{y_2(E_1-E_2)}{\eta }
$$

和

$$
\alpha_2^{\mathrm{new}}  = 
\begin{cases}
\begin{aligned}
&H, &\alpha_2^{\mathrm{new,unc}} >H \\\\
&\alpha_2^{\mathrm{new,unc}} ,&L \le \alpha_2^{\mathrm{new,unc}}  \le H \\\\
&L,&\alpha_2^{\mathrm{new,unc}}  < L
\end{aligned}
\end{cases}
$$
所以，$\alpha_2^{\mathrm{new}}$是依赖于$|E_1-E_2|$，为了加快计算速度，一种简单的做法是选择$\alpha_2$，使对应的$|E_1-E_2|$最大。因为$\alpha_1$已定，$E_1$也确定了，如果$E_1$为正，那么选择最小的$E_i$作为$E_2$,如果$E_1$为负，那么选择最大的$E_i$作为$E_2$。为了节省计算时间，将所有$E_i$值保存在一个列表中。

在特殊情况下，如果选择的第二个变量不能够让目标函数有足够的下降，那么可以采用启发式规则，遍历在间隔边界上的支持向量，依次用其对应的变量作为$\alpha_2$,直到目标函数有足够的下降。若找不到合适的$\alpha_2$,则遍历所有样本点，直到目标函数有足够下降，如果都没有足够下降，则放弃选择的第一个变量$\alpha_1$，重新寻找$\alpha_1$

在每次完成两个变量的优化后，都要重新计算阈值$b$和差值$E_i$，当$0<\alpha_1^\mathrm{new}<C$时，由KKT条件可知：

$$
\sum_{i=1}^N\alpha_iy_iK_{i1}+b=y_1
$$

得

$$
b_1^\mathrm{new} = y_1 - \sum_{i=3}^N\alpha_iy_iK_{i1}-\alpha_1^\mathrm{new}y_1K_{11}-\alpha_2^\mathrm{new}y_2K_{21}
$$

因为

$$
\begin{aligned}
    
E_1 &= g(x_1)-y_1 \\\\
&= \sum_{i=1}^N \alpha_iy_iK_{i1}+b-y_1 \\\\
&= \sum_{i=3}^N \alpha_iy_iK_{i1} + \alpha_1^\mathrm{old}y_1K_{11}+\alpha_2^\mathrm{old}y_2K_{21}+b^\mathrm{old}-y_1
\end{aligned}
$$

所以

$$
b_1^{\mathrm{new}} = -E_1-y_1K_{11}(\alpha_1^{\mathrm{new}}-\alpha_1^{\mathrm{old}})-y_2K_{21}(\alpha_2^{\mathrm{new}}-\alpha_2^{\mathrm{old}})+b^{\mathrm{old}}
$$

同样，如果$0<\alpha_2^\mathrm{new}<C$，那么：

$$
b_2^{\mathrm{new}} = -E_2-y_1K_{12}(\alpha_1^{\mathrm{new}}-\alpha_1^{\mathrm{old}})-y_2K_{22}(\alpha_2^{\mathrm{new}}-\alpha_2^{\mathrm{old}})+b^{\mathrm{old}}
$$

如果$\alpha_1^\mathrm{new},\alpha_2^\mathrm{new}$同时满足条件$0<\alpha_i^\mathrm{new}<C,i=1,2$，那么$b_1^\mathrm{new}=b_2^\mathrm{new}$,如果$\alpha_1^\mathrm{new},\alpha_2^\mathrm{new}$是$0$或者$C$,那么$b_1^\mathrm{new}$和$b_2^\mathrm{new}$以及他们之间的数都是符合KKT条件的阈值，这时选择他们的中点作为$b^\mathrm{new}$。
在每次完成两个变量的优化之后，还必须更新对应的$E_i$值，并将他们保存在列表中，$E_i$值的更新要用到$b^\mathrm{new}$，以及所有支持向量对应的$\alpha_j$:

$$
E_i^\mathrm{new} = \sum_S y_j\alpha_jK(x_i,x_j)+b^\mathrm{new}-y_i
$$

其中，$S是所有支持向量$x_j$的集合。

### SMO 算法

输入：训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$,其中，$x_i \in \mathcal{X}=\mathrm{R}^n$,$y_i \in \mathcal{Y}=\{-1,+1\}$,$i=1,2,\cdots,N$，精度$\varepsilon$:

输出：近似解$\hat{\alpha}$

1. 取初值$\alpha^{(0)}=0$,令$k=0$;
2. 选取优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$，解析求解两个变量的最优化问题。
  
   先计算
   
   $$
    \alpha_2^{\mathrm{new,unc}} = \alpha_2^{(k)} + \frac{y_2(E_1-E_2)}{K_{11}+K_{22}-2K_{12} }
   $$

    得到

   $$
    \alpha_2^{(k+1)}  = 
    \begin{cases}
    \begin{aligned}
    &H, &\alpha_2^{\mathrm{new,unc}} >H \\\\
    &\alpha_2^{\mathrm{new,unc}} ,&L \le \alpha_2^{\mathrm{new,unc}}  \le H \\\\
    &L,&\alpha_2^{\mathrm{new,unc}}  < L
    \end{aligned}
    \end{cases}
   $$
    和
   $$
    \alpha_1^{(k+1)} = \alpha_1^{k} + y_1y_2(\alpha_2^{k}- \alpha_2^{(k+1)}) 
   $$
    更新

   $$
    b_1^{(k+1)} = -E_1-y_1K_{11}(\alpha_1^{(k+1)}-\alpha_1^{(k)})-y_2K_{21}(\alpha_2^{(k+1)}-\alpha_2^{(k)})+b^{(k)}
   $$
   $$
    b_2^{(k+1)} = -E_2-y_1K_{12}(\alpha_1^{(k+1)}-\alpha_1^{(k)})-y_2K_{22}(\alpha_2^{(k+1)}-\alpha_2^{(k)})+b^{(k)}
   $$
   $$
    b^{(k+1)} = \frac{b_1^{(k+1)}+b_2^{(k+1)}}{2}
   $$
   $$
    E_1^\mathrm{(k+1)} = \sum_S y_j\alpha_jK(x_1,x_j)+b^{(k+1)}-y_1
   $$
   $$
    E_2^\mathrm{(k+1)} = \sum_S y_j\alpha_jK(x_2,x_j)+b^{(k+1)}-y_2
   $$
   
3. 若在精度$\varepsilon$范围内满足停机条件

    $$
    \sum_{i=1}^N\alpha_iy_i=0 \\\\

    0 \le \alpha_i \le C,i=1,2,\cdots,N \\\\
    y_i\cdot g(x_i)=
    \begin{cases}
        \begin{aligned}
            \ge 1, & \{x_i|\alpha_i=0\} \\\\
            =1, & \{x_i|0<\alpha_i <C\} \\\\
            \le 1, & \{x_i|\alpha_i=C\} 
        \end{aligned}
    \end{cases}
    $$
    其中，$g(x_i)=\sum_{i=1}^N \alpha_j y_j K(x_j,x_i)+b$

    则转（4）：否则令$k=k+1$，转（2）

4. 取 $\hat{\alpha} = \alpha^{(k+1)}$

## SVR

## scikit-learn SVM 算法库

## 例子


---
* 李航《统计学习方法》
* 课件
* 周志华《机器学习》
* 刘建平博客
