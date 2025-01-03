# Log Sum Exp

对于输入$\mathbf x\in \mathbb R^d$，计算：
$$
\begin{aligned}
\mathbf{o}
 = \mathrm{lse}(\mathbf x) =\log\left(
\sum_{j=1}^d \exp(x_j)
\right) .
\end{aligned}
$$

补充定义：
$$
\mathrm{se}(\mathbf x)=
\sum_{j=1}^d \exp(x_j).
$$
那么：
$$
\mathbf o= \log\mathrm{se}(\mathbf x).
$$




## Forward

输入：$\mathbf x\in \mathbb R^d$。

我们定义：
$$
\begin{aligned}
f(\mathbf x) &\ge  \max_{i=1}^d \{x_1,\ldots, x_d \}, \\
\mathbf{o}
 & = \mathrm{lse}(\mathbf x) \\
 &= \log\left(
\sum_{j=1}^d \exp(x_j)
\right)  \\
&=\log\left(
\sum_{j=1}^d \exp(x_j -f(\mathbf x))
\right) + f(\mathbf x) \\
&\triangleq \mathrm{slse}(\mathbf x) +  f(\mathbf x), \\
\mathrm{se}(\mathbf x) & = \exp(\mathrm{slse}(\mathbf x) +  f(\mathbf x)) \\
&= \exp(\mathrm{slse}(\mathbf x)) \exp(f(\mathbf x)) \\
&\triangleq \mathrm{sse}(\mathbf x) \exp(f(\mathbf x)),\\
\mathrm{lse}(\mathbf x)
&= \log(\mathrm{sse}(\mathbf x)) + f(\mathbf x).

\end{aligned}
$$
其中`slse`是stable log sum exp的缩写，`sse`是stable sum exp的缩写。

给定$\mathbf x_1 \in \mathbb R^{d_1}, \mathbf x_2 \in \mathbb R^{d_2}, \mathbf x=[\mathbf x_1, \mathbf x_2]\in \mathbb R^{d_1+d_2}=\mathbb R^{d}$，注意到：
$$
\begin{aligned}

\mathbf{lse}(\mathbf x) &=\log\left(
\sum_{j=1}^e \exp(x_j)
\right) \\
&=  \log\left(
\sum_{j=1}^{d_1} \exp(x_j)
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j)
\right) \\
&= \log\left(
\exp(\mathrm{lse}(\mathbf x_1))
+ \exp(\mathrm{lse}(\mathbf x_2))
\right) \\

&= \log\left(
\exp(\mathrm{lse}(\mathbf x_1)-f(\mathbf x))
+ \exp(\mathrm{lse}(\mathbf x_2)-f(\mathbf x))
\right) +f(\mathbf x) \\



&= \log\left(
\exp(\mathrm{slse}(\mathbf x_1)+f(\mathbf x_1)-f(\mathbf x))
+ \exp(\mathrm{slse}(\mathbf x_2)+f(\mathbf x_2)-f(\mathbf x))
\right)+f(\mathbf x) \\

f(\mathbf x)&=\max(f(\mathbf x_1),f(\mathbf x_2)).
\end{aligned}
$$
所以我们可以利用block wise的递推/并行进行前向加速计算，但是注意到在合并block的时候，我们会使用exp, add, log的操作，这样会增加一些计算量，为了优化这点，我们考虑使用$\mathrm{sse}$函数：
$$
\begin{aligned}

\mathbf{sse}(\mathbf x) &=
\sum_{j=1}^e \exp(x_j-f(\mathbf x))
\\
&=
\sum_{j=1}^{d_1} \exp(x_j-f(\mathbf x))
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j-f(\mathbf x)) \\

&=
\sum_{j=1}^{d_1} \exp(x_j-f(\mathbf x_1))\exp(f(\mathbf x_1 )-f(\mathbf x))
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j-f(\mathbf x_2))\exp(f(\mathbf x_2 )-f(\mathbf x)) \\
&= \exp(f(\mathbf x_1 )-f(\mathbf x)) \mathbf{sse}(\mathbf x_1) + \exp(f(\mathbf x_2 )-f(\mathbf x))\mathbf{sse}(\mathbf x_2)  \\


f(\mathbf x)&=\max(f(\mathbf x_1),f(\mathbf x_2)).
\end{aligned}
$$
我们给出如下算法：

假设$\mathbf x= [\mathbf x_1, \ldots, \mathbf x_k]\in \mathbb R^{kd}$。

### 递推版本

- 记$m=0,  {sse}=0$；
- for $i=1,\ldots ,k$：
  - $m_i =\max(\mathbf x_i)$；
  - ${m}'=\max({m_i},  m)$；
  - ${sse}_i= \sum_{j=1}^d \exp(x_{i,j}-{m}')$；
  - ${sse}= \exp(m-m') sse +  sse_i$；
  - $m=m'$；
- return $m, sse$；



### 并行版本

- 记$m=0,  {sse}=0$；
- for $i=1,\ldots ,k$，并行计算出：
  - $m_i =\max(\mathbf x_i)$；
  - ${sse}_i= \sum_{j=1}^d \exp(x_{i,j}-{m}_i)$；
- for $i=1,\ldots, k$：
  - ${m}'=\max({m_i},  m)$；
  - ${sse}= \exp(m-m') sse + \exp(m_i-m') sse_i$；
  - $m=m'$；
- return $m, sse$；



### 稳定性分析

注意到$\exp(m-m')\le 1, \exp(m_i-m')\le 1, \exp(x_{i,j}-m_i)\le 1$，所以每一项计算都是数值稳定的。



## Backward

输入：$\mathbf {do}\in \mathbb R$。

计算：
$$
\begin{aligned}
p_{i}&= \exp(x_i - \mathbf o) ,\\
\frac{\partial o}{\partial x_i}
&= p_i, \\
\mathbf{dx}&= \mathbf{do} \odot \mathbf p. \\

\end{aligned}
$$
