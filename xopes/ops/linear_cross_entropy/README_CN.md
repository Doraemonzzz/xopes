# Linear Cross Entropy

对于输入$\mathbf x\in \mathbb R^d, \mathbf W\in \mathbb R^{v\times d}, \mathbf b \in \mathbb R^v$，以及one hot label $\mathbf y\in \mathbb R^v,y_k =1$，smooth参数$\lambda\in[0, 1]$，其中：
$$
\mathbf {\bar y}= (1-\lambda) \mathbf y+\lambda /e  \mathbf 1,\\
\mathbf 1^\top \mathbf {\bar y}=(1-\lambda)  + \lambda=1.
$$
输出：
$$
\begin{aligned}
\mathbf z &=\mathbf W\mathbf x + \mathbb b,  \\
r&=\log\left(
\sum_{j=1}^v \exp(z_j)
\right)   \\


o &= - \sum_{i=1}^v \bar y_i \left(z_i -r \right) \\
&= - \sum_{i=1}^v ( (1-\lambda) y_i+\lambda /v  ) \left(z_i - r\right) \\
&= - \sum_{i=1}^v (1-\lambda) y_i\left(z_i - r\right)
-\lambda/v  \sum_{i=1}^e \left(z_i -r\right)
\\

&=- (1-\lambda) \left(z_k- r\right) -\lambda/v \left( \sum_{i=1}^v z_i -v r \right)\\
&= - (1-\lambda)z_k + (1-\lambda) r - \lambda/v \left( \sum_{i=1}^v z_i\right) + \lambda r \\
&= - (1-\lambda)z_k + r - \lambda/v \left( \sum_{i=1}^v z_i\right).
\end{aligned}
$$


## Forward

- 记$m=0,  {sse}=0, s=0, z=0$；
- $\mathbf z= \mathbf W\mathbf x +\mathbb b\in \mathbb R^v$；
- $m =\max(\mathbf z)$；
- ${sse}= \sum_{j=1}^v \exp(y_{i,j}-{m})$；
- $s=\sum_{j=1}^e z_{ij}$；
- $z=z_{k}$；
- $c=y_k\neq ig$；
  - for reduce；
- return $-(1-\lambda) z+m+\log sse-\lambda/v s$；



## Backward

输入：$\mathbf {do}\in \mathbb R$。

计算：
$$
\begin{aligned}
p_{k}&= \exp(z_k - r) , \\
\frac{\partial o}{\partial z_i}
&= -(1-\lambda)\frac{\partial z_k}{\partial z_i} + \frac{\partial r}{\partial z_i}-
\lambda/v \frac{\partial \left( \sum_{i=1}^v z_i \right)}{\partial z_i}\\
&=  -(1-\lambda) 1_{i=k}+p_i -\lambda /v, \\
\frac{\partial \mathbf o}{\partial \mathbf  z}& =  -(1-\lambda) \mathbf y + \mathbf p - \lambda/ v,     \\
\mathbf {dz}&= {\mathbf {do}}\frac{\partial \mathbf o}{\partial \mathbf  z}\in \mathbb R^{v}, \\
\mathbf {dx}&= \mathbf {dz} \mathbf W^\top, \\

\mathbf {dW}&=(\mathbf {dz}) \mathbf {x}^\top , \\

\mathbf{db}&= \mathbf {dz}.
\end{aligned}
$$

对于$\mathbf {dz}$：
$$
\begin{aligned}
\mathbf {dz}
& =  \mathbf{do}\left( -(1-\lambda) \mathbf y_{\mathrm{one-hot}} + \mathbf p - \lambda/ e  \right).
\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {dx}&= \mathbf {dz} \mathbf W^\top, \\
\mathbf {dW}&=  (\mathbf{dz})\mathbf x^\top,
\\
\mathbf{db}&= \mathbf {dz}.
\end{aligned}
$$
算法：

- $\mathbf p= \exp(\mathbf z-\mathrm{lse})$；
- $
  \mathbf {dz}  =  \mathbf{do}\left( -(1-\lambda) \mathbf y_{\mathrm{one-hot}} + \mathbf p - \lambda/ v \right)$；
- $\mathbf {dx}= \mathbf {dz} \mathbf W^\top$；
- $\mathbf {dW}=  (\mathbf{dz})\mathbf x^\top$；
- $\mathbf{db}= \mathbf {dz}$；
- return $\mathbf{dx}, \mathbf {dW}, \mathbf {db}$；



## 方案：划分词表（探索版本，弃用）

### Forward

输入：$\mathbf x\in \mathbb R^d, \mathbf W\in \mathbb R^{v\times d}, \mathbf b\in  \mathbb R^v$，以及one hot label $\mathbf y\in \mathbb R^v,y_t =1$，smooth参数$\lambda\in[0, 1]$。

#### 递推版本

假设：
$$
\begin{aligned}
\mathbf W
&=
\left[\begin{matrix}\mathbf W_1 \\
\vdots \\
\mathbf W_k
\end{matrix}\right]
\in \mathbb R^{ v \times d}, \\
\mathbf b &=
\left[\begin{matrix}
\mathbf b_1 \\
\vdots\\
\mathbf b_k
\end{matrix}\right]\in \mathbb R^{ v },\\
\mathbf y &=
\left[\begin{matrix}
\mathbf y_1 \\
\vdots\\
\mathbf y_k
\end{matrix}\right]\in \mathbb R^{ v }.
\end{aligned}
$$

- 记$m=0,  {sse}=0, s=0, z=0$；
- for $i=1,\ldots ,k$：
  - $\mathbf z_i =\mathbf W_i \mathbf x  \in \mathbb R^{v/k}$；
  - $m_i =\max(\mathbf z_i)$；
  - ${m}'=\max({m_i},  m)$；
  - ${sse}_i= \sum_{j=1}^{v/k} \exp(y_{i,j}-{m}')$；
  - ${sse}= \exp(m-m') sse +  sse_i$；
  - $m=m'$；
  - $s_i=\sum_{j=1}^{v/k} z_{ij}$；
  - $s=s+s_i$；
  - if $t\in [iv/k+1, (i+1)v/k]$, then
    - $z=z_{i, t-iv/k}$；
- return $-(1-\lambda) z+m+\log sse-\lambda/v s$；



#### 并行版本

- 记$m=0,  {sse}=0, s=0, z=0$；
- for $i=1,\ldots ,k$，并行计算出：
  - $\mathbf z_i =  \mathbf W_i\mathbf x \in \mathbb R^{v/k}$；
  - $m_i =\max(\mathbf z_i)$；
  - ${sse}_i= \sum_{j=1}^{v/k} \exp(y_{i,j}-{m}')$；
  - $s_i=\sum_{j=1}^{v/k} z_{ij}$；
  - if $t\in [iv/k+1, (i+1)v/k]$, then
    - $z=z_{i, t-iv/k}$；
- for $i=1,\ldots, k$：
  - ${m}'=\max({m_i},  m)$；
  - ${sse}= \exp(m-m') sse + \exp(m_i-m') sse_i$；
  - $m=m'$；
  - $s=s+s_i$；
- return $-(1-\lambda) z+m+\log sse-\lambda/v s$；



### Backward

输入：$\mathbf {do}\in \mathbb R$。

计算：
$$
\begin{aligned}
p_{k}&= \exp(z_k - r) , \\
\frac{\partial o}{\partial z_i}
&= -(1-\lambda)\frac{\partial z_k}{\partial z_i} + \frac{\partial r}{\partial z_i}-
\lambda/v \frac{\partial \left( \sum_{i=1}^v z_i \right)}{\partial z_i}\\
&=  -(1-\lambda) 1_{i=k}+p_i -\lambda /v, \\
\frac{\partial \mathbf o}{\partial \mathbf  z}& =  -(1-\lambda) \mathbf y + \mathbf p - \lambda/ v,     \\
\mathbf {dz}&= {\mathbf {do}}\frac{\partial \mathbf o}{\partial \mathbf  z}\in \mathbb R^{v}, \\
\mathbf {dx}&= \mathbf {dz} \mathbf W^\top, \\

\mathbf {dW}&=(\mathbf {dz}) \mathbf {x}^\top , \\

\mathbf{db}&= \mathbf {dz}.
\end{aligned}
$$

对于$\mathbf {dz}$：
$$
\begin{aligned}
\mathbf {dz}
& =  \mathbf{do}\left( -(1-\lambda) \mathbf y_{\mathrm{one-hot}} + \mathbf p - \lambda/ v  \right) \\
& \triangleq  \mathbf {dz}_1+\mathbf {dz}_2,\\
 \mathbf {dz}_1& = \mathbf{do}\left( -(1-\lambda) \mathbf y_{\mathrm{one-hot}}- \lambda/ v \right), \\
 \mathbf {dz}_2& = \mathbf{do}\odot \mathbf p. \\
\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {dx}&= \mathbf {dz} \mathbf W^\top \\
&= \mathbf{dz}_1 \mathbf W^\top + \mathbf {dz}_2 \mathbf W^\top , \\
&= \mathbf{do}\left( -(1-\lambda) \mathbf y_{\mathrm{one-hot}}\mathbf W^\top - \lambda/ v\mathbf W^\top + \mathbf p \mathbf W^\top  \right)    \\
\mathbf {dW}&=  (\mathbf{dz})\mathbf x^\top  \\
&=(\mathbf{dz}_1) \mathbf x^\top   + (\mathbf {dz}_2) \mathbf x^\top  , \\
& = \mathbf {do} \left(
 -(1-\lambda)\mathbf y_{\mathrm{one-hot}}  \mathbf x^\top- \lambda / v \mathbf x^\top
 + \mathbf p\mathbf x^\top
\right)

\\
\mathbf{db}&= \mathbf {dz}.
\end{aligned}
$$
算法：

test

对于$\mathbf {dz}_1$，不需要使用递推即可提前计算。

对于$\mathbf {dz_2}$（为了方便讨论，我们先忽略标量$\mathbf {do}$），注意到有如下关系（考虑chunk递归），第$j$个chunk计算得到的lse为$\mathrm{lse}_j$，累加到第$j$个chunk的lse为$\mathrm{lse}^j$，那么：
$$
\begin{aligned}
\exp(\mathrm{lse}^j) & =\exp(\mathrm{lse}^{j-1}) + \exp(\mathrm{lse}_j),  \\
\lambda_j &= \exp(\mathrm{lse}^{j-1}) /\exp(\mathrm{lse}^j) \\
&= \exp(\mathrm{lse}^{j-1}) /\left( \exp(\mathrm{lse}^{j-1}) + \exp(\mathrm{lse}_j) \right).
\end{aligned}
$$
补充：
$$
\begin{aligned}
\prod_{j=1}^s \lambda_j
&= \prod_{j=1}^s \exp(\mathrm{lse}^{j-1}) /\exp(\mathrm{lse}^j) \\
&=  \exp(\mathrm{lse}^{0})/\exp(\mathrm{lse}^s)\\
&= \exp(-\mathrm{lse}^s).
\end{aligned}
$$


第$j$个chunk内的概率为$\mathbf p_j\in \mathbb R^{v/k}$，累加到全局的概率为$\mathbf p_i^j\in \mathbb R^{v/k},i=1,\ldots, k-1$，那么：
$$
\begin{aligned}
\mathbf p^{j}_i &=\exp(\mathbf z_i)/\exp(\mathrm{lse}^j)\\
 &= \exp(\mathbf z_i)/ \exp(\mathrm{lse}^{j-1}) \times \exp(\mathrm{lse}^{j-1}) /\exp(\mathrm{lse}^j)\\
 &= \lambda_j  \mathbf p^{j-1}_i, j > i,\\

 \mathbf p_{j} &= \exp(\mathbf z_j)/\exp(\mathrm{lse}_j),\\
  \mathbf p^{j}_j&= \exp(\mathbf z_j)/\exp(\mathrm{lse}^j)  \\
 &=\exp(\mathbf z_j)/\exp(\mathrm{lse}_j)\times\exp(\mathrm{lse}_j) /\exp(\mathrm{lse}^j)\\
 &=  \mathbf p_{j}(1-\lambda_j).

 \end{aligned}
$$
假设循环到第$j$个chunk时的$\mathbf {dz}_2\mathbf W^\top$为$\mathbf {du}^j$，那么：
$$
\begin{aligned}
\mathbf {du}^j
&= \lambda_j \mathbf {du}^{j-1} + (1-\lambda_j)\mathbf p_{j} \mathbf W_j^\top \\
&\triangleq \lambda_j \mathbf {du}^{j-1} + (1-\lambda_j)\mathbf {du}_j.
\end{aligned}
$$
证明：
$$
\begin{aligned}
\mathbf {du}^j
& =[\mathbf p^j_1, \ldots \mathbf p_{j}^j][\mathbf W_1, \ldots, \mathbf W_j]^\top\\
&= \sum_{i=1}^j \mathbf p^j_i \mathbf W_i^\top \\
&= \sum_{i=1}^{j-1}\mathbf p^{j}_i \mathbf W_i^\top  +\mathbf p^j_j \mathbf W_j^\top\\
&= \sum_{i=1}^{j-1}\lambda_j \mathbf p^{j-1}_i \mathbf W_i^\top  +\mathbf p^j_j \mathbf W_j^\top\\
&= \lambda_j\mathbf {du}^{j-1} + (1-\lambda_j)  \mathbf p_{j} \mathbf W_j^\top\\
&=  \lambda_j \mathbf {du}^{j-1} + (1-\lambda_j)\mathbf p_{j} \mathbf W_j^\top.
\end{aligned}
$$
假设循环到第$j$个chunk时的$\mathbf x^\top  \mathbf {dz}_2$为$\mathbf {dv}^j\in \mathbb R^{d\times {je}}$，那么：
$$
\begin{aligned}
\mathbf {dv}^j
& = [\lambda_j \mathbf {dv}^{j-1}, (1-\lambda_j) \mathbf x^\top  \mathbf p_j]\\
& = [\lambda_j \mathbf {dv}^{j-1}, (1-\lambda_j)\mathbf {dv}_j].

\end{aligned}
$$
证明：
$$
\begin{aligned}
\mathbf {dv}^j_i& = \mathbf x^\top \mathbf p_i^j \\
&=  \mathbf x^\top \lambda_j \mathbf p_i^{j-1} \\
&=  \lambda_j\mathbf {dv}^{j-1}_i \\
i&\le j-1,  \\
\mathbf {dv}^j_j &=\mathbf x^\top \mathbf p^j_j \\
&= (1-\lambda_j) \mathbf x^\top \mathbf p_j.
\end{aligned}
$$
展开上式可得：
$$
\begin{aligned}
\mathbf {dv}^k_i
& = \mathbf{dv}_i (1-\lambda_i)\prod_{j=i+1}^k \lambda_j  \\
&=\mathbf{dv}_i (1-\lambda_i)(\exp(-\mathrm{lse}^k)/\exp(-\mathrm{lse}^i))\\
&=\mathbf{dv}_i (1-\lambda_i)(\exp(\mathrm{lse}^i-\mathrm{lse}^k)).
\end{aligned}
$$

注意到实际使用时因为有batch size，所以$\mathbf{dv}$的递推无法使用（每个batch的decay不同）。
