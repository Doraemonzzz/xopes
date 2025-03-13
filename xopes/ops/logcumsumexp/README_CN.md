# LogCumSumExp

对于输入$\mathbf x\in \mathbb R^n$，和初始状态$x_0$，LogCumSumExp的输出为：
$$
\begin{aligned}
 o_i & = \log \sum_{j=1}^{i} \exp(x_j).
\end{aligned}
$$
这可以理解为三个函数的复合：
$$
\begin{aligned}
y_j &= \exp(x_j), \\
\mathbf z &= \mathrm{cumsum}(\mathbf y), \\
\mathbf o &= \log \mathbf z, \\
j &= 0,\ldots, n.

\end{aligned}
$$
所以可以沿用cumsum以及logsumexp的思路进行实现。



## Forward

对于输入$\mathbf x\in \mathbb R^n$计算：
$$
\begin{aligned}
 o_i & = \log \sum_{j=1}^{i} \exp(x_j).
\end{aligned}
$$
注意到：
$$
\begin{aligned}
o_i
&= \log \sum_{j=1}^{i} \exp(x_j) \\
&= \log \sum_{j=1}^{i-1} \exp(x_j) +  \exp(x_i) \\
&= \log \left(\exp(o_{i-1})+  \exp(x_i) \right).
\end{aligned}
$$
注意到上式不稳定，引入：
$$
m_i=\max_{j\le i}\{x_j\}.
$$
那么：
$$
\begin{aligned}
o_i
&= \log \left(\exp(o_{i-1}-m_{i})+  \exp(x_i-m_i) \right) + m_i, \\
o_i-m_i &= \log \left(\exp(o_{i-1}-m_{i-1}+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right) \\
 &= \log \left(\exp(\bar o_{i-1}+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right) \\
 &= \bar o_i, \\
o_i & = \bar o_i + m_i.
\end{aligned}
$$
注意下式成立，
$$
\begin{aligned}
\bar o_i &= \log \left(\exp(\bar o_{i-1}+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right).
\end{aligned}
$$
故我们维护每个block的最大值：
$$
\begin{aligned}
m&= n/B, \\
M_0&=  -\infty, \\
M_{i}  & =\max\{ M_{i-1}, x_{(i-1)B+k},k=1,\ldots, d\}, \\
\bar O_i &= [\bar o_{(i-1)B+1},\ldots, \bar o_{(i-1)B+B}], \\
X_i&=[x_{(i-1)B+1},\ldots,  x_{(i-1)B+B}],  \\
Y_{i, k}&= \log\left(\sum_{j=1}^k\exp(x_{(i-1)B+j}- M_{i}) \right),   \\
\bar O_i&= \log \left(\exp(\bar O_{i-1, B}+M_{i-1}-M_{i})+  \exp(Y_i) \right),\\
O_i&= \bar O_i + M_i.

\end{aligned}
$$
对于$Y_{i, k}= \log\left(\sum_{j=1}^k\exp(x_{(i-1)B+j}- M_{i}) \right)$使用前缀和即可。



## Backward

根据：
$$
\begin{aligned}
y_j &= \exp(x_j), \\
\mathbf z &= \mathrm{cumsum}(\mathbf y), \\
\mathbf o &= \log \mathbf z.

\end{aligned}
$$
可以得出：
$$
\begin{aligned}
\mathbf {dz} &= \frac{\partial l}{\partial \mathbf z} \\
&= \frac{\partial l}{\partial \mathbf o}\odot \frac{\partial \mathbf o}{\partial \mathbf z} \\
&= \mathbf{do} /\mathbf z, \\
 {dy}_i&= \sum_{j=i}^n dz_j, \\
 \mathbf {dx} &= \mathbf {dy}\odot \mathbf y.

\end{aligned}
$$

所以可以使用cumsum的思路进行加速。
