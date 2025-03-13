# LogCumSumExp

For input $\mathbf x\in \mathbb R^n$ and initial state $x_0$, the output of LogCumSumExp is:
$$
\begin{aligned}
 o_i & = \log \sum_{j=1}^{i} \exp(x_j).
\end{aligned}
$$
This can be understood as a composition of three functions:
$$
\begin{aligned}
y_j &= \exp(x_j), \\
\mathbf z &= \mathrm{cumsum}(\mathbf y), \\
\mathbf o &= \log \mathbf z, \\
j &= 0,\ldots, n.
\end{aligned}
$$
So we can follow the implementation approach of cumsum and logsumexp.

## Forward

For input $\mathbf x\in \mathbb R^n$, compute:
$$
\begin{aligned}
 o_i & = \log \sum_{j=1}^{i} \exp(x_j).
\end{aligned}
$$
Note that:
$$
\begin{aligned}
o_i
&= \log \sum_{j=1}^{i} \exp(x_j) \\
&= \log \sum_{j=1}^{i-1} \exp(x_j) +  \exp(x_i) \\
&= \log \left(\exp(o_{i-1})+  \exp(x_i) \right).
\end{aligned}
$$
Note that the above formula is unstable, so we introduce:
$$
m_i=\max_{j\le i}\{x_j\}.
$$
Then:
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
Note that the following equation holds:
$$
\begin{aligned}
\bar o_i &= \log \left(\exp(\bar o_{i-1}+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right).
\end{aligned}
$$
Therefore, we maintain the maximum value of each block:
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
For $Y_{i, k}= \log\left(\sum_{j=1}^k\exp(x_{(i-1)B+j}- M_{i}) \right)$, we can use prefix sum.

## Backward

Based on:
$$
\begin{aligned}
y_j &= \exp(x_j), \\
\mathbf z &= \mathrm{cumsum}(\mathbf y), \\
\mathbf o &= \log \mathbf z.
\end{aligned}
$$
We can derive:
$$
\begin{aligned}
\mathbf {dz} &= \frac{\partial l}{\partial \mathbf z} \\
&= \frac{\partial l}{\partial \mathbf o}\odot \frac{\partial \mathbf o}{\partial \mathbf z} \\
&= \mathbf{do} /\mathbf z, \\
 {dy}_i&= \sum_{j=i}^n dz_j, \\
 \mathbf {dx} &= \mathbf {dy}\odot \mathbf y.
\end{aligned}
$$

So we can use the cumsum approach for acceleration.
