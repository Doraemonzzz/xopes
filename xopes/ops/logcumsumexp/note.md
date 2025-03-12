# LogCumSumExp

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

反向：
$$
\begin{aligned}
dx_i
& = \frac{\partial l}{\partial x_i}  \\
& = \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \frac{\partial o_j}{\partial x_i}  \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \left[ \frac{\exp(x_i)}
{\sum_{k=1}^{j} \exp(x_k)} \right] \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \left[ \frac{\exp(x_i)}
{\sum_{k=1}^{j} \exp(x_k)} \right] \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j).
\end{aligned}
$$

注意到：
$$
\begin{aligned}
dx_i &=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j) \\
&= \frac{\partial l}{\partial o_i} \times \exp(x_i- o_i) + \sum_{j= i+1}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j)  \\
&= \frac{\partial l}{\partial o_i} \times \exp(x_i- o_i) + \exp(x_i- x_{j+1})\sum_{j= i+1}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_{j+1}- o_j).
\end{aligned}
$$

所以我们给出如下算法：

反向：
$$
\begin{aligned}
p_i&=\exp(x_i-o_i), \\
\lambda_{i+1} &=\exp(x_i -x_{i+1}), \\
u_i &\triangleq dx_i, \\
v_i &\triangleq p_i\frac{\partial l}{\partial o_i} , \\
u_n&= v_n, \\
u_i&=  \lambda_{i+1} u_{i+1}+ v_i .

\end{aligned}
$$

所以后续参考Linear Attention/RNN的思路即可。







递推上式可得：
$$
\begin{aligned}
u_i
&=  \lambda_{i+1} u_{i+1}+ v_i \\
&= \lambda_{i+1}(\lambda_{i+2} u_{i+2} + v_{i+1}) + v_i \\
&= \lambda_{i+1}\lambda_{i+2} u_{i+2} +  \lambda_{i+1}v_{i+1} + v_i \\
&= \lambda_{i+1}\lambda_{i+2} u_{i+2} +  \lambda_{i+1}v_{i+1} + v_i \\
&= \sum_{j=i}^n \left(\prod_{k=i+1}^j \lambda_k  \right)v_j, \\

\prod_{k=i+1}^j \lambda_k
&= \prod_{k=i+1}^j  \exp(x_{k-1} -x_{k}) \\
&=\exp(x_{i} -x_{j}), \\

u_i &= \sum_{j=i}^n\exp(x_{i} -x_{j}) v_j \\
&=  \sum_{j=i}^n\exp(x_{i} -x_{j})  p_j\frac{\partial l}{\partial o_j} \\
&= \sum_{j=i}^n\exp(x_{i} -x_{j})  \exp(x_j - o_j)\frac{\partial l}{\partial o_j} \\
&= \sum_{j=i}^n\exp(x_{i}  - o_j)\frac{\partial l}{\partial o_j}.
\end{aligned}
$$




注意到：
$$
\begin{aligned}
{dx_i}\times {\exp(-x_i)}
&=   \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(- o_j)\\

&= \sum_{j= s}^{n}\frac{\partial l}{\partial o_j} \times \exp(- o_j) + \sum_{j=i}^{s-1}\frac{\partial l}{\partial o_j} \times \exp(- o_j) \\

&= {dx_s}\times {\exp(-x_s)} + \sum_{j=i}^{s-1}\frac{\partial l}{\partial o_j} \times \exp(- o_j) \\

{dx_i} &= {dx_s}\times {\exp(x_i -x_s)} + \sum_{j=i}^{s-1}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j).
\end{aligned}
$$


考虑一般的Linear RNN:
$$
\begin{aligned}
s_t &= \lambda_t \odot s_{t-1} + k_t \odot v_t, \\
o_t &= q_t\odot  s_t, \\
t&=1,\ldots, n.

\end{aligned}
$$
反向计算为：
$$
\begin{aligned}
dq_t &= do_t \odot s_t,   \\
ds_{n+1} &= do_n \odot  q_n, \\
ds_{n} &=ds_{n+1}, \\
ds_{t} &= \frac{\partial l}{\partial s_{t+1}} \frac{\partial s_{t+1}}{\partial s_{t}} + do_t \odot q_t \\
&= \lambda_{t+1} ds_{t+1}  + do_t \odot q_t,  \\
ds_0 &= \lambda_1 ds_1, \\
dk_t &= ds_{t} \odot v_t, \\
dv_t &= ds_t \odot k_t, \\
t&= 1, \ldots, {n-1}.
\end{aligned}
$$
假设：
