# Chunk Rnn

参考TTT的论文，我们知道Linear Attention的State $\mathbf s_t$更新对应如下优化问题：
$$
\mathbf s_{t+1} = \min_{\mathbf s_t} \mathcal L(\mathbf s_t^\top \mathbf k_t, \mathbf v_t)
\triangleq  \min_{\mathbf S_t} \mathcal L( \mathbf u_t, \mathbf v_t).
$$
因此：
$$
\mathbf s_{t+1}= \mathbf s_t - \eta_t
\mathbf k_t
\frac{\partial  \mathcal L( \mathbf u_t, \mathbf v_t)}
{\partial \mathbf u_t}^\top \triangleq
\mathbf s_t - \eta_t
\mathbf k_t
\mathbf p_t^\top.
$$
展开上式可得：
$$
\mathbf s_{t}=
\mathbf s_0 - \sum_{j=1}^t \eta_j
\mathbf k_j
\mathbf p_j^\top.
$$
其中$\mathbf s_t \in \mathbb R^{d\times e}, \mathbf q_t , \mathbf k_t \in \mathbb R^d, \mathbf v_t\in \mathbb R^{d},\eta_t \in \mathbb R^+$。注意到$\eta_t$可以被吸收，为了后续方便讨论，我们记：
$$
\begin{aligned}
\mathbf s_{t+1} &=
\mathbf s_t +
\mathbf k_t
\mathbf p_t^\top , \\

\mathbf s_{t} &=
\mathbf s_0 + \sum_{j=1}^t
\mathbf k_j
\mathbf p_j^\top.
\end{aligned}
$$



## Batch/Chunk

上式是间隔1步的递推，可以理解为sgd；另一种则是间隔多步的递推，即mini-batch-sgd，为了避免和原本的batch混淆，我们使用chunk代替batch的概念，并且假设$\mathbf X_t$为第$t$个chunk的数据，$\mathbf S_t= \mathbf s_{tc}$。

此时：
$$
\begin{aligned}
\mathbf S_{t}
& = \mathbf s_{tc} \\
&= \mathbf s_{tc-1} + \mathbf k_{tc-1} \mathbf v_{tc-1}^\top . \\
&= \ldots. \\
&= \mathbf s_{(t-1)c}  + \sum_{j=1}^{c}

\mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top \\
&= \mathbf S_{t-1}   + \mathbf K_t^\top \mathbf P_t.

\end{aligned}
$$
注意到：
$$
\mathbf o_t^\top  = \mathbf q_t^\top \mathbf s_t.
$$
那么：
$$
\begin{aligned}
\mathbf o_{(t-1)c+i}^\top
&= \mathbf q_{(t-1)c+i}^\top \mathbf s_{(t-1)c+i }\\

&=
\mathbf q_{(t-1)c+i}^\top \mathbf s_{(t-1)} + \sum_{j=1}^i  \mathbf q_{(t-1)c+i}^\top
\mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top  \\

&= \mathbf q_{(t-1)c+i}^\top \mathbf S_{t-1} +   \sum_{j=1}^i  \mathbf q_{(t-1)c+i}^\top
\mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top
,  \\

\mathbf O_t
&= \mathbf Q_t \mathbf S_{t-1} + \mathrm{tril}(\mathbf Q_t \mathbf K_t^\top) \mathbf P_t.

\end{aligned}
$$



## Decay/Momentum

我们可以引入decay/momentum：
$$
\mathbf s_{t+1} =
\lambda_t \mathbf s_t +
\mathbf k_t
\mathbf p_t^\top.
$$
那么：
$$
\begin{aligned}
\mathbf s_{t} &=
\gamma_t \mathbf s_0 + \sum_{j=1}^t \gamma_i /\gamma_j
\mathbf k_j
\mathbf p_j^\top,  \\
\gamma_j &= \prod_{i=1}^j \lambda_i.

\end{aligned}
$$
此时：
$$
\begin{aligned}
\mathbf S_{t}
& = \mathbf s_{tc} \\
&=  \lambda_{tc}  \mathbf s_{tc-1} + \mathbf k_{tc-1} \mathbf v_{tc-1}^\top . \\
&= \ldots. \\
&=  \gamma_{tc}/\gamma_{(t-1)c} \mathbf s_{(t-1)c}  + \sum_{j=1}^{c}
\gamma_{tc}/\gamma_{(t-1)c+j}
\mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top \\
&=  \gamma_{tc}/\gamma_{(t-1)c} \mathbf S_{t-1}   +  [\Theta_t\odot \mathbf K_t]^\top \mathbf P_t, \\

\mathbf \Theta_{t,j} &=\gamma_{tc}/\gamma_{(t-1)c+j}, \\
\Gamma_{t,j} &= \gamma_{(t-1)c +j}/\gamma_{(t-1)c}.

\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf o_{(t-1)c+i}^\top
&= \mathbf q_{(t-1)c+i}^\top \mathbf s_{(t-1)c+i }\\

&=
\mathbf q_{(t-1)c+i}^\top  \gamma_{(t-1)c+i}/\gamma_{(t-1)c} \mathbf s_{(t-1)} + \sum_{j=1}^i  \mathbf q_{(t-1)c+i}^\top
\gamma_{(t-1)c+i}/\gamma_{(t-1)c+j}
\mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top  \\

&= \mathbf q_{(t-1)c+i}^\top \gamma_{(t-1)c +j}/\gamma_{(t-1)c}\mathbf S_{t-1} +   \sum_{j=1}^i  \mathbf q_{(t-1)c+i}^\top
\gamma_{(t-1)c+i}/\gamma_{(t-1)c+j} \mathbf k_{(t-1)c+j}
\mathbf p_{(t-1)c+j}^\top
,  \\

\mathbf O_t
&= \mathbf {\bar Q}_t \mathbf S_{t-1} + \mathrm{tril}(\mathbf {\bar Q}_t \mathbf {\bar K}_t^\top) \mathbf P_t.

\end{aligned}
$$
注意到上式还是有点复杂的，作为一个简化的版本，我们考虑如下递推：
$$
\begin{aligned}
\mathbf S_t &= \mathrm{diag}(\mathbf F_t)) \mathbf S_{t-1} + \mathbf K_t ^\top \mathbf P_t, \\

\mathbf O_t
&= \mathbf Q_t \mathbf S_{t-1} + \mathrm{tril}(\mathbf Q_t \mathbf K_t^\top) \mathbf P_t .
\end{aligned}
$$
其中$\mathbf F_t = f(\mathbf X_t), f:\mathbb R^{c\times d}\to \mathbb R^d$。注意到结合这个公式，我们不难可以将上述操作推广为：
$$
\begin{aligned}
\mathbf S_t &= \mathrm{diag}(\mathbf F_t)) \mathbf S_{t-1} + \mathbf K_t ^\top \mathbf P_t, \\

\mathbf O_t
&= \mathbf Q_t \mathbf S_{t-1} + g(\mathrm{tril}(\mathbf Q_t \mathbf K_t^\top)) \mathbf P_t .
\end{aligned}
$$
其中$g$是激活函数，例如Relu和Softmax等等，在后续讨论中，我们沿用此公式进行讨论，注意到因为我们实际使用时不需要原本的loss，所以我们设计网络时直接使用上式即可。



## 实例化

从上面的公式不难看出，我们通过选择不同的$\mathbf P_t$可以得到不同的递推更新。

### Linear Attention

$$
\mathbf P_t=\mathbf V_t.
$$



### TTT

$$
\mathbf P_t=\mathbf V_t- \mathrm{normalize}(\mathbf K_t \mathbf S_{t-1}) + \mathbf K_t .
$$



### DeltaNet

$$
\begin{aligned}
\mathbf P_t &=\mathbf U_t - \mathbf W_t \mathbf S_t\\
&= \mathbf T_t (\mathbf V_t - \mathbf K_t \mathbf S_t), \\
\mathbf W_t &= \mathbf T_t  \mathbf K_t, \\
\mathbf U_t &= \mathbf T_t  \mathbf V_t, \\
\mathbf T_t &= \mathbf (\mathbf I + \mathrm{tril}(\mathrm{diag}(\beta_t)\mathbf K_t \mathbf K_t^\top, -1))^{-1}
\mathrm{diag}(\beta_t).

\end{aligned}
$$



### TTT MLP

我们可以假设：
$$
\mathbf P_t = \nabla_{\mathbf u_t}\frac 1 2 \| f(\mathbf u_t)  - \mathbf v_t\|^2.
$$
其中$f$是某个网络。



## 高效实现

后续我们讨论下式：
$$
\begin{aligned}
\mathbf S_t &= \mathrm{diag}(\mathbf F_t) \mathbf S_{t-1} + \mathbf K_t ^\top \mathbf P_t, \\

\mathbf O_t
&= \mathbf Q_t \mathbf S_{t-1} + g(\mathrm{tril}(\mathbf Q_t \mathbf K_t^\top)) \mathbf P_t .
\end{aligned}
$$
将第一个公式更新为：
$$
\begin{aligned}
\mathbf {\bar S}_t &= \mathbf K_t ^\top \mathbf P_t, \\
\mathbf S_t &= \mathrm{diag}(\mathbf F_t) \mathbf S_{t-1} +\mathbf {\bar S}_t.

\end{aligned}
$$



### Fwd

如果$\mathbf P_t$不依赖于$\mathbf S_t$，此时$\mathbf P_t$可以提前算出来，此时用如下算法即可。

并行计算：
$$
\mathbf {\bar S}_t =\mathbf K_t ^\top \mathbf P_t,  g(\mathrm{tril}(\mathbf Q_t \mathbf K_t^\top)) \mathbf P_t.
$$
执行元素递归：
$$
\mathbf S_t = \mathrm{diag}(\mathbf F_t) \mathbf S_{t-1} +\mathbf {\bar S}_t.
$$

例如可以取：
$$
\mathbf P_t = \mathbf V_t -\mathbf K_t-\mathbf K_t \mathbf S_0.
$$
如果$\mathbf P_t$依赖于$\mathbf S_t$，不妨假设：
$$
\mathbf P_t = \mathbf V_t -\mathbf K_t - \mathbf K_t \mathbf S_{t-1}.
$$


Todo:

- test v;
- test v - k;
- test v - k - k s0;
- test v - kst
- test v - k - k st
- test v - k - normalize(k s);
- test no normalize;
-





另一个方案，对于$\mathbf k_t $过一个element wise的递推即可，因为$ \mathbf K_t \mathbf S_{t-1}$本质上是$f(\mathbf K_t, \mathbf K_1 ,\ldots \mathbf K_{t-1})$。
