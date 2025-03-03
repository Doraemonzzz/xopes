# Lightning Attention with Decay(Sequential Recurrence)

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\lambda$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= \lambda  \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times e}.
$$



## Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\lambda$。

我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= \lambda  \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times e}.
$$

定义：
$$
\begin{aligned}

\prod_{j=1}^t \lambda  & =\lambda^t = \Lambda_t.
\end{aligned}
$$
展开公式可得：
$$
\begin{aligned}
\mathbf s_t &= \lambda  \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&=  \lambda
\left( \lambda \mathbf s_{t-2} +\mathbf k_{t-1} \mathbf v_{t-1}^\top  \right)+\mathbf k_t \mathbf v_t^\top  \\
&=\lambda  \lambda  \mathbf s_{t-2}  + \lambda  \mathbf k_{t-1} \mathbf v_{t-1}^\top
+ \mathbf k_t \mathbf v_t^\top \\
&=  \ldots \\
&= \Lambda_t \mathbf s_0  + \sum_{j=1}^t \left(\odot_{i=j+1}^t \lambda_i \right)  \mathbf k_j \mathbf v_j^\top \\
&=\lambda^t \mathbf s_0  + \sum_{j=1}^t \lambda^{t-j} \mathbf k_j\mathbf v_j ^\top.
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf q_t^\top
\left[\lambda^{t-j}  \mathbf k_j\mathbf v_j ^\top  \right]

&= \lambda^{t-j}\mathbf q_t^\top \mathbf k_j\mathbf v_j^\top .
\end{aligned}
$$
所以：
$$
\begin{aligned}
\mathbf o_t^\top
&=\lambda^t   \mathbf q_t^\top \mathbf s_0 +
\sum_{j=1}^t \lambda^{t-j}\mathbf q_t^\top \mathbf k_j\mathbf v_j^\top   ,\\


\mathbf s_t
&=\lambda^t    \mathbf s_0  + \sum_{j=1}^t\lambda^{t-j} \mathbf k_j\mathbf v_j ^\top.


\end{aligned}
$$



## Backward

### $\mathbf{dq}_n, \mathbf{dk}_n,\mathbf {dv}_n$

$$
\begin{aligned}
\mathbf {ds}_{n+1} &= \mathbf {ds}_n ,  \\
\mathbf {ds}_n  &= \mathbf {ds}_{n+1} + \mathbf{q}_n\mathbf {do}^\top_n, \\

\mathbf {ds}_t &= \lambda\mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {ds}_0&= \lambda \mathbf {ds}_1,  \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dv}_t& = \mathbf k_t^\top \mathbf {ds}_t.
\end{aligned}
$$

利用：
$$
\begin{aligned}
\mathbf O,  {\mathbf s}_n &= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \Gamma, \mathbf s, \mathrm{false}), \\

\mathbf {dQ}, {\mathbf s}_n &= f(\mathbf {dO}, \mathbf V, \mathbf K, \Gamma, \Lambda, \mathbf s, \mathrm{false}), \\

\mathbf {dK},  {\mathbf {ds}_0} &= f(\mathbf {V}, \mathbf {dO}, \mathbf Q, \Gamma, \Lambda, \mathbf {ds}, \mathrm{true}), \\

\mathbf {dV},  {\mathbf {ds}_0} &= f(\mathbf {K}, \mathbf Q,\mathbf {dO} , \Lambda, \Gamma, \mathbf {ds}, \mathrm{true}).
\end{aligned}
$$
