# Out product linear recurrence

给定输入$\mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf o & = \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf o_i &= \mathrm{diag}(\lambda_i) \mathbf o_{i-1} + \mathbf k_i \mathbf v_i^\top.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d\times e}.
$$


## Forward

输入：$\mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，注意如果Decay为空，我们使用$\Lambda=1-\mathbf K$（我们默认$0\le \mathbf K \le 1$）。

计算：
$$
\begin{aligned}
\mathbf o & = \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf o_i &= \mathrm{diag}(\lambda_i) \mathbf o_{i-1} + \mathbf k_i \mathbf v_i^\top.
\end{aligned}
$$


## Backward

输入：$\mathbf {dO}\in \mathbb R^{n\times d\times e}$。

计算：
$$
\begin{aligned}
\mathbf{dkv}_{n+1} &= \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf{dkv}_{i}&= \mathrm{diag}(\lambda_i)  \mathbf{dkv}_{i+1} + \mathbf{do}_{i}, \\
\mathbf{dk}_i &=\mathbf{dkv}_{i} \mathbf{v}_i, \\
\mathbf{dv}_i &=\mathbf k_i\mathbf{dkv}_{i}^\top  . \\


\end{aligned}
$$
