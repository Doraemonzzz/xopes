# Lightning Attention with Naive Linear Recurrence

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times d}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：

$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d}, \\
\mathbf s_i &= \lambda_i \odot \mathbf s_{i-1} + \mathbf k_i\odot \mathbf v_i, \\
\mathbf o_i &= \mathbf q_i \odot \mathbf s_i.
\end{aligned}
$$

返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d}.
$$
