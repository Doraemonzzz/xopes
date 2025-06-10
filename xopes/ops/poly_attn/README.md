# Polynomial Attention

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$, 阶数$m$ ,我们计算如下结果：
$$
\begin{aligned}
\mathbf Q &= \mathbf Q , \\
\mathbf S &= \left(1+ \mathbf Q\mathbf K^\top/m \right)^m ,\\
\mathbf D &= \mathbf S  \mathbf 1_n, \\
\mathbf P &= \mathrm{diag}(\mathbf D)^{-1} \mathbf S , \\
\mathbf O &=  \mathbf P\mathbf V.
\end{aligned}
$$
