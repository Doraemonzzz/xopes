# Lightning Attention Positional Encoding

给定输入$\mathbf q\in \mathbb R^{d}, \mathbf k\in \mathbb R^{d}, \mathbf V\in \mathbb R^{n\times e}$, 初起始state $\mathbf s_0$，以及Decay $\lambda$，我们执行如下操作：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= \lambda  \mathbf s_{i-1} + \mathbf k \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q^\top\mathbf s_i \in \mathbb R^{e}.
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
这接近于SSM, 所以可以编码相对位置信息，此函数和Lightning Attention with Scalar Decay的函数非常相似，区别在于此处每个位置的$\mathbf q, \mathbf k$是相同的。
