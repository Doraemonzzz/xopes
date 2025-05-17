# Flash Attention w/wo scalar/constant decay

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，Decay $\log\Lambda\in \mathbb R^{n}$，Scale $\alpha\in \mathbb R^n$，我们计算如下结果：
$$
\begin{aligned}
\mathbf M_{ij} &=
\begin{cases}
\sum_{t=j+1}^i \log\Lambda_t, & i \ge j, \\
-\infty, & i < j.
\end{cases} \\
\mathbf Q &= \mathbf Q / \alpha, \\
\mathbf O &= \mathrm{Softmax}\left(\mathbf Q\mathbf K^\top + \mathbf M \right) \mathbf V.
\end{aligned}
$$

## Forward

假设chunk size为$c$，初始化$\mathbf m_i=-\infty \times  \mathbf 1_c \in \mathbb R^{c}, \mathbf O_i = \mathbf 0\in \mathbb R^{c\times e}$，我们迭代计算：
$$
\begin{aligned}
\mathbf S_i^{j} &=(\mathbf Q_i^\top \mathbf K_j + \mathbf M_{ij}) \in \mathbb R^{c\times c}, \\
\mathbf {m}_i^{j} &= \mathrm{lse}( \mathbf S_i^{j})\in \mathbb R^{c}, \\
\mathbf { O}_i^j &= \exp(\mathbf S_i^{j} - \mathbf m_i^{j})  \mathbf V_j \in \mathbb R^{c\times e}, \\

\mathbf {\bar m}_i &= \mathrm{lse}([\mathbf m_i, \mathbf m_i^j]) \in \mathbb R^{c}, \\

\mathbf p_i &= \exp(\mathbf m_i - \mathbf {\bar m}_i), \\

\mathbf O_i &= \mathbf p_i \odot \mathbf {O}_i + (1-\mathbf p_i) \odot \mathbf O_i^j, \\

\mathbf m_i &= \mathbf {\bar m}_i.
\end{aligned}
$$


## Backward
$$

\begin{aligned}
\mathbf {dV} &= \mathrm{Softmax}([\mathbf Q\mathbf K^\top] \odot \mathbf M )^\top \mathbf {dO}.
\end{aligned}

$$
