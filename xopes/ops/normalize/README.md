
# Normalize

For an input $\mathbf{x} \in \mathbb{R}^d$, this section discusses various normalization algorithms. These algorithms are defined as:
$$
\mathbf{o} = f(\mathbf{x}) = c \times \frac{\mathbf{x}_1 \odot \mathbf{w}}{\sqrt{\mathbf{x}_1^\top \mathbf{x}_1}} + \mathbf{b}.
$$
where $c \in \mathbb{R}$, $\mathbf{w}, \mathbf{b} \in \mathbb{R}^d$, and $\mathbf{x}_1 = \mathbf{x}$ or $\mathbf{x}_1 = \mathbf{x} - \bar{\mathbf{x}}, \bar{\mathbf{x}} = \left(\sum_{i=1}^d x_i\right)/d$. Specifically, we consider the following operator:
$$
\mathbf{o} = f(\mathbf{x} + \mathbf{y}),
$$
where $\mathbf{y}$ is a residual branch, which is commonly used in Transformers. When $\mathbf{w}, \mathbf{b}$ are not used, it is equivalent to $\mathbf{w} = \mathbf{b} = \mathbf{1}$. When $\mathbf{y}$ is not used, it is equivalent to $\mathbf{y} = \mathbf{0}$.

## Forward

**Input**: $\mathbf{x}, \mathbf{y}, \mathbf{w}, \mathbf{b}, c$, where $c$ is a constant and not trainable.

**Computation**:
$$
\begin{aligned}
\mathbf{p} & = \mathbf{x} + \mathbf{y}, \\
\mathbf{q} & = \mathbf{p}, \mathrm{or}, \\
\mathbf{q} & = \mathbf{p} - \left(\sum_{i=1}^d p_i\right)/d, \\
\sigma & = \sqrt{\mathbf{q}^\top \mathbf{q}}, \\
\mathbf{r} & = \mathbf{q} / \sigma, \\
\mathbf{o} & = c \times \mathbf{r} \odot \mathbf{w} + \mathbf{b}.
\end{aligned}
$$

## Backward

**Input**: $\mathbf{do}$.

**Computation**:
$$
\begin{aligned}
\mathbf {db}&= \mathbf {do},\\
\mathbf {dw}&= \mathbf {do} \odot (c\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do} \odot (c\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3   \\
&=1/\sigma  (1_{i=j}-r_i r_j)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r    \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp},\\
\mathbf {dy}& = \mathbf {dp}.

\end{aligned}
$$
