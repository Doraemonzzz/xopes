

# Inverse Attention with Data-Dependent Decay(Sequential Recurrence)

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n}$，记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
0, & i < j.
\end{cases}
$$
考虑如下Linear Attention:
$$
\begin{aligned}
\mathbf O &=\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right]^{-1} \mathbf V ,\\
\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right] \mathbf O &= \mathbf V.

\end{aligned}
$$
另一种形式为：
$$
\left[\left( \mathbf O \mathbf Q^\top\right) \odot \mathbf M
\right] \mathbf K = \mathbf V,

\left[\left( \mathbf Q \mathbf O^\top\right) \odot \mathbf M
\right] \mathbf K = \mathbf V,
$$
