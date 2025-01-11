# Householder Transform

For an input vector $\mathbf{x} \in \mathbb{R}^d$ and a vector $\mathbf{y} \in \mathbb{R}^d$, we consider the Householder transform:
$$
\mathbf{o} = f(\mathbf{x}, \mathbf{y}) =
(\mathbf{I}_d - 2\mathbf{y}\mathbf{y}^\top / (\mathbf{y}^\top \mathbf{y})) \mathbf{x}
= \mathbf{x} - 2 (\mathbf{y}^\top \mathbf{x}) / (\mathbf{y}^\top \mathbf{y}) \mathbf{y}.
$$

## Forward

Input: $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$.

Computation:
$$
\begin{aligned}
\sigma & = \sqrt{\mathbf{y}^\top \mathbf{y} / d}, \\
\mathbf{\bar{y}} & = \mathbf{y} / \sigma, \\
\mathbf{o} & = (\mathbf{I}_d - 2\mathbf{y}\mathbf{y}^\top / (\mathbf{y}^\top \mathbf{y})) \mathbf{x} \\
& = (\mathbf{I}_d - 2\mathbf{\bar{y}}\mathbf{\bar{y}}^\top / d) \mathbf{x} \\
& = \mathbf{x} - 2 (\mathbf{\bar{y}}^\top \mathbf{x} / d) \mathbf{\bar{y}}.
\end{aligned}
$$

## Backward

Input: $\mathbf{do} \in \mathbb{R}^d$.

Computation:
$$
\begin{aligned}
\frac{\partial o_i}{\partial x_j}
& = 1_{1=j} - 2 \bar{y}_j \bar{y}_i / d, \\
\frac{\partial \mathbf{o}}{\partial \mathbf{x}}
& = \mathbf{I} - 2 \mathbf{\bar{y}} \mathbf{\bar{y}}^\top / d, \\
\mathbf{dx} & = \left(\frac{\partial \mathbf{o}}{\partial \mathbf{x}}\right)^\top \mathbf{do} \\
& = (\mathbf{I} - 2 \mathbf{\bar{y}} \mathbf{\bar{y}}^\top / d)^\top
\end{aligned}
$$
