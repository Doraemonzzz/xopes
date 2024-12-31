
# Householder Transform

For an input $\mathbf{x} \in \mathbb{R}^d$ and a vector $\mathbf{y} \in \mathbb{R}^d$, we consider the Householder transform:
$$
\mathbf{o} = f(\mathbf{x}, \mathbf{y}) =
(\mathbf{I}_d - 2\mathbf{y}\mathbf{y}^\top / (\mathbf{y}^\top \mathbf{y}))\mathbf{x}
= \mathbf{x} - 2(\mathbf{y}^\top \mathbf{x}) / (\mathbf{y}^\top \mathbf{y})\mathbf{y}.
$$

## Forward

**Input**: $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$.

**Computation**:
$$
\begin{aligned}
\sigma & = \sqrt{\mathbf{y}^\top \mathbf{y}}, \\
\mathbf{\bar{y}} & = \mathbf{y} / \sigma, \\
\mathbf{o} & = (\mathbf{I}_d - 2\mathbf{y}\mathbf{y}^\top / (\mathbf{y}^\top \mathbf{y}))\mathbf{x} \\
& = (\mathbf{I}_d - 2\mathbf{\bar{y}}\mathbf{\bar{y}}^\top)\mathbf{x} \\
& = \mathbf{x} - 2(\mathbf{\bar{y}}^\top \mathbf{x})\mathbf{\bar{y}}.
\end{aligned}
$$

## Backward

**Input**: $\mathbf{do}$.

**Computation**:
$$
\begin{aligned}
\frac{\partial o_i}{\partial x_j}
& = 1_{i=j} - 2\bar{y}_j\bar{y}_i, \\
\frac{\partial \mathbf{o}}{\partial \mathbf{x}}
& = \mathbf{I} - 2\mathbf{\bar{y}}\mathbf{\bar{y}}^\top, \\
\mathbf{dx} & = \left(\frac{\partial \mathbf{o}}{\partial \mathbf{x}}\right)^\top \mathbf{do} \\
& = (\mathbf{I} - 2\mathbf{\bar{y}}\mathbf{\bar{y}}^\top)^\top \mathbf{do} \\
& = \mathbf{do} - 2(\mathbf{\bar{y}}^\top \mathbf{do})\mathbf{\bar{y}}.
\end{aligned}
$$

For the gradient with respect to another term, consider $f$:
$$
\begin{aligned}
\frac{\partial o_i}{\partial \bar{y}_j}
& = -2\frac{\partial \left((\sum_{k=1}^d \bar{y}_k x_k) \bar{y}_i\right)}{\partial \bar{y}_j}, \\
& = -2\left(x_j\bar{y}_i + (\mathbf{\bar{y}}^\top \mathbf{x})\mathbf{1}_{i=j}\right), \\
\frac{\partial \mathbf{o}}{\partial \mathbf{\bar{y}}}
& = -2\mathbf{\bar{y}}\mathbf{x}^\top - 2\mathbf{\bar{y}}^\top \mathbf{x} \mathbf{I}, \\
\mathbf{d\bar{y}} & = \left(\frac{\partial \mathbf{o}}{\partial \mathbf{\bar{y}}}\right)^\top \mathbf{do} \\
& = \left(-2\mathbf{\bar{y}}\mathbf{x}^\top - 2\mathbf{\bar{y}}^\top \mathbf{x} \mathbf{I}\right)^\top \mathbf{do} \\
& = -2 (\mathbf{do}^\top \mathbf{\bar{y}})\mathbf{x} - 2(\mathbf{\bar{y}}^\top \mathbf{x}) \mathbf{do}.
\end{aligned}
$$

Using the results derived in the Normalize section:
$$
\begin{aligned}
\frac{\partial \mathbf{\bar{y}}}{\partial \mathbf{y}}
& = 1 / \sigma (\mathbf{I} - \mathbf{\bar{y}}\mathbf{\bar{y}}^\top), \\
\mathbf{dy} & = \left(\frac{\partial \mathbf{\bar{y}}}{\partial \mathbf{y}}\right)^\top \mathbf{d\bar{y}} \\
& = 1 / \sigma (\mathbf{I} - \mathbf{\bar{y}}\mathbf{\bar{y}}^\top) \mathbf{d\bar{y}} \\
& = 1 / \sigma (\mathbf{d\bar{y}} - (\mathbf{\bar{y}}^\top \mathbf{d\bar{y}})\mathbf{\bar{y}}).
\end{aligned}
$$
