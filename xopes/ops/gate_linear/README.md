# Gate Linear

## Forward Propagation

Given the inputs $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^{n \times d_1}$, weights $\mathbf{W} \in \mathbb{R}^{d_1 \times d_2}$, residual term $\mathbf{r} \in \mathbb{R}^{n \times d_2}$, and activation function $f$, the output is computed as:
$$
\mathbf{o} = [f(\mathbf{x}_1) \odot \mathbf{x}_2] \mathbf{W} + \mathbf{r}.
$$
The forward pass caches $\mathbf{x}_1, \mathbf{x}_2, \mathbf{W}$.

## Backward Propagation

Given the input $\mathbf{do} \in \mathbb{R}^{n \times r_2}$, compute:
$$
\begin{aligned}
\mathbf{dr} & = \mathbf{do}, \\
\mathbf{y} & = f(\mathbf{x}_1) \odot \mathbf{x}_2, \\
\mathbf{dW} & = \mathbf{y}^\top \mathbf{do}, \\
\mathbf{dy} & = \mathbf{do} \mathbf{W}^\top, \\
\mathbf{dx}_2 & = f(\mathbf{x}_1) \odot \mathbf{dy}, \\
\mathbf{dx}_1 & = f(\mathbf{x}_1)' \odot \mathbf{x}_2 \odot \mathbf{dy}.
\end{aligned}
$$
