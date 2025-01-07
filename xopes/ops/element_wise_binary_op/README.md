
# Element-wise Binary Operation

For inputs $\mathbf x \in \mathbb R^{n_1 \times \ldots \times n_k \times n_{k+1} \times \ldots \times n_{k+m}}$ and $\mathbf y \in \mathbb R^{n_1 \times \ldots \times n_k}$, along with a binary operator $f$, we perform the following operation:
$$
\mathbf o_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}} = f(\mathbf x_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}},
\mathbf y_{i_1, \ldots, i_k}).
$$

## Forward

**Input:** $\mathbf x \in \mathbb R^{n_1 \times \ldots \times n_k \times n_{k+1} \times \ldots \times n_{k+m}}$ and $\mathbf y \in \mathbb R^{n_1 \times \ldots \times n_k}$, along with a binary operator $f$.

**Computation:**
$$
\mathbf o_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}} = f(\mathbf x_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}},
\mathbf y_{i_1, \ldots, i_k}).
$$

## Backward

**Input:** $\mathbf{do} \in \mathbb R^{n_1 \times \ldots \times n_k \times n_{k+1} \times \ldots \times n_{k+m}}$.

**Computation:**
$$
\begin{aligned}
\mathbf{dx}_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}}
&= \mathbf{do}_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}} \odot
\frac{\partial \mathbf o_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}}}
{\partial \mathbf x_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}}}, \\
\mathbf{dy}_{i_1, \ldots, i_k}
&= \sum_{i_{k+1}, \ldots, i_{k+m}} \mathbf{do}_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}} \odot
\frac{\partial \mathbf o_{i_1, \ldots, i_k, i_{k+1}, \ldots, i_{k+m}}}
{\partial \mathbf y_{i_1, \ldots, i_k}}.
\end{aligned}
$$

## Supplementary

Through experiments, it was found that `torch.compile` provides reasonably good results.
