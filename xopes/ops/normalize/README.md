
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

## Fuse Normalize and Residual

Here we discuss how to implement **Fuse Normalize and Residual** for a Transformer layer.

### Naive Implementation

Assuming the input is $\mathbf{x}$, the naive implementation follows:
$$
\begin{aligned}
\mathbf{x}_0 & = \mathbf{x}, \\
\mathbf{y}_k & = \mathrm{norm}(\mathbf{x}_{k-1}), \\
\mathbf{x}_k & = f_k(\mathbf{y}_k) + \mathbf{x}_{k-1}, \\
\mathbf{o} & = \mathbf{x}_n, \\
k & = 1, \ldots, n.
\end{aligned}
$$

Notice that:
$$
\begin{aligned}
\mathbf{y}_k & = \mathrm{norm}(\mathbf{x}_{k-1}) \\
& = \mathrm{norm}(\mathbf{x}_{k-2} + \mathbf{z}_{k-1}), \\
\mathbf{z}_k & \triangleq f_k(\mathbf{y}_k), \\
\mathbf{x}_k & = \mathbf{x}_{k-1} + \mathbf{z}_k.
\end{aligned}
$$

### Fused Implementation

Based on the observations above, the fused implementation is:
$$
\begin{aligned}
\mathbf{p}_0 & = \mathbf{x}, \\
\mathbf{r}_0 & = \mathbf{0}, \\
\mathbf{r}_k & = \mathbf{p}_{k-1} + \mathbf{r}_{k-1}, \\
\mathbf{q}_k & = \mathrm{norm}(\mathbf{r}_k), \\
\mathbf{p}_k & = f_k(\mathbf{q}_k), \\
\mathbf{o} & = \mathbf{p}_n + \mathbf{r}_n, \\
k & = 1, \ldots, n.
\end{aligned}
$$

### Proof of Equivalence

Using mathematical induction, we prove that the two implementations yield the same result.

#### Base Case: $n = 1$

For the naive implementation:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{naive}} & = \mathbf{x}_1 \\
& = f_1(\mathbf{y}_1) + \mathbf{x}_0 \\
& = f_1(\mathrm{norm}(\mathbf{x}_0)) + \mathbf{x}_0 \\
& = f_1(\mathrm{norm}(\mathbf{x})) + \mathbf{x}.
\end{aligned}
$$

For the fused implementation:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{fuse}} & = \mathbf{p}_1 + \mathbf{r}_1 \\
& = f_1(\mathbf{q}_1) + \mathbf{r}_1 \\
& = f_1(\mathrm{norm}(\mathbf{p}_0 + \mathbf{r}_0)) + \mathbf{p}_0 \\
& = f_1(\mathrm{norm}(\mathbf{x})) + \mathbf{x}.
\end{aligned}
$$

Thus, $\mathbf{o}_{\mathrm{naive}} = \mathbf{o}_{\mathrm{fuse}}$ for $n = 1$.

#### Inductive Step: Assume true for $n-1$, prove for $n$

For the naive implementation:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{naive}} & = \mathbf{x}_n \\
& = f_n(\mathbf{y}_n) + \mathbf{x}_{n-1} \\
& = f_n(\mathrm{norm}(\mathbf{x}_{n-1})) + \mathbf{x}_{n-1}.
\end{aligned}
$$

For the fused implementation:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{fuse}} & = \mathbf{p}_n + \mathbf{r}_n \\
& = f_n(\mathbf{q}_n) + \mathbf{p}_{n-1} + \mathbf{r}_{n-1} \\
& = f_n(\mathrm{norm}(\mathbf{p}_{n-1} + \mathbf{r}_{n-1})) + \mathbf{p}_{n-1} + \mathbf{r}_{n-1}.
\end{aligned}
$$

From the induction hypothesis:
$$
\mathbf{x}_{n-1} = \mathbf{p}_{n-1} + \mathbf{r}_{n-1}.
$$

Thus:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{fuse}} & = f_n(\mathrm{norm}(\mathbf{x}_{n-1})) + \mathbf{x}_{n-1} \\
& = \mathbf{o}_{\mathrm{naive}}.
\end{aligned}
$$

Therefore, the conclusion holds for all $n$.

### Backward Propagation Update

Consider the case where the function has inputs and outputs:
$$
\mathbf{o} = \mathrm{norm}(\mathbf{x} + \mathbf{y}), \quad \mathbf{r} = \mathbf{x} + \mathbf{y}.
$$

The gradients are updated as:
$$
\begin{aligned}
\mathbf{dx} & = \mathbf{dx} + \mathbf{dr}, \\
\mathbf{dy} & = \mathbf{dx}.
\end{aligned}
$$
