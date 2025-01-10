# Normalize

For input $\mathbf{x} \in \mathbb{R}^d$, this section discusses various normalization algorithms. These algorithms are defined as follows:
$$
\begin{aligned}
\mathbf{o}
&= f(\mathbf{x}) \\
&= c \times \frac{\mathbf{x}_1 \odot \mathbf{w}}{\sqrt{\mathbf{x}_1^\top \mathbf{x}_1}} + \mathbf{b} \\
&= \frac{c}{\sqrt{d}} \times \frac{\mathbf{x}_1 \odot \mathbf{w}}{\sqrt{\mathbf{x}_1^\top \mathbf{x}_1 / d}} + \mathbf{b}.
\end{aligned}
$$
(The last equation is used to reduce the impact of numerical precision.)

Here, $c \in \mathbb{R}$, $\mathbf{w}, \mathbf{b} \in \mathbb{R}^d$, and $\mathbf{x}_1 = \mathbf{x}$ or $\mathbf{x}_1 = \mathbf{x} - \bar{\mathbf{x}}, \bar{\mathbf{x}} = (\sum_{i=1}^d x_i) / d$. Specifically, we consider the following operator:
$$
\mathbf{o} = f(\mathbf{x} + \mathbf{y}),
$$
where $\mathbf{y}$ is a residual branch, commonly seen in Transformers. When $\mathbf{w}$ and $\mathbf{b}$ are not used, it is equivalent to setting $\mathbf{w} = \mathbf{b} = \mathbf{1}$; when $\mathbf{y}$ is not used, it is equivalent to setting $\mathbf{y} = \mathbf{0}$.

## Forward

Input: $\mathbf{x}, \mathbf{y}, \mathbf{w}, \mathbf{b}, c$, where $c$ is a constant and not learnable.

Computation:
$$
\begin{aligned}
\mathbf{p} &= \mathbf{x} + \mathbf{y}, \\
\mathbf{q} &= \mathbf{p}, \text{ or } \\
\mathbf{q} &= \mathbf{p} - \left(\sum_{i=1}^d p_i\right) / d, \\
\sigma &= \sqrt{\mathbf{q}^\top \mathbf{q} / d}, \\
\mathbf{r} &= \mathbf{q} / \sigma, \\
\mathbf{o} &= \frac{c}{\sqrt{d}} \times \mathbf{r} \odot \mathbf{w} + \mathbf{b}.
\end{aligned}
$$

## Backward

Input: $\mathbf{do}$.

Computation:
$$
\begin{aligned}
\mathbf{db} &= \mathbf{do}, \\
\mathbf{dw} &= \mathbf{do} \odot \left(\frac{c}{\sqrt{d}} \times \mathbf{r}\right), \\
\mathbf{dr} &= \mathbf{do} \odot \left(\frac{c}{\sqrt{d}} \times \mathbf{w}\right), \\
\frac{\partial r_i}{\partial q_j}
&= \frac{1_{i=j}}{\sigma} - \frac{q_i}{\sigma^2} \frac{\partial \sigma}{\partial q_j} \\
&= \frac{1_{i=j}}{\sigma} - \frac{q_i}{\sigma^2} \left(\frac{1}{2} \times (\mathbf{q}^\top \mathbf{q})^{-1/2} \times \frac{2 q_j}{\sqrt{d}}\right) \\
&= \frac{1_{i=j}}{\sigma} - \frac{q_i}{\sigma^2} \left(\frac{\sigma^{-1}}{\sqrt{d}} \times \frac{q_j}{\sqrt{d}}\right) \\
&= \frac{1_{i=j}}{\sigma} - \frac{q_i q_j}{\sigma^3 / d} \\
&= \frac{1}{\sigma} \left(1_{i=j} - \frac{r_i r_j}{d}\right), \\
\frac{\partial \mathbf{r}}{\partial \mathbf{q}}
&= \frac{1}{\sigma} \left(\mathbf{I} - \frac{\mathbf{r} \mathbf{r}^\top}{d}\right), \\
\mathbf{dq}
&= \left(\frac{\partial \mathbf{r}}{\partial \mathbf{q}}\right)^\top \mathbf{dr} \\
&= \frac{1}{\sigma} \left(\mathbf{I} - \frac{\mathbf{r} \mathbf{r}^\top}{d}\right) \mathbf{dr} \\
&= \frac{1}{\sigma} \left(\mathbf{dr} - \frac{(\mathbf{r}^\top \mathbf{dr}) \mathbf{r}}{d}\right), \\
\mathbf{dp} &= \mathbf{dq}, \text{ or } \\
\mathbf{d}p_k &= \sum_{i=1}^d \mathbf{d}q_i \frac{\partial q_i}{\partial p_k} \\
&= \sum_{i=1}^d \mathbf{d}q_i \left(1_{i=k} - \frac{1}{d}\right) \\
&= \mathbf{dq}_k - \frac{1}{d} \left(\sum_{i=1}^d \mathbf{dq}_i\right), \\
\mathbf{dp} &= \mathbf{dq} - \bar{\mathbf{dq}}, \\
\mathbf{dx} &= \mathbf{dp}, \\
\mathbf{dy} &= \mathbf{dp}.
\end{aligned}
$$

## Fuse Normalize and Residual

Consider how to implement fused normalization and residual connections for Transformer layers.

### Naive Implementation

Given input $\mathbf{x}$:
$$
\begin{aligned}
\mathbf{x}_0 &= \mathbf{x}, \\
\mathbf{y}_k &= \mathrm{norm}(\mathbf{x}_{k-1}), \\
\mathbf{x}_k &= f_k(\mathbf{y}_k) + \mathbf{x}_{k-1}, \\
\mathbf{o} &= \mathbf{x}_n, \\
k &= 1, \ldots, n.
\end{aligned}
$$
Observe that:
$$
\begin{aligned}
\mathbf{y}_k &= \mathrm{norm}(\mathbf{x}_{k-1}) \\
&= \mathrm{norm}(\mathbf{x}_{k-2} + \mathbf{z}_{k-1}), \\
\mathbf{z}_k &\triangleq f_k(\mathbf{y}_k), \\
\mathbf{x}_k &= \mathbf{x}_{k-1} + \mathbf{z}_k.
\end{aligned}
$$
From this observation, we derive the fused implementation:
$$
\begin{aligned}
\mathbf{p}_0 &= \mathbf{x}, \\
\mathbf{r}_0 &= \mathbf{0}, \\
\mathbf{r}_k &= \mathbf{p}_{k-1} + \mathbf{r}_{k-1}, \\
\mathbf{q}_k &= \mathrm{norm}(\mathbf{r}_k), \\
\mathbf{p}_k &= f_k(\mathbf{q}_k), \\
\mathbf{o} &= \mathbf{p}_n + \mathbf{r}_n, \\
k &= 1, \ldots, n.
\end{aligned}
$$
We prove the equivalence of the two methods by mathematical induction.

**Base Case ($n=1$):**
$$
\begin{aligned}
\mathbf{o}_{\mathrm{naive}} &= \mathbf{x}_1 \\
&= f_1(\mathbf{y}_1) + \mathbf{x}_0 \\
&= f_1(\mathrm{norm}(\mathbf{x}_0)) + \mathbf{x}_0 \\
&= f_1(\mathrm{norm}(\mathbf{x})) + \mathbf{x}, \\
\mathbf{o}_{\mathrm{fuse}} &= \mathbf{p}_1 + \mathbf{r}_1 \\
&= f_1(\mathbf{q}_1) + \mathbf{r}_1 \\
&= f_1(\mathrm{norm}(\mathbf{p}_0 + \mathbf{r}_0)) + \mathbf{p}_0 \\
&= f_1(\mathrm{norm}(\mathbf{x})) + \mathbf{x}.
\end{aligned}
$$

**Inductive Step:** Assuming equivalence for $n-1$, we prove for $n$:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{naive}} &= \mathbf{x}_n \\
&= f_n(\mathbf{y}_n) + \mathbf{x}_{n-1} \\
&= f_n(\mathrm{norm}(\mathbf{x}_{n-1})) + \mathbf{x}_{n-1}, \\
\mathbf{o}_{\mathrm{fuse}} &= \mathbf{p}_n + \mathbf{r}_n \\
&= f_n(\mathbf{q}_n) + \mathbf{p}_{n-1} + \mathbf{r}_{n-1} \\
&= f_n(\mathrm{norm}(\mathbf{p}_{n-1} + \mathbf{r}_{n-1})) + \mathbf{p}_{n-1} + \mathbf{r}_{n-1}.
\end{aligned}
$$
By the induction hypothesis:
$$
\mathbf{x}_{n-1} = \mathbf{p}_{n-1} + \mathbf{r}_{n-1}.
$$
Thus:
$$
\begin{aligned}
\mathbf{o}_{\mathrm{fuse}}
&= f_n(\mathrm{norm}(\mathbf{x}_{n-1})) + \mathbf{x}_{n-1} \\
&= \mathbf{o}_{\mathrm{naive}}.
\end{aligned}
$$

### Backpropagation Update

With input and output:
$$
\mathbf{o} = \mathrm{norm}(\mathbf{x} + \mathbf{y}), \quad \mathbf{r} = \mathbf{x} + \mathbf{y},
$$
the gradients are updated as:
$$
\begin{aligned}
\mathbf{dx} &= \mathbf{dx} + \mathbf{dr}, \\
\mathbf{dy} &= \mathbf{dx}.
\end{aligned}
$$
