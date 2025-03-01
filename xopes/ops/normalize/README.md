# Normalize with Residual or Gate

For an input $\mathbf{x} \in \mathbb{R}^d$, this section discusses various normalization algorithms, defined as follows:
$$
\begin{aligned}
\mathbf{o} &= f(\mathbf{x}) \\
&= c \times \frac{\mathbf{x}_1 \odot \mathbf{w}}{\sqrt{\mathbf{x}_1^\top \mathbf{x}_1}} + \mathbf{b} \\
&= (c/\sqrt{d}) \times \frac{\mathbf{x}_1 \odot \mathbf{w}}{\sqrt{\mathbf{x}_1^\top \mathbf{x}_1/d}} + \mathbf{b}.
\end{aligned}
$$
(The last equation is used to reduce numerical precision errors.)

Here, $c \in \mathbb{R}$, $\mathbf{w}, \mathbf{b} \in \mathbb{R}^d$, and $\mathbf{x}_1$ can be either $\mathbf{x}$ or $\mathbf{x}_1 = \mathbf{x} - \bar{\mathbf{x}}$, where $\bar{\mathbf{x}} = (\sum_{i=1}^{d} x_i)/d$.

We consider two classes of operators. The first includes residual connections, referred to as **residual with residual**:
$$
\mathbf{o} = f(\mathbf{x} + \mathbf{y}).
$$
Here, $\mathbf{y}$ is a residual branch, commonly used in Transformers. If $\mathbf{w}$ and $\mathbf{b}$ are not used, this is equivalent to setting $\mathbf{w} = \mathbf{b} = \mathbf{1}$. If $\mathbf{y}$ is not used, it is equivalent to setting $\mathbf{y} = \mathbf{0}$.

The second class is **residual with gate**:
$$
\mathbf{o} = f(\mathbf{x} \odot g(\mathbf{y})) \quad \text{or} \quad \mathbf{o} = f(\mathbf{x}) \odot g(\mathbf{y}).
$$

## Normalize with Residual

### Forward

**Input**: $\mathbf{x}, \mathbf{y}, \mathbf{w}, \mathbf{b}, c$, where $c$ is a constant and not learnable.

**Computation**:
$$
\begin{aligned}
\mathbf{p} &= \mathbf{x} + \mathbf{y}, \\
\mathbf{q} &= \mathbf{p}, \quad \text{or} \\
\mathbf{q} &= \mathbf{p} - \left(\sum_{i=1}^{d} p_i\right)/d, \\
\sigma &= \sqrt{\mathbf{q}^\top \mathbf{q} / d}, \\
\mathbf{r} &= \mathbf{q} / \sigma, \\
\mathbf{o} &= (c/\sqrt{d}) \times \mathbf{r} \odot \mathbf{w} + \mathbf{b}.
\end{aligned}
$$

### Backward

**Input**: $\mathbf{do}$.

**Computation**:
$$
\begin{aligned}
\mathbf{db} &= \mathbf{do}, \\
\mathbf{dw} &= \mathbf{do} \odot (c/\sqrt{d} \times \mathbf{r}), \\
\mathbf{d r} &= \mathbf{do} \odot (c/\sqrt{d} \times \mathbf{w}), \\
\frac{\partial r_i}{\partial q_j} &= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j} \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt{d} \times q_j / \sqrt{d} \right) \\
&= 1_{i=j}/\sigma - q_i q_j / \sigma^3 / d, \\
\frac{\partial \mathbf{r}}{\partial \mathbf{q}} &= 1/\sigma (\mathbf{I} - \mathbf{r} \mathbf{r}^\top / d), \\
\mathbf{dq} &= 1/\sigma \left( \mathbf{dr} - (\mathbf{r}^\top \mathbf{dr}) \mathbf{r} / d \right), \\
\mathbf{dp} &= \mathbf{dq}, \quad \text{or} \\
\mathbf{dp} &= \mathbf{dq} - \bar{\mathbf{dq}}, \\
\mathbf{dx} &= \mathbf{dp}, \\
\mathbf{dy} &= \mathbf{dp}.
\end{aligned}
$$

### Fusing Normalize and Residual

For a Transformer layer, we consider how to fuse normalization and residual connections.

A **naïve** implementation, given an input $\mathbf{x}$:
$$
\begin{aligned}
\mathbf{x}_0 &= \mathbf{x}, \\
\mathbf{y}_k &= \mathrm{norm}(\mathbf{x}_{k-1}), \\
\mathbf{x}_k &= f_k(\mathbf{y}_k) + \mathbf{x}_{k-1}, \\
\mathbf{o} &= \mathbf{x}_n, \quad k = 1, \dots, n.
\end{aligned}
$$
Observing that:
$$
\begin{aligned}
\mathbf{y}_k &= \mathrm{norm}(\mathbf{x}_{k-1}) \\
&= \mathrm{norm}(\mathbf{x}_{k-2} + \mathbf{z}_{k-1}), \\
\mathbf{z}_k &\triangleq f_k(\mathbf{y}_k), \\
\mathbf{x}_k &= \mathbf{x}_{k-1} + \mathbf{z}_k.
\end{aligned}
$$
we derive the **fused** implementation:
$$
\begin{aligned}
\mathbf{p}_0 &= \mathbf{x}, \\
\mathbf{r}_0 &= \mathbf{0}, \\
\mathbf{r}_k &= \mathbf{p}_{k-1} + \mathbf{r}_{k-1}, \\
\mathbf{q}_k &= \mathrm{norm}(\mathbf{r}_k), \\
\mathbf{p}_k &= f_k(\mathbf{q}_k), \\
\mathbf{o} &= \mathbf{p}_n + \mathbf{r}_n, \quad k = 1, \dots, n.
\end{aligned}
$$
Using mathematical induction, we can prove that both implementations produce the same result.

### Backpropagation Updates

The function’s input and output:
$$
\mathbf{o} = \mathrm{norm}(\mathbf{x} + \mathbf{y}), \quad \mathbf{r} = \mathbf{x} + \mathbf{y}.
$$
Thus:
$$
\begin{aligned}
\mathbf{dx} &= \mathbf{dx} + \mathbf{dr}, \\
\mathbf{dy} &= \mathbf{dx}.
\end{aligned}
$$

---

## Normalize with Gate

### Forward

**Pre-gate**:
$$
\begin{aligned}
\mathbf p  & =\mathbf x\odot  g(\mathbf y),\\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o_1&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b, \\
\mathbf o& = \mathbf o_1.

\end{aligned}
$$

**Post-gate**:
$$
\begin{aligned}
\mathbf p  & =\mathbf x\odot  g(\mathbf y),\\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o_1&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b, \\
\mathbf o& = \mathbf o_1.

\end{aligned}
$$

post-gate:
$$
\begin{aligned}
\mathbf p  & =\mathbf x,\\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o_1&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b, \\
\mathbf o& = \mathbf o_1\odot g(\mathbf y).

\end{aligned}
$$

### Backward

**pre-gate**:
$$
\begin{aligned}
\mathbf {do}_1 & = \mathbf {do} ,  \\
\mathbf {db}&= \mathbf {do}_1,\\
\mathbf {dw}&= \mathbf {do}_1 \odot (c/\sqrt d\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do}_1 \odot (c /\sqrt d\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt d\times q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3 /d   \\
&=1/\sigma  (1_{i=j}-r_i r_j /d)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r /d   \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp} \odot g(\mathbf y) ,\\
\mathbf {dy}& = \mathbf {dp} \odot \mathbf x, \\

\mathbf {dy} &= g'(\mathbf y) \odot \mathbf {dy}.

\end{aligned}
$$

**post-gate**:
$$
\begin{aligned}
\mathbf {do}_1 & = \mathbf {do} \odot g(\mathbf y) ,  \\
\mathbf {db}&= \mathbf {do}_1,\\
\mathbf {dw}&= \mathbf {do}_1 \odot (c/\sqrt d\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do}_1 \odot (c /\sqrt d\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt d\times q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3 /d   \\
&=1/\sigma  (1_{i=j}-r_i r_j /d)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r /d   \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp},\\
\mathbf {dy}& = \mathbf {do} \odot \mathbf o_1, \\

\mathbf {dy} &= g'(\mathbf y) \odot \mathbf {dy}.

\end{aligned}
$$
