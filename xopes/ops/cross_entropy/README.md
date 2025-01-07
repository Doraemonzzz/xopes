# Cross Entropy

Given the input $\mathbf z \in \mathbb R^v$ and a one-hot label $\mathbf y \in \mathbb R^v, y_k = 1$, along with a smoothing parameter $\lambda \in [0, 1]$, where:
$$
\begin{aligned}
\mathbf {\bar y} &= (1-\lambda) \mathbf y + \lambda / v \mathbf 1, \\
\mathbf 1^\top \mathbf {\bar y} &= (1-\lambda) + \lambda = 1.
\end{aligned}
$$

The output is:
$$
\begin{aligned}
r &= \log\left( \sum_{j=1}^v \exp(z_j) \right), \\

o &= - \sum_{i=1}^v \bar y_i \left(z_i - r \right) \\
  &= - \sum_{i=1}^v \left((1-\lambda) y_i + \lambda / v \right) \left(z_i - r\right) \\
  &= - \sum_{i=1}^v (1-\lambda) y_i\left(z_i - r\right) - \lambda / v \sum_{i=1}^v \left(z_i - r\right) \\
  &= - (1-\lambda) \left(z_k - r\right) - \lambda / v \left( \sum_{i=1}^v z_i - v r \right) \\
  &= - (1-\lambda)z_k + (1-\lambda) r - \lambda / v \left( \sum_{i=1}^v z_i \right) + \lambda r \\
  &= - (1-\lambda)z_k + r - \lambda / v \left( \sum_{i=1}^v z_i \right).
\end{aligned}
$$

---

## Forward

Input: $\mathbf z \in \mathbb R^v$, a one-hot label $\mathbf y \in \mathbb R^v, y_k = 1$, smoothing parameter $\lambda \in [0, 1]$, and ignore index $i_g$.

Compute:
$$
\begin{aligned}
r &= \log\sum_{j=1}^v \exp(z_j), \\
n &= \mathbf 1_{z_k \neq i_g}, \\
s &= \sum_{j=1}^v z_j, \\
o &= -(1-\lambda) z_k + r - \lambda / v s.
\end{aligned}
$$
Here, $n$ is used for reduction across multiple samples (e.g., for mean reduction, return $o / n$).

---

## Backward

Input: $\mathbf{do} \in \mathbb R$.

Compute:
$$
\begin{aligned}
p_k &= \exp(z_k - r), \\
\frac{\partial o}{\partial z_i}
&= -(1-\lambda) \frac{\partial z_k}{\partial z_i} + \frac{\partial r}{\partial z_i} - \lambda / v \frac{\partial \left( \sum_{i=1}^v z_i \right)}{\partial z_i} \\
&= -(1-\lambda) 1_{i=k} + p_i - \lambda / v, \\
\frac{\partial \mathbf o}{\partial \mathbf z}
&= -(1-\lambda) \mathbf y + \mathbf p - \lambda / v, \\
\mathbf{dz}
&= \mathbf{do} \frac{\partial \mathbf o}{\partial \mathbf z} \in \mathbb R^v.
\end{aligned}
$$

For batched inputs, the computation is:
$$
\begin{aligned}
\frac{\partial \mathbf o}{\partial \mathbf Z}
&= -(1-\lambda) \mathbf Y + \mathbf P - \lambda / v \in \mathbb R^{b \times v}, \\
\mathbf{dZ}
&= \mathbf{dO} \odot \frac{\partial \mathbf o}{\partial \mathbf Z}.
\end{aligned}
$$

From the above, $\frac{\partial \mathbf o}{\partial \mathbf Z}$ can be computed directly during the forward pass and cached. During the backward pass, compute the element-wise product $\mathbf{dO} \odot \frac{\partial \mathbf o}{\partial \mathbf Z}$ based on the input $\mathbf{dO}$.

---

## Additional Notes

Two implementations are provided depending on whether the vocabulary is split:
1. **`ce_triton`**: This version does not consider vocabulary splitting. It computes $\frac{\partial \mathbf o}{\partial \mathbf Z}$ during the forward pass and performs element-wise multiplication in the backward pass.
2. **`ce_parallel_triton`**: This version considers vocabulary splitting. It caches the log-sum-exp (LSE) during the forward pass and uses it during the backward computation.
