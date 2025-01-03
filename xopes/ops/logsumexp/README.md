
# LogSumExp

To compute the following for input $\mathbf{x} \in \mathbb{R}^n$:
$$
\begin{aligned}
\mathbf{o} = \mathrm{lse}(\mathbf{x}) = \log\left(\sum_{j=1}^n \exp(x_j)\right).
\end{aligned}
$$

Additionally, define:
$$
\mathrm{se}(\mathbf{x}) = \sum_{j=1}^n \exp(x_j).
$$

Thus:
$$
\mathbf{o} = \log \mathrm{se}(\mathbf{x}).
$$

## Forward

Input: $\mathbf{x} \in \mathbb{R}^d$.

We define:
$$
\begin{aligned}
f(\mathbf{x}) &\geq \max \{x_1, \ldots, x_d \}, \\
\mathbf{o} &= \mathrm{lse}(\mathbf{x}) \\
&= \log\left(\sum_{j=1}^n \exp(x_j)\right) \\
&= \log\left(\sum_{j=1}^n \exp(x_j - f(\mathbf{x}))\right) + f(\mathbf{x}) \\
&\triangleq \mathrm{slse}(\mathbf{x}) + f(\mathbf{x}), \\
\mathrm{se}(\mathbf{x}) &= \exp(\mathrm{slse}(\mathbf{x}) + f(\mathbf{x})) \\
&= \exp(\mathrm{slse}(\mathbf{x})) \exp(f(\mathbf{x})) \\
&\triangleq \mathrm{sse}(\mathbf{x}) \exp(f(\mathbf{x})), \\
\mathrm{lse}(\mathbf{x}) &= \log(\mathrm{sse}(\mathbf{x})) + f(\mathbf{x}).
\end{aligned}
$$

Here, `slse` stands for stable log sum exp, and `sse` stands for stable sum exp.

Given $\mathbf{x}_1 \in \mathbb{R}^{n_1}, \mathbf{x}_2 \in \mathbb{R}^{n_2}, \mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2] \in \mathbb{R}^{n_1 + n_2} = \mathbb{R}^n$, note that:
$$
\begin{aligned}
\mathbf{lse}(\mathbf{x}) &= \log\left(\sum_{j=1}^n \exp(x_j)\right) \\
&= \log\left(\sum_{j=1}^{n_1} \exp(x_j) + \sum_{j=n_1+1}^{n_1+n_2} \exp(x_j)\right) \\
&= \log\left(\exp(\mathrm{lse}(\mathbf{x}_1)) + \exp(\mathrm{lse}(\mathbf{x}_2))\right) \\
&= \log\left(\exp(\mathrm{lse}(\mathbf{x}_1) - f(\mathbf{x})) + \exp(\mathrm{lse}(\mathbf{x}_2) - f(\mathbf{x}))\right) + f(\mathbf{x}) \\
&= \log\left(\exp(\mathrm{slse}(\mathbf{x}_1) + f(\mathbf{x}_1) - f(\mathbf{x})) + \exp(\mathrm{slse}(\mathbf{x}_2) + f(\mathbf{x}_2) - f(\mathbf{x}))\right) + f(\mathbf{x}), \\
f(\mathbf{x}) &= \max(f(\mathbf{x}_1), f(\mathbf{x}_2)).
\end{aligned}
$$

Thus, we can leverage block-wise recursion/parallelism for forward computation acceleration. However, merging blocks involves `exp`, `add`, and `log` operations, which add some computational overhead. To optimize this, we consider using the $\mathrm{sse}$ function:
$$
\begin{aligned}
\mathbf{sse}(\mathbf{x}) &= \sum_{j=1}^n \exp(x_j - f(\mathbf{x})) \\
&= \sum_{j=1}^{n_1} \exp(x_j - f(\mathbf{x})) + \sum_{j=n_1+1}^{n_1+n_2} \exp(x_j - f(\mathbf{x})) \\
&= \sum_{j=1}^{n_1} \exp(x_j - f(\mathbf{x}_1)) \exp(f(\mathbf{x}_1) - f(\mathbf{x})) + \sum_{j=n_1+1}^{n_1+n_2} \exp(x_j - f(\mathbf{x}_2)) \exp(f(\mathbf{x}_2) - f(\mathbf{x})) \\
&= \exp(f(\mathbf{x}_1) - f(\mathbf{x})) \mathbf{sse}(\mathbf{x}_1) + \exp(f(\mathbf{x}_2) - f(\mathbf{x})) \mathbf{sse}(\mathbf{x}_2), \\
f(\mathbf{x}) &= \max(f(\mathbf{x}_1), f(\mathbf{x}_2)).
\end{aligned}
$$

We present the following algorithms, assuming $\mathbf{x} = [\mathbf{x}_1, \ldots, \mathbf{x}_k] \in \mathbb{R}^{kn}$.

### Recursive Version

1. Initialize $m = 0, \mathrm{sse} = 0$.
2. For $i = 1, \ldots, k$:
   - Compute $m_i = \max(\mathbf{x}_i)$;
   - Update $m' = \max(m_i, m)$;
   - Compute $\mathrm{sse}_i = \sum_{j=1}^n \exp(x_{i,j} - m')$;
   - Update $\mathrm{sse} = \exp(m - m') \mathrm{sse} + \mathrm{sse}_i$;
   - Set $m = m'$.
3. Return $m, \mathrm{sse}$.

### Parallel Version

1. Initialize $m = 0, \mathrm{sse} = 0$.
2. Compute in parallel for $i = 1, \ldots, k$:
   - $m_i = \max(\mathbf{x}_i)$;
   - $\mathrm{sse}_i = \sum_{j=1}^n \exp(x_{i,j} - m_i)$.
3. Iterate over $i = 1, \ldots, k$:
   - Update $m' = \max(m_i, m)$;
   - Update $\mathrm{sse} = \exp(m - m') \mathrm{sse} + \exp(m_i - m') \mathrm{sse}_i$;
   - Set $m = m'$.
4. Return $m, \mathrm{sse}$.

### Stability Analysis

Note that $\exp(m - m') \leq 1, \exp(m_i - m') \leq 1, \exp(x_{i,j} - m_i) \leq 1$, ensuring numerical stability at each step.

## Backward

Input: $\mathbf{do} \in \mathbb{R}$.

Compute:
$$
\begin{aligned}
p_i &= \exp(x_i - \mathbf{o}), \\
\frac{\partial o}{\partial x_i} &= p_i, \\
\mathbf{dx} &= \mathbf{do} \odot \mathbf{p}.
\end{aligned}
$$
