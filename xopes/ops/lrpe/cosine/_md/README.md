# Lrpe Cosine md

## Notation and Input

- `lrpe_cosine_md_torch`: Lrpe Cosine Pytorch version.
- `lrpe_cosine_md_triton`, `lrpe_cosine_md_fwd_triton`, `lrpe_cosine_md_bwd_triton`:
  - Parallel over sequence.
- `lrpe_cosine_md_cache_triton`, `lrpe_cosine_md_cache_fwd_triton`, `lrpe_cosine_md_cache_bwd_triton`:
  - Parallel over sequence, with cache theta.



Fwd input:
$$
\mathbf X \in \mathbb R^{(n+l)\times d}, \mathbf \theta \in \mathbb R^{e},
l\in \mathbb N,k\in \mathbb N,
n=\prod_{j=1}^k n_j,  \\
e\ge d/n, e\in \mathbb N, \\
\text{act: activation function name}, \text{dim: activation function operate dim}.
$$

Fwd output:
$$
\mathbf O\in \mathbb R^{(n+l)\times 2d}.
$$
Bwd input:
$$
\begin{aligned}
\mathbf {dO} \in \mathbb R^{(n+l)\times 2d}.
\end{aligned}
$$
Bwd output:
$$
\mathbf {dX}\in \mathbb R^{(n+l)\times d}.
$$
Where $n$ denotes the sequence length, and $d$ denotes the head dimension, $l$ denotes the length of extra token(e.g, condition), $k$ denotes the number dimensions.

Here we map the serial number $s$ to $(s_1, \ldots, s_k)$, s.t:
$$
\sum_{j=1}^k s_j N_j = s, \\
N_1=1, N_j = n_{j-1}N_j.
$$
For example, we can represent $s = 5$ as $(2, 1)$, where $(n_2, n_1) = (3, 2)$.

In addition, we do not use LRPE for the first $l$ tokens, i.e.,
$$
\mathbf {\bar X}_j= \mathrm{concat}(\mathbf X_j, \mathbf 0)\in \mathbb R^{2\times d}.
$$
and this is equivalent to:
$$
\mathbf {\bar X}_j= \mathrm{concat}(\mathbf  {\bar X}  \odot \cos(\mathbf 0),\mathbf  {\bar X} \odot  \sin(\mathbf 0))\in \mathbb R^{2\times d}.
$$


## Forward

$$
\begin{aligned}
\Theta_{j} &= \mathbb 0\in \mathbb R^{d},0\le j \le l-1,    \\
\Theta_{s+l} &= \mathrm{concat}(\{s_j\theta| j=1,\ldots, k\})[:d] , \Theta\in \mathbb R^{d}, \\
{\mathbf {\bar X}}&=f_{\text{act}}(\mathbf X, \mathrm{dim}),\\
\mathbf O &=\mathrm{concat}([\mathbf  {\bar X}  \odot \cos(\Theta),\mathbf  {\bar X} \odot  \sin(\Theta)])
\in \mathbb R^{(n+l)\times 2d}.
\end{aligned}
$$



## Backward

$$
\begin{aligned}
&\Theta_{j} = \mathbb 0\in \mathbb R^{d},0\le j \le l-1,    \\
&\Theta_{s+l} = \mathrm{concat}(\{s_j\theta| j=1,\ldots, k\})[:d] , \Theta\in \mathbb R^{d}, \\
&\mathbf{dO} =\mathrm{concat}[\mathbf{dO}_{\cos},\mathbf{dO}_{\sin}],\\
&\mathbf{dO}_{\cos},\mathbf{dO}_{\sin} \in \mathbb R^{(n+l)\times d},  \\
&\mathbf{d{\bar X}} = \mathbf{dO}_{\cos} \odot \cos(\Theta) + \mathbf{dO}_{\sin}\odot \sin( \Theta), \\
&\mathbf {d X} = f'_{\text{act}}(\mathbf{d{\bar X}}).
\end{aligned}
$$
