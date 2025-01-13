# Lrpe Cosine 1d

## Notation and Input

- `lrpe_cosine_torch`: Lrpe Cosine 1d Pytorch version.
- `lrpe_cosine_1d_sp_triton`, `lrpe_cosine_1d_sp_fwd_triton`, `lrpe_cosine_1d_sp_bwd_triton`:
  - Parallel over sequence, use for scenarios where dim != -2.
- `lrpe_cosine_1d_bp_triton`, `lrpe_cosine_1d_bp_fwd_triton`, `lrpe_cosine_1d_bp_bwd_triton`:
  - Perform parallel in block units, use for scenarios where dim = -2.


## Forward

Fwd input:
$$
\mathbf X \in \mathbb R^{n\times d}, \mathbf \theta \in \mathbb R^{d},
\mathrm{offset}\in \mathbb N, \\
\text{act: activation function name}, \text{dim: activation function operate dim}.
$$

Fwd output:
$$
\mathbf O\in \mathbb R^{n\times 2d}.
$$
Computation:
$$
\begin{aligned}
\Theta_{sk} &= (\mathrm{offset}+s) \theta_{k} , \Theta_s\in \mathbb R^{d}, \\
{\mathbf {\bar X}}&=f_{\text{act}}(\mathbf X, \mathrm{dim}),\\
\mathbf O &=\mathrm{concat}([\mathbf {\bar X}  \odot \cos(\Theta),\mathbf {\bar X}  \odot  \sin(\Theta)])
\in \mathbb R^{n\times 2d}.
\end{aligned}
$$



## Backward

Bwd input:
$$
\begin{aligned}
\mathbf {dO} \in \mathbb R^{n\times 2d}.
\end{aligned}
$$
Bwd output:
$$
\mathbf {dX}\in \mathbb R^{n\times d}.
$$
Where $n$ denotes the sequence length, and $d$ denotes the head dimension, $\mathrm{offset}$ denotes the offset(only use this during the inference stage for language model.)

Computation:
$$
\begin{aligned}
&\Theta_{st} = (\mathrm{offset}+s) \theta_{t} , \Theta\in \mathbb R^{ d}, \\
&\mathbf{dO} =\mathrm{concat}[\mathbf{dO}_{\cos},\mathbf{dO}_{\sin}],\\
&\mathbf{dO}_{\cos},\mathbf{dO}_{\sin} \in \mathbb R^{n\times d},  \\
&\mathbf{d{\bar X}} = \mathbf{dO}_{\cos} \odot \cos(\Theta) + \mathbf{dO}_{\sin}\odot \sin( \Theta), \\
&\mathbf {d X} = f'_{\text{act}}(\mathbf{d{\bar X}}, \mathrm{dim}).
\end{aligned}
$$
