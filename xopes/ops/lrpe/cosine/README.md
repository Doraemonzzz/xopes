# Lrpe Cosine

## Notation and Input

- `lrpe_cosine_torch`: Lrpe Cosine Pytorch version.
- `lrpe_cosine_triton`: Lrpe Cosine Triton Version.
  - `lrpe_cosine_triton_fwd`, `lrpe_cosine_triton_bwd`.



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



## Forward

$$
\begin{aligned}
\Theta_{st} &= (\mathrm{offset}+s) \theta_{t} , \Theta\in \mathbb R^{n\times d}, \\
{\mathbf {\bar X}}&=f_{\text{act}}(\mathbf X),\\
\mathbf O &=\mathrm{concat}([\mathbf X  \odot \cos(\Theta),\mathbf X  \odot  \sin(\Theta)])
\in \mathbb R^{n\times 2d}.
\end{aligned}
$$



## Backward

$$
\begin{aligned}
&\Theta_{st} = (\mathrm{offset}+s) \theta_{t} , \Theta\in \mathbb R^{n\times d}, \\
&\mathbf{dO} =\mathrm{concat}[\mathbf{dO}_{\cos},\mathbf{dO}_{\sin}],\\
&\mathbf{dO}_{\cos},\mathbf{dO}_{\sin} \in \mathbb R^{n\times d},  \\
&\mathbf{d{\bar X}} = \mathbf{dO}_{\cos} \odot \cos(\Theta) + \mathbf{dO}_{\sin}\odot \sin( \Theta), \\
&\mathbf {d X} = f'_{\text{act}}(\mathbf{d{\bar X}}).
\end{aligned}
$$
