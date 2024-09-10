# Lrpe Cosine

## Notation and Input

- `lrpe_cosine_torch`: Lrpe Cosine Pytorch version.
- `lrpe_cosine_triton`: Lrpe Cosine Triton Version.
  - `lrpe_cosine_triton_fwd`, `lrpe_cosine_triton_bwd`.



Fwd input:
$$
\begin{aligned}
\mathbf X \in \mathbb R^{n\times d}, \mathbf \theta \in \mathbb R^{d},
\mathrm{offset}\in \mathbb N.
\end{aligned}
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
&\mathbf{dX} = \mathbf{dO}_{\cos} \odot \cos(\Theta) + \mathbf{dO}_{\sin}\odot \sin( \Theta).
\end{aligned}
$$
