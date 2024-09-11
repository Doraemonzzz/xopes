# Fused Linear Attention with Output Gate Fused Act and Lrpe Non Causal Version

## Notation and Input

- `flao_al_non_causal`: Fused Linear Attention With Output Gate Pytorch, it also use act function and lrpe, use as our baseline.
- `flao_fal_non_causal`: Fused Linear Attention With Output Gate, Act, Lrpe in Pytorch, with manually computed gradients.

Fwd input:
$$
\begin{aligned}
\mathbf Q \in \mathbb R^{n\times d},
\mathbf K \in \mathbb R^{m\times d},
\mathbf V \in \mathbb R^{m\times e},
\mathbf G \in \mathbb R^{n\times e},
\Theta \in \mathbb R^{k}.
\end{aligned}
$$

Fwd output:
$$
\mathbf O \in \mathbb R^{n\times e}.
$$

Bwd input:
$$
\begin{aligned}
\mathbf {dQ} \in \mathbb R^{n\times d},
\mathbf {dK} \in \mathbb R^{m\times d},
\mathbf {dV} \in \mathbb R^{m\times e},
\mathbf {dG} \in \mathbb R^{n\times e},
\Theta \in \mathbb R^{k}.
\end{aligned}
$$

Bwd output:
$$
\mathbf {dO}\in \mathbb R^{n\times e}.
$$

Where $n, m$ denotes the sequence length, and $d, e$ denotes the head dimension, $k$ denotes $\Theta$ head dim.


Here, we only list the forward pass, as the backward pass is merely a combination of previous methods.

## Forward

$$
\begin{aligned}
\bar {\mathbf Q}&= \mathrm{lrpe}(f_q \mathbf (Q), \Theta), \\
\bar {\mathbf K}&= \mathrm{lrpe}(f_k \mathbf (K), \Theta), \\
\bar {\mathbf V}&= f_v (\mathbf V), \\
\bar {\mathbf G}&= f_v (\mathbf G), \\
\mathbf {KV}&=\bar {\mathbf K} \bar {\mathbf V} \in \mathbb R^{d\times e}, \\
\mathbf {QKV}&= \bar {\mathbf Q}[\mathbf {KV}]\in \mathbb R^{n \times e},  \\
\mathbf O &= \bar {\mathbf G}\odot [\mathbf {QKV}] \in \mathbb R^{n \times e} .
\end{aligned}
$$
