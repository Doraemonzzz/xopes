# Fused Linear Attention with Output Gate Non Causal Version

## Notation and Input

- `lao_non_causal_torch`: Linear Attention With Output Gate Pytorch version, the naive implementation of Linear Attention with Output Gate in PyTorch.
- `flao_non_causal_torch`: Fused Linear Attention With Output Gate Pytorch, with manually computed gradients.
- `flao_non_causal_triton`: The Triton version of `flao_non_causal_torch`, where the forward pass is implemented using Triton, and the backward pass is implemented using Torch (as I found no speed advantage in the forward pass.)

Fwd input:
$$
\begin{aligned}
\mathbf Q \in \mathbb R^{n\times d},
\mathbf K \in \mathbb R^{m\times d},
\mathbf V \in \mathbb R^{m\times e},
\mathbf G \in \mathbb R^{n\times e}.
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
\mathbf {dG} \in \mathbb R^{n\times e}.
\end{aligned}
$$

Bwd output:
$$
\mathbf {dO}\in \mathbb R^{n\times e}.
$$

Where $n, m$ denotes the sequence length, and $d, e$ denotes the head dimension.



## Forward

$$
\begin{aligned}
\mathbf {KV}&=\mathbf K^\top \mathbf V \in \mathbb R^{d\times e}, \\
\mathbf {QKV}&= \mathbf Q[\mathbf {KV}]\in \mathbb R^{n \times e},  \\
\mathbf O &= \mathbf G \odot [\mathbf {QKV}] \in \mathbb R^{n \times e} .
\end{aligned}
$$

## Backward

$$
\begin{aligned}
\mathbf {KV}&=\mathbf K^\top \mathbf V \in \mathbb R^{d\times e}, \\
\mathbf {QKV}&= \mathbf Q[\mathbf {KV}]\in \mathbb R^{n \times e},  \\
\mathbf {dG}&= \mathbf {dO} \odot \mathbf {QKV} \in \mathbb R^{n \times e}, \\
\mathbf {dQKV}&= \mathbf {dO}\odot \mathbf {G} \in \mathbb R^{n \times e}, \\
\mathbf {dQ}&= [\mathbf {dQKV}] [\mathbf {KV}] \in \mathbb R^{n\times d}, \\
\mathbf {dKV}&= \mathbf Q^\top [\mathbf{dQKV}]\in \mathbb R^{d\times e},  \\
\mathbf {dK} &= \mathbf V [\mathbf {dKV}^\top]\in \mathbb R^{n\times d},  \\
\mathbf {dV} &= \mathbf K [\mathbf {dKV}]\in \mathbb R^{n\times e}.
\end{aligned}
$$
