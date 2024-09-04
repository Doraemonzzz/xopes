# Lrpe Cosine

## Notation and Input

$$
\begin{aligned}
x &: h, n, d,\\
\theta &: h, d, \\
do&: h, n, 2d.
\end{aligned}
$$

Where $h$ denotes the number of heads, $n$ denotes the sequence length, and $d$ denotes the head dimension.

## Forward

$$
\begin{aligned}
\bar \theta_{i, j, k} &= \theta_{ij}k ,  \\
o&=\mathrm{concat}([x \odot \cos(\theta), x\odot \sin(\theta)]).
\end{aligned}
$$

## Backward

$$
\begin{aligned}
do &=\mathrm{concat}[do_{\cos}, do_{\sin}],\\
dx &= do_{\cos} \odot \cos(\bar \theta) + do_{\sin}\odot \sin(\bar \theta).
\end{aligned}
$$
