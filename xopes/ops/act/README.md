# Activation function



## Relu

### Fwd

$$
\mathbf O =\mathbf X\odot \mathbf 1_{\mathbf X\ge 0}.
$$



### Bwd

$$
\mathbf {dX} = \mathbf {dO} \odot \mathbf 1_{\mathbf X\ge 0}.
$$





## Sigmoid

### Fwd

$$
\mathbf O = \mathrm{Sigmoid}(\mathbf X).
$$



### Bwd

$$
\begin{aligned}
\mathbf {dX}
&=\mathbf {dO} \odot
 \mathrm{Sigmoid}(\mathbf X) \odot (1- \mathrm{Sigmoid}(\mathbf X))

\end{aligned}
$$





## Silu/Swish

### Fwd

$$
\mathbf O =\mathbf X\odot \mathrm{Sigmoid}(\mathbf X).
$$



### Bwd

$$
\begin{aligned}
\mathbf {dX}
&=\mathbf {dO} \odot
[\mathrm{Sigmoid}(\mathbf X) + \mathrm{Sigmoid}(\mathbf X)\odot \mathbf X \odot (1- \mathrm{Sigmoid}(\mathbf X))] \\
&=\mathbf {dO}\odot \mathrm{Sigmoid}(\mathbf X)
\odot [1+ \mathbf X \odot (1- \mathrm{Sigmoid}(\mathbf X))]

\end{aligned}
$$
