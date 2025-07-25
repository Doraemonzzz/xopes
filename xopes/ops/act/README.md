# Activation function

## Notation

Input: $\mathbf x\in \mathbb R^d$.

Output: $\mathbf o\in \mathbb R^d$.



## Relu

### Fwd

$$
\mathbf {o} =\mathbf x\odot \mathbf 1_{\mathbf x\ge 0}.
$$



### Bwd

$$
\mathbf {dx} = \mathbf {do} \odot \mathbf 1_{\mathbf x\ge 0}.
$$



## Sigmoid

### Fwd

$$
\mathbf o = \mathrm{Sigmoid}(\mathbf x).
$$



### Bwd

$$
\begin{aligned}
\mathbf {dx}
&=\mathbf {do} \odot
 \mathrm{Sigmoid}(\mathbf x) \odot (1- \mathrm{Sigmoid}(\mathbf x)).

\end{aligned}
$$





## Silu/Swish

### Fwd

$$
\mathbf o =\mathbf x\odot \mathrm{Sigmoid}(\mathbf x).
$$



### Bwd

$$
\begin{aligned}
\mathbf {dx}
&=\mathbf {do} \odot
[\mathrm{Sigmoid}(\mathbf x) + \mathrm{Sigmoid}(\mathbf x)\odot \mathbf x \odot (1- \mathrm{Sigmoid}(\mathbf x))] \\
&=\mathbf {do}\odot \mathrm{Sigmoid}(\mathbf x)
\odot [1+ \mathbf x \odot (1- \mathrm{Sigmoid}(\mathbf x))].

\end{aligned}
$$



## Softmax

### Fwd

$$
\mathbf o_s =\frac{\exp(x_s)}{\sum_{t=1}^d \exp(x_t)}.
$$



### Bwd

$$
\begin{aligned}
\frac{\partial o_t}{\partial x_s}
&= \frac{\partial \frac{\exp(x_t)}{\sum_{k=1}^d \exp(x_k)}}{\partial x_s} \\
&= \frac{\exp(x_t)}{\sum_{k=1}^d \exp(x_k)} \mathbf 1_{s=t}- \frac{\exp(x_t)\exp(x_s)}{\left(\sum_{k=1}^d \exp(x_k)\right)^2}  \\

\mathbf {dx}_s
& = \frac{\partial l}{\partial x_s} \\
&= \sum_{t=1}^ d
\frac{\partial l}{\partial o_t}
\frac{\partial o_t}{\partial x_s}
\\
&= \sum_{t=1}^ d
\frac{\partial l}{\partial o_t}

\left(
 \frac{\exp(x_t)}{\sum_{k=1}^d \exp(x_k)} \mathbf 1_{s=t}- \frac{\exp(x_t)\exp(x_s)}{\left(\sum_{k=1}^d \exp(x_k)\right)^2}
\right) \\
&=
o_s\frac{\partial l}{\partial o_s}  -o_s\sum_{t=1}^ d \frac{\partial l}{\partial o_t} o_t  \\
\mathbf {dx}&=  \mathbf o \odot \mathbf {do} - \mathbf o  (\mathbf {do}^{\top} \mathbf o).
\end{aligned}
$$
