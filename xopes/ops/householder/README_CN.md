# Householder Transform

对于输入$\mathbf x\in \mathbb R^d$，以及向量$\mathbf y\in \mathbb R^d$，我们考虑Houdeholder变换：
$$
\mathbf o=f(\mathbf x, \mathbf y)=
(\mathbf I_d - 2\mathbf y\mathbf y^\top /(\mathbf y^\top \mathbf y))\mathbf x
=\mathbf x-2 (\mathbf y^\top \mathbf x)/(\mathbf y^\top \mathbf y)\mathbf y.
$$


## Forward

输入：$\mathbf x, \mathbf y \in \mathbb R^d$。

计算：
$$
\begin{aligned}
\sigma & =\sqrt{\mathbf y^\top \mathbf y}, \\
\mathbf {\bar y} &=  \mathbf y / \sigma, \\
\mathbf o&=  (\mathbf I_d - 2\mathbf y\mathbf y^\top /(\mathbf y^\top \mathbf y))\mathbf x\\
&=(\mathbf I_d - 2\mathbf {\bar y}\mathbf {\bar y}^\top )\mathbf x \\
&=\mathbf x - 2(\mathbf {\bar y}^\top \mathbf x)\mathbf {\bar y}.
\end{aligned}
$$


## Backward

输入：$\mathbf {do}$。

计算：
$$
\begin{aligned}

\frac{\partial  o_i}{\partial  x_j}
&= 1_{1=j}-2 \bar y_j \bar y_i ,\\
\frac{\partial \mathbf {o}}{\partial \mathbf {x}}

&=\mathbf I - 2 \mathbf {\bar y}\mathbf {\bar y} ^\top ,  \\
\mathbf {dx}&= \left(\frac{\partial \mathbf {o}}{\partial \mathbf {x}} \right)^\top \mathbf {do}\\
&= (\mathbf I - 2 \mathbf {\bar y}\mathbf {\bar y} ^\top)^\top\mathbf {do}\\
&= \mathbf {do} -2 (\mathbf {\bar y} ^\top \mathbf {do})\mathbf {\bar y}.
\end{aligned}
$$
关于另一项的梯度，注意到f：
$$
\begin{aligned}
\frac{\partial  o_i}{\partial  {\bar y}_j}
&= -2\frac{\partial \left( (\sum_{k=1}^d   {\bar y}_k x_k) \bar y_i \right)}{\partial {\bar y}_j} ,\\
&= -2\left(  x_j \bar y_i +( \mathbf {\bar y}^\top \mathbf x) \mathbf 1_{i=j}\right),\\

\frac{\partial \mathbf {o}}{\partial \mathbf {\bar y}}
&=-2 \mathbf {\bar y}\mathbf x^\top - 2\mathbf {\bar y}^\top \mathbf x \mathbf I ,\\

\mathbf {d{\bar y}}&= \left(\frac{\partial \mathbf {o}}{\partial \mathbf {\bar y}}\right)^\top \mathbf {do} \\
&= \left( -2 \mathbf {\bar y}\mathbf x^\top - 2\mathbf {\bar y}^\top \mathbf x \mathbf I  \right)^\top  \mathbf {do}\\
&= -2 (\mathbf{do}^\top \mathbf {\bar y})\mathbf x   - 2(\mathbf {\bar y}^\top \mathbf x)  \mathbf {do}.
\end{aligned}
$$
根据normalize部分的推导可得：
$$
\begin{aligned}
\frac{\partial {\mathbf {\bar y}}}{\partial \mathbf y}
&= 1/\sigma(\mathbf I -\mathbf {\bar y}\mathbf {\bar y}^\top ),  \\
\mathbf {dy}&=\left(\frac{\partial {\mathbf {\bar y}}}{\partial \mathbf y} \right)^\top\mathbf {d \bar y}\\
&=  1/\sigma(\mathbf I -\mathbf {\bar y}\mathbf {\bar y}^\top )\mathbf {d \bar y} \\
&= 1/\sigma (\mathbf {d \bar y}- (\mathbf {\bar y}^\top \mathbf {d \bar y})\mathbf {\bar y}).
\end{aligned}
$$
