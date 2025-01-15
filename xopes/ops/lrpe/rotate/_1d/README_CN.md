
# Lrpe Rotate 1d

## 前向传播

前向传播输入：
$$
\mathbf X \in \mathbb R^{n\times d }, \mathbf \theta \in \mathbb R^{d/2},
\mathrm{offset}\in \mathbb N, \\
\text{act: 激活函数名称}, \text{dim: 激活函数作用维度}.
$$

前向传播输出：
$$
\mathbf O\in \mathbb R^{n\times d}.
$$
计算过程：
$$
\begin{aligned}
{\mathbf {\bar X}}&=f_{\text{act}}(\mathbf X, \mathrm{dim}),\\
\mathbf o_t^\top &=
\bar {\mathbf x}_t^\top \mathbf W_t ,\\
\mathbf W_t(\Theta) &= \text{block-diag}\{\mathbf W_{t,1},\ldots \mathbf W_{t,d/2} \} \in \mathbb R^{d\times d},  \\
\mathbf W_{t,k}&= \left[\begin{array}{cc}
\cos \left((t+\mathrm{offset}) \theta_k\right) & -\sin \left((t+\mathrm{offset})\theta_k\right) \\
\sin \left((t+\mathrm{offset})\theta_k\right) & \cos \left((t+\mathrm{offset}) \theta_k\right)
\end{array}\right]\in \mathbb R^{2\times 2}.
\end{aligned}
$$



## 反向传播

反向传播输入：
$$
\begin{aligned}
\mathbf {dO} \in \mathbb R^{n\times d}.
\end{aligned}
$$
反向传播输出：
$$
\mathbf {dX}\in \mathbb R^{n\times d}.
$$
其中，$n$ 表示序列长度，$d$ 表示haed dim，$\mathrm{offset}$表示偏移量（仅在语言模型的推理阶段使用）。

计算过程：
$$
\begin{aligned}
\mathbf {dx}^\top_t &=
 {\mathbf o}_t^\top \mathbf W_t^\top   \\
  &={\mathbf o}_t^\top \mathbf W_t^\top   \\
 &= {\mathbf o}_t^\top \mathbf W_t(\Theta)^\top,   \\
  &= {\mathbf o}_t^\top \mathbf W_t(-\Theta),   \\
\mathbf {d X}& = f'_{\text{act}}(\mathbf{d{\bar X}}, \mathrm{dim}).
\end{aligned}
$$



## 补充

在实现的时候，我们还会支持如下两个功能：

- 因为有head的概念，所以我们会支持$\theta$形状为$(h, d/2), (d/2),(h)$；
- 我们允许只对部分的head dim只用rope操作；
