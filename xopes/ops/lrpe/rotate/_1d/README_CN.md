
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

注意到：
$$
\begin{aligned}

\left[\begin{array}{cc}
\cos \left(\theta\right) & -\sin \left(\theta\right) \\
\sin \left(\theta\right) & \cos \left(\theta\right)
\end{array}\right]\left[\begin{array}{cc}
x_1 \\ x_2
\end{array}\right]
&=
\left[\begin{array}{cc}
x_1\cos \theta -x_2\sin\theta ,  x_2\cos\theta+x_1\sin \theta
\end{array}\right]

\end{aligned}
$$
而对特征做置换不改变结果以及实现便利性，我们假设：
$$
\mathbf X = [\mathbf X_1,\mathbf X_2], \mathbf X_i\in \mathbb R^{n\times d/2}.
$$
那么最终的输出为：
$$
\begin{aligned}
\mathbf O_1 &= \mathbf X_1 \odot \cos\theta - \mathbf X_2 \odot \sin \theta ,\\
\mathbf O_2 &= \mathbf X_1 \odot \sin\theta +\mathbf X_2 \odot \cos \theta, \\
\mathbf O&= [\mathbf O_1 , \mathbf O_2].
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

我们假设：
$$
\mathbf {dO}= [\mathbf {dO}_1,\mathbf {dO}_2], \mathbf {dO}_i\in \mathbb R^{n\times d/2}.
$$
那么最终的输出为：
$$
\begin{aligned}
\mathbf {dX}_1 &= \mathbf {dO}_1 \odot \cos\theta + \mathbf {dO}_2 \odot \sin \theta ,\\
\mathbf {dX}_2 &= -\mathbf {dO}_1\odot \sin\theta +\mathbf {dO}_2 \odot \cos \theta, \\
\mathbf {dO}&= [\mathbf {dO}_1,\mathbf {dO}_2].
\end{aligned}
$$



## 补充

在实现的时候，我们还会支持如下两个功能：

- 因为有head的概念，所以我们会支持$\theta$形状为$(h, d/2), (d/2),(h)$；
- 我们允许只对部分的head dim只用rope操作；
