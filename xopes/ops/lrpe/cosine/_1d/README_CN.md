# Lrpe Cosine 1d

## 符号与输入

- `lrpe_cosine_torch`：Lrpe Cosine 1d 的 Pytorch 版本。
- `lrpe_cosine_1d_sp_triton`, `lrpe_cosine_1d_sp_fwd_triton`, `lrpe_cosine_1d_sp_bwd_triton`：
  - 针对序列进行并行计算，适用于 `dim != -2` 的情况。
- `lrpe_cosine_1d_bp_triton`, `lrpe_cosine_1d_bp_fwd_triton`, `lrpe_cosine_1d_bp_bwd_triton`：
  - 按块单位进行并行计算，适用于 `dim = -2` 的情况。

## 前向传播

前向传播输入：
$$
\mathbf X \in \mathbb R^{n\times d}, \mathbf \theta \in \mathbb R^{d},
\mathrm{offset}\in \mathbb N, \\
\text{act: 激活函数名称}, \text{dim: 激活函数作用维度}.
$$

前向传播输出：
$$
\mathbf O\in \mathbb R^{n\times 2d}.
$$
计算过程：
$$
\begin{aligned}
\Theta_{tk} &= (\mathrm{offset}+t) \theta_{k} , \Theta_t\in \mathbb R^{d}, \\
{\mathbf {\bar X}}&=f_{\text{act}}(\mathbf X, \mathrm{dim}),\\
\mathbf O &=\mathrm{concat}([\mathbf {\bar X}  \odot \cos(\Theta),\mathbf {\bar X}  \odot  \sin(\Theta)])
\in \mathbb R^{n\times 2d}.
\end{aligned}
$$



## 反向传播

反向传播输入：
$$
\begin{aligned}
\mathbf {dO} \in \mathbb R^{n\times 2d}.
\end{aligned}
$$
反向传播输出：
$$
\mathbf {dX}\in \mathbb R^{n\times d}.
$$
其中，$n$ 表示序列长度，$d$ 表示head dim，$\mathrm{offset}$ 表示偏移量（仅在语言模型的推理阶段使用）。

计算过程：
$$
\begin{aligned}
&\Theta_{tk} = (\mathrm{offset}+t) \theta_{k} , \Theta_t\in \mathbb R^{ d}, \\
&\mathbf{dO} =\mathrm{concat}[\mathbf{dO}_{\cos},\mathbf{dO}_{\sin}],\\
&\mathbf{dO}_{\cos},\mathbf{dO}_{\sin} \in \mathbb R^{n\times d},  \\
&\mathbf{d{\bar X}} = \mathbf{dO}_{\cos} \odot \cos(\Theta) + \mathbf{dO}_{\sin}\odot \sin( \Theta), \\
&\mathbf {d X} = f'_{\text{act}}(\mathbf{d{\bar X}}, \mathrm{dim}).
\end{aligned}
$$



## 补充

在实现的时候，因为有head的概念，所以我们会支持$\theta$形状为$(h, d)$和$(d)$两个版本。