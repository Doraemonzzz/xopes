# Element-wise binary op

对于输入$\mathbf x\in \mathbb R^{n_1\times \ldots n_k \times n_{k+1}\times \ldots \times n_{k+m}}$和$\mathbf y\in \mathbb R^{n_1\times \ldots n_k}$以及二元运算符$f$，我们执行如下操作：
$$
\mathbf o_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}= f(\mathbf x_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}},
\mathbf y_{i_1,\ldots, i_k}).
$$


## Forward

输入：$\mathbf x\in \mathbb R^{n_1\times \ldots n_k \times n_{k+1}\times \ldots \times n_{k+m}}$和$\mathbf y\in \mathbb R^{n_1\times \ldots n_k}$以及二元运算符$f$。

计算：
$$
\mathbf o_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}= f(\mathbf x_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}},
\mathbf y_{i_1,\ldots, i_k}).
$$


## Backward

输入：$\mathbf{do}\in \mathbb R^{n_1\times \ldots n_k \times n_{k+1}\times \ldots \times n_{k+m}}$。

计算：
$$
\begin{aligned}
\mathbf {dx}_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}
&= \mathbf {di}_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}\odot  \frac{\partial \mathbf o_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}}
{\partial \mathbf x_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}} }, \\

\mathbf {dy}_{i_1,\ldots, i_k}
&= \sum_{i_{k+1},\ldots, i_{k+m}}\mathbf {di}_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}\odot  \frac{\partial \mathbf o_{i_1,\ldots, i_k, i_{k+1},\ldots, i_{k+m}}}
{\partial \mathbf y_{i_1,\ldots, i_k, i_{k}} }.
\end{aligned}
$$


## 补充

经过试验发现，`torch.compile`足够给出一个比较好的结果。
