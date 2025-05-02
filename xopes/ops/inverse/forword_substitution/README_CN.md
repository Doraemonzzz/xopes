# Matrix inverse forword substitution

给定矩阵$\mathbf A\in \mathbb R^{n\times n}$，其中：
$$
\mathbf A_{ij}=0, i > j.
$$
我们的目标是计算$\mathbf A^{-1}$。

根据定义可得：
$$
\sum_{j=1}^i a_{ij} x_{jk}= 1_{i=k}.
$$
向量化该方程即为：
$$
a_{ii} x_{ik} =1_{i=k}-\sum_{j=1}^{i-1} a_{ij} x_{jk}, \\
\mathbf x_i^\top  = \mathbf e_{i}^\top / a_{ii} - 1/a_{ii}\sum_{j=1}^{i-1} a_{ij} \mathbf x_{j}
=\mathbf e_{i}^\top  / a_{ii}-  1/a_{ii} [\mathbf a_i[:i-1]^\top \mathbf  X[:i-1, :]].
$$
注意到如果我们不需要后续使用$\mathbf A$，则可以使用inplace操作，这是因为，为了计算$\mathbf x_i$，我们需要$\mathbf A$的第$i$行，而$\mathbf X[:i-1, :]$可以存储在$\mathbf A $的前$i-1$行。
