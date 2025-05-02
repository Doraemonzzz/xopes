# Matrix inverse Jacobian

给定矩阵$\mathbf A\in \mathbb R^{n\times n}$，其中：
$$
\mathbf A_{ij}=0, i > j.
$$
我们的目标是计算$\mathbf A^{-1}$。

我们记：
$$
\mathbf L = \mathrm{tril}(\mathbf A), \Lambda = \mathrm{diag}(\mathbf A).
$$
注意到：
$$
\begin{aligned}
&\mathbf A \mathbf x = \mathbf b, \\
\Leftrightarrow & (\Lambda +\mathbf L) \mathbf x = \mathbf b, \\
\Leftrightarrow & (\mathbf I - (-\Lambda^{-1} \mathbf L) ) \mathbf x = \Lambda ^{-1}\mathbf b .
\end{aligned}
$$
最后一行记为：
$$
\begin{aligned}
(\mathbf I - \mathbf L_1) \mathbf x &= \mathbf b_1 , \\
\mathbf x &= (\mathbf I - \mathbf L_1)^{-1} \mathbf b_1 \\

&= \sum_{k=0}^\infty  \mathbf L_1^k \mathbf b_1,  \\

\mathbf x_k &= \mathbf L_1 \mathbf x_{k-1} + \mathbf b_1.

\end{aligned}
$$
