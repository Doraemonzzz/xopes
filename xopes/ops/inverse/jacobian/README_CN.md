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


### 分块求逆

假设已知$\mathbf A_{ij}^{-1}, j \le i$，那么：
$$
\begin{aligned}
\sum_{j=k}^{i+1} \mathbf A_{i+1,j} \mathbf A^{-1}_{jk} &= \mathbf 0, k < i+1,  \\

\mathbf A_{i+1, k}^{-1} & =-\mathbf A_{i+1, i+1}^{-1} \left( \sum_{j=k}^{i} \mathbf A_{i+1, j} \mathbf A^{-1}_{jk}\right).

\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf A_{21}^{-1} &= -\mathbf A_{22}^{-1} \mathbf A_{21} \mathbf A_{11}^{-1}, \\
\mathbf A_{32}^{-1} &=- \mathbf A_{33}^{-1} \mathbf A_{32} \mathbf A_{22}^{-1}, \\
\mathbf A_{31}^{-1} &=- \mathbf A_{33}^{-1} \left(
\mathbf A_{31} \mathbf A_{11}^{-1}  +
\mathbf A_{32} \mathbf A_{21}^{-1} \right), \\
\mathbf A_{43}^{-1} &= - \mathbf A_{44}^{-1} \mathbf A_{43} \mathbf A_{33}^{-1},  \\

\mathbf A_{42}^{-1} &= - \mathbf A_{44}^{-1}
\left(
\mathbf A_{42} \mathbf A_{22}^{-1} + \mathbf A_{43} \mathbf A_{32}^{-1}
\right), \\

\mathbf A_{41}^{-1} &= - \mathbf A_{44}^{-1}
\left(
\mathbf A_{41} \mathbf A_{11}^{-1} + \mathbf A_{42} \mathbf A_{21}^{-1}+ \mathbf A_{43} \mathbf A_{31}^{-1}
\right).

\end{aligned}
$$
