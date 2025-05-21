# 参数化

这一部分，我们讨论如何参数化vector decay $\mathbf D_t$:
$$
\mathbf D_t=\mathrm{diag}(\lambda_t) + \mathbf a_t \mathbf b_t^\top.
$$
首先根据之前的分析，我们可以引入变量$\mathbf c_t$，使得：
$$
\mathbf b_t = \mathbf c_t \odot \lambda_t.
$$
那么：
$$
\begin{aligned}
\mathbf D_t &= \mathrm{diag}(\lambda_t) + \mathbf a_t \mathbf b_t^\top \\
&= \mathrm{diag}(\lambda_t) + \mathbf a_t \mathbf c_t^\top \mathrm{diag}(\lambda_t) \\
&= (\mathbf I_d + \mathbf a_t \mathbf c_t^\top) \mathrm{diag}(\lambda_t).
\end{aligned}
$$
接下来讨论稳定性的问题，要使得Linear Attention的稳定性成立，需要$\mathbf D_t$的特征值的绝对值需要$\le 1$，因为Decay $\lambda_t$的绝对值小于等于1，所以我们只需要考虑下式的特征值即可：
$$
\mathbf E_t = \mathbf I_d + \mathbf a_t \mathbf c_t^\top.
$$
根据线代的知识可得，
$$
\begin{aligned}
\mathbf E_t \mathbf a_t &= (1+\mathbf c_t^\top \mathbf a_t) \mathbf a_t, \\
\mathbf E_t \mathbf u_t &= \mathbf u_t, \\
\mathbf a_t^\top \mathbf u_t &= 0.
\end{aligned}
$$
$\mathbf E_t$有$d-1$个特征值为1，1个特征值为$1+\mathbf c_t \mathbf a_t^\top$，因此我们希望：
$$
-1 \le 1+\mathbf c_t \mathbf a_t^\top \le 1, \\
-2 \le \mathbf c_t \mathbf a_t^\top \le 0.
$$
为了满足第二个式子，我们只能取：
$$
\begin{aligned}
\mathbf c_t &= -\beta_t \mathbf a_t, \\
-2 &\le -\beta_t \| \mathbf a_t \|_2^2 \le 0.
\end{aligned}
$$
为了简化记号，我们假设$\| \mathbf a_t \|_2^2 = 1$，那么：
$$
-2 \le -\beta_t \le 0, \\
0 \le \beta_t \le 2.
$$
综上，参数化的方案为：
$$
\begin{aligned}

\mathbf c_t &= -\beta_t \mathbf a_t, \\
\mathbf a_t^\top \mathbf a_t &= 1, \\
\mathbf D_t &= (\mathbf I_d - \beta_t \mathbf a_t \mathbf a_t^\top) \mathrm{diag}(\lambda_t), \\
\beta_t &\in [0, 2].
\end{aligned}
$$
