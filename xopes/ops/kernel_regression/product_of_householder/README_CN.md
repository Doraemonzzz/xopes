# Product of householder

给定输入$\mathbf Q'\in \mathbb R^{n\times d}, \mathbf K'\in \mathbb R^{n\times d},\mathbf \Alpha, \mathbf \Beta \in \mathbb R^n$，记：
$$
\begin{aligned}
\mathbf M_{ij} &=
\begin{cases}
1, & i \ge j, \\
0, & i < j.
\end{cases} \\

\mathbf Q &=  \mathbf Q' \odot \mathbf \Alpha, \\

\mathbf K &= \mathbf K' \odot \mathbf \Beta.

\end{aligned}
$$



## Forward

考虑递推公式：
$$
\begin{aligned}
\mathbf L_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf L_{t-1}, \\
\mathbf R_t & =  \mathbf R_{t-1}(\mathbf I + \mathbf q_t \mathbf k_t^\top ).
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf L_t  &=  (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf L_{t-1}=\mathbf L_{t-1} + \mathbf q_t (\mathbf k_t^\top  \mathbf L_{t-1}) \triangleq \mathbf L_{t-1} + \mathbf q_t \mathbf u_t^\top ,  \\

\mathbf R_t  &=  \mathbf R_{t-1}  (\mathbf I + \mathbf q_t \mathbf k_t^\top )=\mathbf R_{t-1} + (\mathbf R_{t-1} \mathbf q_t) \mathbf k_t^\top \triangleq \mathbf R_{t-1} + \mathbf h_t \mathbf k_t^\top
\end{aligned}
$$
我们的返回是$\mathbf u_t$和$\mathbf h_t$。

注意到两者恰好为转置的关系，即：
$$
\mathbf R_t^\top   = (\mathbf I + \mathbf q_t \mathbf k_t^\top )^\top  \mathbf R_{t-1}^\top =

(\mathbf I + \mathbf k_t \mathbf q_t ^\top )\mathbf R_{t-1}^\top.
$$
所以我们只讨论$\mathbf L_t$的部分，然后此时用回linear attention的记号，将$\mathbf L_t$替换为$\mathbf S_t$，输出$\mathbf u_t$为$\mathbf o_t$，即为：
$$
\mathbf S_t = (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf S_{t-1}.
$$
此方程即为kernel regression causal linear版本中$\mathbf V=\mathbf 0$的版本，因此解为：
$$
\begin{aligned}
\ [\mathbf I + \mathrm{tril}([(\mathbf Q  \mathbf K^\top)\odot \mathbf M], -1) ]  \mathbf O   &= \mathbf V, \\

\mathbf V &= \mathbf 0.
\end{aligned}
$$

所以可以复用之前的算子。



## 应用

在Path Transformer中，我们需要实现：
$$
\begin{aligned}
 \mathbf S_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf S_{t-1}, \\
\mathbf S_t^{-1} &=  \mathbf S_{t-1}^{-1} (\mathbf I + \mathbf q_t \mathbf k_t^\top )^{-1}, \\
&= \mathbf S_{t-1}^{-1} \left (\mathbf I - \frac{\mathbf q_t \mathbf k_t^\top}{1+ \mathbf q_t^\top \mathbf k_t}  \right) \\
&=   \mathbf S_{t-1}^{-1}  \left (\mathbf I - \frac {\alpha_t\beta_t}{1+ \alpha_t\beta_t\mathbf q_t'^\top \mathbf k_t'}{\mathbf q_t' \mathbf k_t'^\top}  \right).
\end{aligned}
$$
通常$\mathbf Q'=\mathbf K', \|\mathbf q'_t \|^2=1$，此时：
$$
\begin{aligned}
 \mathbf S_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf S_{t-1}, \\
\mathbf S_t^{-1} &=  \mathbf S_{t-1}^{-1} (\mathbf I + \mathbf q_t \mathbf k_t^\top )^{-1}, \\
&= \mathbf S_{t-1}^{-1} \left (\mathbf I - \frac{\mathbf q_t \mathbf k_t^\top}{1+ \mathbf q_t^\top \mathbf k_t}  \right) \\
&=   \mathbf S_{t-1}^{-1}  \left (\mathbf I - \frac {\alpha_t\beta_t}{1+ \alpha_t\beta_t}{\mathbf q_t' \mathbf k_t'^\top}  \right).
\end{aligned}
$$
这说明inverse of product of householder也可以使用之前的算法计算。
