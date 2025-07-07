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
\mathbf l_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf l_{t-1}, \\
\mathbf r_t & =  \mathbf r_{t-1}(\mathbf I + \mathbf q_t \mathbf k_t^\top ).
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf l_t  &=  (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf l_{t-1}=\mathbf l_{t-1} + \mathbf q_t (\mathbf k_t^\top  \mathbf l_{t-1}) \triangleq \mathbf l_{t-1} + \mathbf q_t \mathbf u_t^\top ,  \\

\mathbf r_t  &=  \mathbf r_{t-1}  (\mathbf I + \mathbf q_t \mathbf k_t^\top )=\mathbf r_{t-1} + (\mathbf r_{t-1} \mathbf q_t) \mathbf k_t^\top \triangleq \mathbf r_{t-1} + \mathbf h_t \mathbf k_t^\top
\end{aligned}
$$
我们的返回是$\mathbf u_t$和$\mathbf h_t$。

注意到两者恰好为转置的关系，即：
$$
\mathbf r_t^\top   = (\mathbf I + \mathbf q_t \mathbf k_t^\top )^\top  \mathbf r_{t-1}^\top =

(\mathbf I + \mathbf k_t \mathbf q_t ^\top )\mathbf r_{t-1}^\top.
$$
所以我们只讨论$\mathbf l_t$的部分，然后此时用回linear attention的记号，将$\mathbf l_t$替换为$\mathbf s_t$，输出$\mathbf u_t$为$\mathbf o_t$，即为：
$$
\begin{aligned}
\mathbf s_t & = (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf s_{t-1}  \\
& = \mathbf s_{t-1} + \mathbf q_t (\mathbf k_t^\top \mathbf s_{t-1} ) \\
& \triangleq \mathbf s_{t-1} + \mathbf q_t \mathbf u_t^\top, \\

\mathbf u_t^\top  &= \mathbf k_t^\top \mathbf s_{t-1} \\
&=\mathbf k_t^\top \left(
\mathbf s_0 + \sum_{j=1}^{t-1} \mathbf q_j \mathbf u_j^\top
\right).

\end{aligned}
$$
写成矩阵形式即为：
$$
\begin{aligned}
\mathbf U &= \mathrm{tril}([(\mathbf K  \mathbf Q^\top)\odot \mathbf M], -1)  \mathbf U + \mathbf K \mathbf s_0, \\
(\mathbf I - \mathrm{tril}([(\mathbf K  \mathbf Q^\top)\odot \mathbf M], -1)) \mathbf U &=  \mathbf K \mathbf s_0.


\end{aligned}
$$
注意到，通常我们取$\mathbf s_0=\mathbf I$，所以此方程即为kernel regression causal linear改变输入即可得到：

- 输入$\mathbf Q= \mathbf K$；
  - 输入$\mathbf Q'= \mathbf K'$；
  - 输入$ \Alpha=\Beta$；
- 输入$\mathbf K = -\mathbf Q$；
  - 输入$\mathbf K' =\mathbf Q'$；
  - 输入$\Beta=-\mathbf A$；
- 输入$\mathbf V= \mathbf K$；



## 应用

在Path Transformer中，我们需要实现：
$$
\begin{aligned}
 \mathbf s_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf s_{t-1}, \\
\mathbf s_t^{-1} &=  \mathbf s_{t-1}^{-1} (\mathbf I + \mathbf q_t \mathbf k_t^\top )^{-1}, \\
&= \mathbf s_{t-1}^{-1} \left (\mathbf I - \frac{\mathbf q_t \mathbf k_t^\top}{1+ \mathbf q_t^\top \mathbf k_t}  \right) \\
&=   \mathbf s_{t-1}^{-1}  \left (\mathbf I - \frac {\alpha_t\beta_t}{1+ \alpha_t\beta_t\mathbf q_t'^\top \mathbf k_t'}{\mathbf q_t' \mathbf k_t'^\top}  \right).
\end{aligned}
$$
通常$\mathbf Q'=\mathbf K', \|\mathbf q'_t \|^2=1$，此时：
$$
\begin{aligned}
 \mathbf s_t &= (\mathbf I + \mathbf q_t \mathbf k_t^\top ) \mathbf s_{t-1}, \\
\mathbf s_t^{-1} &=  \mathbf s_{t-1}^{-1} (\mathbf I + \mathbf q_t \mathbf k_t^\top )^{-1}, \\
&= \mathbf s_{t-1}^{-1} \left (\mathbf I - \frac{\mathbf q_t \mathbf k_t^\top}{1+ \mathbf q_t^\top \mathbf k_t}  \right) \\
&=   \mathbf s_{t-1}^{-1}  \left (\mathbf I - \frac {\alpha_t\beta_t}{1+ \alpha_t\beta_t}{\mathbf q_t' \mathbf k_t'^\top}  \right).
\end{aligned}
$$
这说明inverse of product of householder也可以使用之前的算法计算，注意上式只有当$\alpha_t \beta_t=-2$时才稳定。





## 特殊情况[Householder矩阵]

我们考虑最特殊的情况，$\mathbf q'_t = \mathbf k'_t\triangleq \mathbf w_t$，$\alpha_t=1, \beta_t= -2$，此时：
$$
 \mathbf s_t = (\mathbf I -2 \mathbf w_t \mathbf w_t^\top ) \mathbf s_{t-1}.
$$
那么通过如下变换即可实现相对位置编码：
$$
\begin{aligned}
\mathbf q_t^\top &= \mathbf q_t^\top  \mathbf s_t,
\mathbf k_t^\top &= \mathbf k_t^\top  \mathbf s_t.

\end{aligned}
$$
