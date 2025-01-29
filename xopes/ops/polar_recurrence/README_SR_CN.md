# Polar Recurrence(Sequential Recurrence)

给定输入$\mathbf Q\in \mathbb R^{n\times d_1}, \mathbf \Alpha \in \mathbb R^{n\times d_1}, \mathbf \Beta\in \mathbb R^{n\times d_2}, \mathbf R \in \mathbb R^{n\times d_2}, \mathbf S\in \mathbb R^{n\times d_3}$，初起始state $\mathbf u_0, \mathbf p_0$，以及模长参数$\Gamma \in \mathbb R^{n\times }$和Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf u_0&\in \mathbb R^{d_1\times d_2}, \\
\mathbf p_0&\in \mathbb R^{d_2\times d_3}, \\
\alpha_i & = \gamma_i \odot \alpha_i,   \\
\mathbf u_i &= (\mathbf I + \alpha_i \beta_i^\top) \mathbf u_{i-1}, \\
\mathbf p_i &= \mathrm{diag}\{\lambda_i\} \mathbf p_{i-1} +\mathbf s_i \mathbf t_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top \mathbf u_i  \mathbf p_i.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d_3}.
$$
细节：

- $\mathbf A, \mathbf B$需要做l2 norm；
- $\mathbf R, \mathbf S$最好也做l2 norm；
- $\lambda_i$也可以是全$1$，或者标量；

备注：在实际使用的时候，$d_1=d_2$，因为有矩阵连乘，这里标注不同是为了方便验证矩阵求导的维度。



## Forward

我们执行如下递归：

为了方便后续讨论，我们假设$\alpha_i$已经scale过，即：
$$
\begin{aligned}
\alpha_i & = \gamma_i \alpha_i.
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf u_0&\in \mathbb R^{d_1\times d_2}, \\
\mathbf p_0&\in \mathbb R^{d_2\times d_3}, \\

\mathbf u_i &= (\mathbf I + \alpha_i\beta_i^\top  ) \mathbf u_{i-1} \in \mathbb R^{d_1\times d_2}, \\
\mathbf \eta_i &=   \mathbf u_{i-1}^\top \beta_i \in \mathbb R^{d_2},\\
\mathbf u_i &= (\mathbf I + \alpha_i\beta_i^\top) \mathbf u_{i-1} \\
&= \mathbf u_{i-1} + \alpha_i\eta_i^\top , \\
\mathbf p_i &= \mathrm{diag}\{\lambda_i\} \mathbf p_{i-1} +\mathbf r_i \mathbf s_i^\top \in \mathbb R^{d_2\times d_3}, \\
\mathbf h_i&=  \mathbf u_i^\top \mathbf q_i \in \mathbb R^{d_2},  \\
\mathbf o_i &= \mathbf p_i^\top \mathbf h_i\in \mathbb R^{d_3}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d_3}.
$$



## Backward

给定$\mathbf {do}_1,\ldots, \mathbf {do}_n, \mathbf {du}_n, \mathbf {dp}_n$，计算：
$$
\begin{aligned}
\mathbf {dp}_n & = \mathbf {dp}_n  +  \mathbf {h}_n \mathbf{do}_n^\top \in \mathbb R^{d_2\times d_3},   \\

\mathbf {du}_n & = \mathbf {du}_n  +  \mathbf {q}_n \mathbf{dh}_n^\top  \in \mathbb R^{d_1\times d_2},  \\

\mathbf {dp}_i & = \mathrm{diag}\{\lambda_{i+1}\} \mathbf {dp}_{i+1}  +  \mathbf {h}_i \mathbf{do}_i^\top,    \in \mathbb R^{d_2\times d_3} \\

\mathbf {du}_i
& = \mathbf {du}_{i+1}  +  \mathbf {q}_i \mathbf{dh}_i^\top  \in \mathbb R^{d_1\times d_2},  \\

\mathbf {dh}_i &=   \mathbf {p}_i \mathbf {do}_i \in \mathbb R^{d_2},  \\

\mathbf {dq}_i &=   \mathbf {u}_i \mathbf {dh}_i  \in \mathbb R^{d_1}, \\

\mathbf {dr}_i &= \mathbf {dp}_i \mathbf s_i \in \mathbb R^{d_2}, \\

\mathbf {ds}_i &= \mathbf {dp}_i^\top \mathbf s_i \in \mathbb R^{d_3}, \\

\mathbf {d\alpha}_i
& = \mathbf {du}_i  \eta_i \in \mathbb R^{d_1},  \\

\mathbf {d\eta}_i
& = \mathbf {du}_i^\top  \alpha_i\in \mathbb R^{d_2},  \\

\mathbf {d\beta}_i
& = \mathbf {u}_{i-1}  \mathbf {d\eta}_i^\top \in \mathbb R^{d_2} . \\
\end{aligned}
$$


### 加速方案(TODO)

我们考虑递推：
$$
\begin{aligned}
\mathbf r_i &= (\mathbf I + \beta_i \mathbf k_i \mathbf k_i^\top) \mathbf  r_{i-1},  \\
\mathbf  r_{0} &= \mathbf I, \\
\mathbf u_i &= \mathbf r_i  \mathbf u_0.
\end{aligned}
$$
下面用归纳法证明：
$$
{\mathbf r_n }=\mathbf I + \sum_{i=1}^n \mathbf k_i \mathbf w_i^\top .
$$
$n=1$时结论成立，假设$n-1$时结论成立，那么$n$时：
$$
\begin{aligned}
{\mathbf {r}_n }
&=  (\mathbf I + \beta_n \mathbf k_n \mathbf k_n^\top) {\mathbf {r}}_{n-1}\\
&=\mathbf r_{n-1} +  \beta_n \mathbf k_n \mathbf k_n^\top \mathbf r_{n-1} \\

&= \mathbf I + \sum_{i=1}^{n-1} \mathbf k_i \mathbf w_i^\top
+ \beta_n \mathbf k_n \mathbf k_n^\top
\left(\mathbf I +\sum_{i=1}^{n-1} \mathbf k_i \mathbf w_i^\top\right) \\

&=  \mathbf I + \sum_{i=1}^{n-1} \mathbf k_i \mathbf w_i^\top + \mathbf k_n \left(
\beta_n \mathbf k_n  +\beta_n\sum_{i=1}^{n-1} \mathbf w_i(\mathbf k_i^\top \mathbf k_n)
\right)^\top \\
&\triangleq \mathbf I+\sum_{i=1}^{n} \mathbf k_i \mathbf w_i^\top.

\end{aligned}
$$
其中：
$$
\mathbf w_n=\beta_n \mathbf k_n  +\beta_n\sum_{i=1}^{n-1} \mathbf w_i(\mathbf k_i^\top \mathbf k_n) .
$$
