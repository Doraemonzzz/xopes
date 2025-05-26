# Inverse attention

## Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n}$，记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
0, & i < j.
\end{cases}
$$
我们考虑给定$\mathbf Q, \mathbf K, \mathbf O, \mathbf A, \mathbf S_0$，求解$\mathbf V$：
$$
\begin{aligned}
\mathbf V &=\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right]^{-1} \left( \mathbf O - \Alpha \mathbf Q \mathbf S_0 \right)  \\
&\triangleq \left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right]^{-1}  \mathbf O.
\end{aligned}
$$

### 递推形式
不稳定版本发现$\mathbf s_t, \mathbf v_t$ 会爆炸：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top, \\

\mathbf k_t \mathbf v_t^\top &= \mathbf s_t - \lambda_t \mathbf s_{t-1}, \\
\mathbf q_t^\top \mathbf k_t \mathbf v_t^\top &=  \mathbf q_t^\top \mathbf s_t - \lambda_t \mathbf q_t^\top \mathbf s_{t-1}  \\
&=  \mathbf o_t^\top -\lambda_t \mathbf q_t^\top \mathbf s_{t-1}, \\

\mathbf v_t^\top &= \left(
\mathbf o_t^\top  - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}
\right) / \left(\mathbf q_t^\top  \mathbf k_t  \right).

\end{aligned}
$$
回顾上式可能有两个地方带来问题：
1. $\mathbf s_t$可能是input部分$\mathbf k_t \mathbf v_t^\top$ 的累加，因此$\mathbf s_t$ 会爆炸：
2. $\mathbf v_t^\top$可能是除法部分的问题：
因此我们做如下修改：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + (1-\lambda_t)\mathbf k_t \mathbf v_t^\top, \\

\mathbf v_t^\top &=
\mathbf o_t^\top  - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}.

\end{aligned}
$$
我们将其恢复成原始的Linear Attention，即为：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} +(1-\lambda_t) \mathbf k_t \mathbf v_t^\top, \\
\mathbf o_t^\top &=
\mathbf v_t^\top  + \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}.
\end{aligned}
$$
写成矩阵形式即为：
$$
\begin{aligned}
\mathbf V &=\left[
\mathbf I + \mathrm{tril}(\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M, -1)
\right]^{-1} \left( \mathbf O - \Alpha \mathbf Q \mathbf S_0 \right)  \\
&\triangleq \left[
\mathbf I + \mathrm{tril}(\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M, -1)
\right]^{-1}  \mathbf O.
\end{aligned}
$$
另一方面，将$\mathbf v_t$的定义代入到递推式中可得：
$$
\begin{aligned}
\mathbf s_t &=
\lambda_t \mathbf s_{t-1} + (1-\lambda_t) \mathbf k_t \mathbf v_t^\top, \\
 &=
\lambda_t \mathbf s_{t-1} + (1-\lambda_t) \mathbf k_t
\left(
\mathbf o_t^\top  - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}
\right)  \\
&=
\lambda_t (\mathbf I - (1-\lambda_t) \mathbf k_t \mathbf q_t^\top ) \mathbf s_{t-1}
+ (1-\lambda_t) \mathbf k_t \mathbf o_t^\top.
\end{aligned}
$$
我们希望$\mathbf I - (1-\lambda_t) \mathbf k_t^\top \mathbf q_t$特征值的绝对值小于等于$1$，注意到该矩阵有$d-1$个$1$特征值，最后一个特征值为：
$$
1-(1-\lambda_t) \mathbf k_t^\top \mathbf q_t\in
[1-(1-\lambda_t)c, 1+(1-\lambda_t)c],
|\mathbf k_t^\top \mathbf q_t| \le c.
$$
整个转移矩阵的特征值为$d-1$个$\lambda_t$特征值，最后一个特征值的范围为：
$$
[\lambda_t-\lambda_t(1-\lambda_t)c, \lambda_t+\lambda_t(1-\lambda_t)c].
$$
如果$c=1$，则特征值的范围为：
$$
[\lambda_t^2, \lambda_t(2-\lambda_t)] \in [\lambda_t^2, 1].
$$
此时符合稳定性的约束。

更一般的，上下界的二次函数分别为：
$$
\begin{aligned}
\lambda_t+\lambda_t(1-\lambda_t)c &= -c\lambda_t^2 +(1+c)\lambda_t \le \frac{(1+c)^2}{4c}, \\
\lambda_t-\lambda_t(1-\lambda_t)c &= c\lambda_t^2 +(1-c)\lambda_t.
\end{aligned}
$$
因为：
$$
\frac{(1+c)^2}{4c} \ge 1,
$$
所以要使得系统稳定必然有$c=1$。



## Backward

根据之前的递推，我们有：
$$
\begin{aligned}
\mathbf p_t &\triangleq (1-\lambda_t) \mathbf {ds}_{t}^\top \mathbf k_t, \\

\mathbf{ds}_{t} &= \lambda_{t+1} (\mathbf I - (1-\lambda_{t+1}) \mathbf k_{t+1} \mathbf q_{t+1}^\top )^\top
\mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\
&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1} (1-\lambda_{t+1}) \mathbf q_{t+1} \left[\mathbf k_{t+1}^\top \mathbf{ds}_{t+1}\right]
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1} (1-\lambda_{t+1}) \mathbf q_{t+1} \left[ \mathbf{ds}_{t+1}^\top \mathbf k_{t+1}\right]^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} \mathbf p_{t+1}^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} (\mathbf p_{t+1} + \mathbf {dv}_{t+1})^\top,

\\
&t=1,\ldots, n- 1, \\
\mathbf {ds}_0&=  \lambda_1 (\mathbf I - (1-\lambda_{1}) \mathbf q_{1} \mathbf k_{1}^\top ) \mathbf {ds}_1 \\


&=\lambda_1 \mathbf{ds}_1 - \lambda_1 \mathbf q_{1} \mathbf p_1^\top,  \\


\mathbf {do}_t^\top &= \frac{\partial l}{\partial \mathbf v_t}^\top \frac{\partial \mathbf v_t}{\partial \mathbf o_t} +  \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf o_t} \\

&= \mathbf {dv}_t^\top + (1- \lambda_t) \mathbf {ds}_t^\top  \mathbf k_t \\

&= \mathbf {dv}_t^\top + \mathbf p_t^\top ,\\

\mathbf {dk}_t^\top &= \frac{\partial l}{\partial \mathbf v_t}^\top \frac{\partial \mathbf v_t}{\partial \mathbf k_t} +  \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf k_t} \\

&= (1- \lambda_t)  \mathbf o_t^\top \mathbf {ds}_t ^\top
 - \lambda_t (1-\lambda_t)  [\mathbf {ds}_t \mathbf {s}_{t-1}^\top \mathbf q_t]^\top \\


 &= (1- \lambda_t) \left[  \mathbf {ds}_t   \mathbf o_t \right]^\top
 + (1-\lambda_t)  \left[\mathbf {ds}_{t} (\mathbf {v}_t - \mathbf {o}_t)\right]^\top  \\

 &=(1- \lambda_t) \left[  \mathbf {ds}_t   \mathbf v_t \right]^\top, \\


\mathbf {dq}_t^\top &= \frac{\partial l}{\partial \mathbf v_t}^\top \frac{\partial \mathbf v_t}{\partial \mathbf q_t} +  \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf q_t} \\

&= -\lambda_t \mathbf {dv}_t^\top \mathbf s_{t-1} ^\top
 - \lambda_t (1-\lambda_t) \left( \mathbf {s}_{t-1}\mathbf {ds}_t^\top \mathbf k_t]\right)^\top \\
&= -\lambda_t  \left[\mathbf s_{t-1} \mathbf {dv}_t\right]^\top
- \lambda_t \left[\mathbf {s}_{t-1} \mathbf p_t \right]^\top  \\
&= -\lambda_t \mathbf {s}_{t-1} \left[\mathbf {dv}_t + \mathbf p_t\right]^\top.
\end{aligned}
$$
计算流程：

1. 计算$\mathbf {do}_t, \mathbf p_t, \mathbf {dk}_t$；
2. 计算$\mathbf {dq}_t$；
