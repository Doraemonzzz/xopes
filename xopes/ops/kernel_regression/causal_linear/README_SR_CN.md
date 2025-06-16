# Kernel Regression(Causal linear) Sequential Recurrence

## Forward

给定输入$\mathbf Q'\in \mathbb R^{n\times d}, \mathbf K'\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}, \mathbf \Alpha, \mathbf \Beta \in \mathbb R^n$，以及Decay $\Lambda\in \mathbb R^{n}$，记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
0, & i < j.
\end{cases}
$$
计算：
$$
\begin{aligned}
\mathbf O &= [\mathbf I + \mathrm{tril}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ]^{-1} \mathbf V, \\

\mathbf Q &=  \mathbf Q' \odot \mathbf \Alpha, \\

\mathbf K &= \mathbf K' \odot \mathbf \Beta.

\end{aligned}
$$
注意到上式等价于：
$$
[\mathbf I + \mathrm{tril}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ] \mathbf O = \mathbf V.
$$




### 递推形式

注意到上式等价于：
$$
[\mathbf I + \mathrm{tril}([(\mathbf Q \mathbf K^\top) \odot \mathbf M], -1) ] \mathbf O = \mathbf V.
$$
即：
$$
\mathbf o_t^\top  +  \mathbf q_t^\top \sum_{j=1}^{t-1} \mathbf k_j\mathbf o_j^\top \alpha_t/ \alpha_j = \mathbf v_t^\top
$$
记：
$$
\mathbf s_t = \sum_{j=1}^{t} \mathbf k_j\mathbf o_j^\top \alpha_t/ \alpha_j.
$$
那么：

$$
\begin{aligned}
\mathbf o_t^\top  +   \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}
& = \mathbf v_t^\top,  \\
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + \mathbf k_t \mathbf o_t^\top \\
&=  \lambda_t \mathbf s_{t-1} +  \mathbf k_t ( \mathbf v_t^\top - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1} ) \\
&= \lambda_t(\mathbf I - \mathbf k_t \mathbf q_t^\top) \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top.
\end{aligned}
$$


## Backward

根据之前的递推，我们有：
$$
\begin{aligned}
\mathbf p_t &\triangleq \mathbf {ds}_{t}^\top \mathbf k_t, \\

\mathbf{ds}_{t} &= \lambda_{t+1} (\mathbf I - \mathbf k_{t+1} \mathbf q_{t+1}^\top )^\top
\mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf{q}_{t+1}\mathbf {do}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf q_{t+1} \left[\mathbf k_{t+1}^\top \mathbf{ds}_{t+1}\right]
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {do}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} \left[ \mathbf{ds}_{t+1}^\top \mathbf k_{t+1}\right]^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {do}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} \mathbf p_{t+1}^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {do}^\top_{t+1} \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} (\mathbf p_{t+1} + \mathbf {do}_{t+1})^\top \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {dv}_{t+1}^\top  ,\\
&t=1,\ldots, n- 1, \\

\mathbf {ds}_0&=  \lambda_1 (\mathbf I -  \mathbf q_{1} \mathbf k_{1}^\top ) \mathbf {ds}_1 \\


&=\lambda_1 \mathbf{ds}_1 - \lambda_1 \mathbf q_{1} \mathbf p_1^\top,  \\


\mathbf {dv}_t^\top &= \frac{\partial l}{\partial \mathbf o_t}^\top \frac{\partial \mathbf o_t}{\partial \mathbf v_t} +  \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf v_t} \\

&= \mathbf {do}_t^\top + \mathbf {ds}_t^\top  \mathbf k_t \\

&= \mathbf {do}_t^\top + \mathbf p_t^\top ,\\

\mathbf {dk}_t^\top &= \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf k_t} \\

&= \mathbf v_t^\top \mathbf {ds}_t ^\top
 - \lambda_t  [\mathbf {ds}_t \mathbf {s}_{t-1}^\top \mathbf q_t]^\top \\


 &=  \left[  \mathbf {ds}_t   \mathbf v_t \right]^\top
 +   \left[\mathbf {ds}_{t} (\mathbf {o}_t - \mathbf {v}_t)\right]^\top  \\

 &=\left[  \mathbf {ds}_t   \mathbf o_t \right]^\top, \\


\mathbf {dq}_t^\top &= \frac{\partial l}{\partial \mathbf o_t}^\top \frac{\partial \mathbf o_t}{\partial \mathbf q_t} +  \frac{\partial l}{\partial \mathbf s_t} \frac{\partial \mathbf s_t}{\partial \mathbf q_t} \\

&= -\lambda_t \mathbf {do}_t^\top \mathbf s_{t-1} ^\top
 - \lambda_t  \left( \mathbf {s}_{t-1}\mathbf {ds}_t^\top \mathbf k_t]\right)^\top \\
&= -\lambda_t  \left[\mathbf s_{t-1} \mathbf {do}_t\right]^\top
- \lambda_t \left[\mathbf {s}_{t-1} \mathbf p_t \right]^\top  \\
&= -\lambda_t  \left[\mathbf {s}_{t-1}(\mathbf {do}_t + \mathbf p_t)\right]^\top \\

&=  -\lambda_t  \left[\mathbf {s}_{t-1}\mathbf {dv}_t\right]^\top
.
\end{aligned}
$$
计算流程：

1. 计算$\mathbf {dv}_t, \mathbf p_t, \mathbf {dk}_t$；
2. 计算$\mathbf {dq}_t$；



### Decay部分梯度

记：
$$
\begin{aligned}

\mathbf r_t & = \lambda_t \odot \mathbf k_t.
\end{aligned}
$$
那么递推式为：
$$
\begin{aligned}
\mathbf s_t
&= \lambda_t (\mathbf I -  \mathbf k_t \mathbf q_t^\top ) \mathbf s_{t-1}
+ \mathbf k_t \mathbf v_t^\top \\
&= \lambda_t (\mathbf I -  \mathbf u_t \mathbf q_t^\top ) \mathbf s_{t-1}
+  \mathbf r_t \mathbf v_t^\top \\
&\triangleq (\lambda_t - \mathbf r_t \mathbf q_t^\top )  \mathbf s_{t-1} + \mathbf r_t \mathbf v_t^\top, \\

\mathbf v_t^\top &=
\mathbf o_t^\top  - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}, \\
\alpha_t &= \prod_{i=1}^t \lambda_i.
\end{aligned}
$$
根据deltanet的结论，我们可得：
$$
\begin{aligned}
\mathbf s_t &= \alpha_t \mathbf s_0+\sum_{i=1}^t \alpha_t/\alpha_i (\mathbf r_i \mathbf {\bar v}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top), \\


\mathbf o_t^\top
&= \mathbf v_t^\top - \lambda_t \mathbf q_t^\top \mathbf s_{t-1} \\
&= \mathbf v_t^\top - \lambda_t\mathbf q_t^\top
\left(
\alpha_{t-1} \mathbf s_0 +
\sum_{i=1}^{t-1} \alpha_{t-1}/\alpha_i (\mathbf r_i \mathbf {\bar v}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top)
\right) \\
&= \mathbf v_t^\top - \mathbf q_t^\top
\left(
\alpha_{t} \mathbf s_0 +
\sum_{i=1}^{t-1} \alpha_{t}/\alpha_i  (\mathbf r_i \mathbf {\bar v}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top)
\right)\\
&= \mathbf v_t^\top - (\mathbf q_t  \alpha_{t})^\top  \sum_{i=1}^{t-1}
\left[(\mathbf r_i/ \alpha_i)\mathbf {\bar v}_i^\top
- (\mathbf r_i/ \alpha_i)\mathbf {\bar q}_i^\top
\right]- (\mathbf q_t \alpha_t)^\top \mathbf s_0.
\end{aligned}
$$


我们先考虑$\mathbf v_t$对于梯度的贡献：
$$
\begin{aligned}
\mathbf {d\log \alpha_t}
&= - \left[\mathbf s_0 \mathbf{do}_t \right]\odot [ \mathbf q_t \odot \alpha_t ]   -
[ \mathbf q_t \odot \alpha_t ]  \odot \sum_{j=1}^{t-1}
\left[ (\mathbf r_j/\alpha_j) \mathbf {\bar v}_j ^\top
-(\mathbf r_j/\alpha_j) \mathbf {\bar q}_j ^\top
\right]
\mathbf{do}_t   \\
&+ [\mathbf r_t/\alpha_t]  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \alpha_s)\mathbf{do}_s ^\top \right] \mathbf {\bar v}_t\right)
- [\mathbf r_t/\alpha_t]  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \alpha_s)\mathbf{do}_s ^\top \right] \mathbf {\bar q}_t\right), \\

\mathbf{d\bar q}_t
&=-[\mathbf s_0 \mathbf {do}_t]\odot \mathbf \alpha_t
-  \alpha_t\odot \sum_{i=1}^{t-1}
\left[(\mathbf r_i/ \alpha_i)\mathbf {\bar v}_i^\top
- (\mathbf r_i/ \alpha_i)\mathbf {\bar q}_i^\top
\right] \mathbf {dv}_t, \\
&= \mathbf{d q}_t +\lambda_t \mathbf s_{t-1}\mathbf p_t,  \\

\mathbf{d \bar k}_t
&=  -(1/\alpha_t)  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \alpha_s)\mathbf{do}_s ^\top \right] \mathbf {\bar v}_t\right) -
\left[\mathbf{ds}_n \alpha_n / \alpha_t\right] \mathbf {\bar v}_t \\
&\triangleq  \mathbf{d\bar u}_t  -
\left[\mathbf{ds}_n \alpha_n / \alpha_t\right] \mathbf {\bar o}_t,\\


\mathbf{dr}_t
&=  (1/\alpha_t)  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \alpha_s)\mathbf{do}_s ^\top \right] \mathbf {\bar q}_t\right) +
\left[\mathbf{ds}_n \alpha_n / \alpha_t\right]^\top \mathbf {\bar q}_t \\

&\triangleq \mathbf{d\bar r}_t + \left[\mathbf{ds}_n \alpha_n / \alpha_t\right]^\top \mathbf {\bar q}_t,

\\
\mathbf {d\log \alpha_t} &= \mathbf q_t \odot \mathbf {d\bar q}_t -
\mathbf k_t \odot \mathbf {d\bar k}_t - \mathbf r_t \odot \mathbf {dr}_t
- \mathbf k_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right]^\top \mathbf {\bar q}_t.

\end{aligned}
$$
接着将下式代入：
$$
\begin{aligned}
\mathbf r_t & = \lambda_t \odot \mathbf k_t.
\end{aligned}
$$
我们有：
$$
\begin{aligned}
\mathbf {dk}_t &=

\mathbf {d\bar k}_t  + \frac{\partial \mathbf r_t}{\partial \mathbf k_t}  \\
&= \mathbf {d\bar k}_t  +  \lambda_t  \odot \mathbf {dr}_t.



\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf k_t \odot \mathbf {d\bar k}_t + \mathbf r_t \odot \mathbf {dr}_t
&= \mathbf k_t \odot \mathbf {dk}_t
+  \lambda_t \odot \mathbf k_t \odot \mathbf {dr}_t \\
&= \mathbf k_t \odot \mathbf {dk}_t.
\end{aligned}
$$
因此：
$$
\mathbf {d\log \alpha_t} = \mathbf q_t \odot \mathbf {d q}_t -
\mathbf k_t \odot \mathbf {dk}_t
+ \lambda_t \mathbf s_{t-1}\mathbf p_t
- \mathbf u_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right]^\top \mathbf {\bar q}_t.
$$
接着考虑$\mathbf s_n$带来的梯度，参考vector decay/deltanet的结论：
$$
\lambda_t \mathbf s_{t-1}\mathbf p_t
- \mathbf u_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \alpha_n / \alpha_t\right]^\top \mathbf {\bar q}_t =
\begin{cases}
0, & t < n,\\
[\mathbf s_n \odot \mathbf {ds}_n] \mathbf 1_e, & t=n.
\end{cases}
$$
综上：
$$
\mathbf {d}\log \alpha_t =
\begin{cases}
\mathbf q_t \odot \mathbf {d q}_t -
\mathbf k_t \odot \mathbf {dk}_t , & t < n \\
\mathbf q_t \odot \mathbf {d q}_t -
\mathbf k_t \odot \mathbf {dk}_t  + [\mathbf s_n \odot \mathbf {ds}_n]1_e, & t=n
\end{cases}
$$
