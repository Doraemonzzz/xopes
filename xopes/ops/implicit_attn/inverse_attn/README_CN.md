# Inverse attention

## Forward

给定输入$\mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n}$，记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
0, & i < j.
\end{cases}
$$
我们考虑给定$\mathbf K, \mathbf V, \mathbf A, \mathbf \Beta, \mathbf S_0$，求解$\mathbf O$：
$$
\begin{aligned}
\mathbf O &=\left[
\left(\mathbf I + \mathrm{diag}(\Gamma)\mathrm{tril}( \mathbf K \mathbf K^\top)\mathrm{diag}(\Beta)\right) \odot \mathbf M
\right]^{-1}  \mathbf V, \\

\left[
\left(\mathbf I + \mathrm{diag}(\Gamma) \mathrm{tril}( \mathbf K \mathbf K^\top)\mathrm{diag}(\Beta)\right) \odot \mathbf M
\right] \mathbf O &= \mathbf V .
\end{aligned}
$$

### 递推形式

我们将其恢复成原始的Linear Attention，即为：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} +  \beta_t\mathbf k_t \mathbf o_t^\top, \\
\mathbf o_t^\top+ \mathbf k_t^\top \mathbf s_{t-1} &=
\mathbf v_t^\top , \\
\mathbf o_t^\top &= \mathbf v_t^\top -  \mathbf k_t^\top \mathbf s_{t-1}.
\end{aligned}
$$
另一方面，将$\mathbf o_t$的定义代入到递推式中可得：
$$
\begin{aligned}
\mathbf s_t &=
\lambda_t \mathbf s_{t-1} +  \beta_t \mathbf k_t \mathbf o_t^\top, \\
 &=
\lambda_t \mathbf s_{t-1} + \beta_t \mathbf k_t
\left(
\mathbf o_t^\top  -  \mathbf k_t^\top \mathbf s_{t-1}
\right)  \\
&=
(\lambda_t  - \beta_t  \mathbf k_t \mathbf k_t^\top ) \mathbf s_{t-1}
+ \beta_t \mathbf k_t \mathbf o_t^\top.
\end{aligned}
$$
Decay矩阵$\lambda_t  - \beta_t  \mathbf k_t \mathbf k_t^\top$有$d-1$个特征值为$\lambda_t$，最后一个特征值为$\lambda_t -\beta_t \in [-1, 1]$。



## Backward

根据之前的递推，我们有：
$$
\begin{aligned}
\mathbf p_t &\triangleq (1-\lambda_t) \mathbf {ds}_{t}^\top \mathbf k_t, \\

\mathbf{ds}_{t} &= (\lambda_{t+1}  - \beta_{t+1} \mathbf k_{t+1} \mathbf k_{t+1}^\top )^\top
\mathbf{ds}_{t+1} - \mathbf{k}_{t+1}\mathbf {do}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \beta_{t+1} \mathbf k_{t+1} \left[\mathbf k_{t+1}^\top \mathbf{ds}_{t+1}\right]
-\mathbf{k}_{t+1}\mathbf {do}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1} (1-\lambda_{t+1}) \mathbf q_{t+1} \left[ \mathbf{ds}_{t+1}^\top \mathbf k_{t+1}\right]^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\

&=  \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} \mathbf p_{t+1}^\top
-\lambda_{t+1}\mathbf{q}_{t+1}\mathbf {dv}^\top_{t+1} \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}  \mathbf q_{t+1} (\mathbf p_{t+1} + \mathbf {dv}_{t+1})^\top \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {do}_{t+1}^\top  ,\\
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
&= -\lambda_t  \left[\mathbf {s}_{t-1}(\mathbf {dv}_t + \mathbf p_t)\right]^\top \\

&=  -\lambda_t  \left[\mathbf {s}_{t-1}\mathbf {do}_t\right]^\top
.
\end{aligned}
$$
计算流程：

1. 计算$\mathbf {do}_t, \mathbf p_t, \mathbf {dk}_t$；
2. 计算$\mathbf {dq}_t$；

### Decay部分梯度

记：
$$
\begin{aligned}
\mathbf u_t &= (1-\lambda_t) \odot \mathbf k_t, \\
\mathbf r_t & = \lambda_t \odot \mathbf u_t.
\end{aligned}
$$
那么递推式为：
$$
\begin{aligned}
\mathbf s_t
&= \lambda_t (\mathbf I - (1-\lambda_t) \mathbf k_t \mathbf q_t^\top ) \mathbf s_{t-1}
+ (1-\lambda_t) \mathbf k_t \mathbf o_t^\top \\
&= \lambda_t (\mathbf I -  \mathbf u_t \mathbf q_t^\top ) \mathbf s_{t-1}
+  \mathbf u_t \mathbf o_t^\top \\
&\triangleq (\lambda_t - \mathbf r_t \mathbf q_t^\top )  \mathbf s_{t-1} + \mathbf u_t \mathbf o_t^\top, \\

\mathbf v_t^\top &=
\mathbf o_t^\top  - \lambda_t  \mathbf q_t^\top \mathbf s_{t-1}, \\
\gamma_t &= \prod_{i=1}^t \lambda_i.
\end{aligned}
$$
根据deltanet的结论，我们可得：
$$
\begin{aligned}
\mathbf s_t &= \gamma_t \mathbf s_0+\sum_{i=1}^t \gamma_t/\gamma_i (\mathbf u_i \mathbf {\bar o}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top), \\


\mathbf v_t^\top
&= \mathbf o_t^\top - \lambda_t \mathbf q_t^\top \mathbf s_{t-1} \\
&= \mathbf o_t^\top - \lambda_t\mathbf q_t^\top
\left(
\gamma_{t-1} \mathbf s_0 +
\sum_{i=1}^{t-1} \gamma_{t-1}/\gamma_i (\mathbf u_i \mathbf {\bar o}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top)
\right) \\
&= \mathbf o_t^\top - \mathbf q_t^\top
\left(
\gamma_{t} \mathbf s_0 +
\sum_{i=1}^{t-1} \gamma_{t}/\gamma_i  (\mathbf u_i \mathbf {\bar o}_i^\top - \mathbf r_i \mathbf {\bar q}_i^\top)
\right)\\
&= \mathbf o_t^\top - (\mathbf q_t  \gamma_{t})^\top  \sum_{i=1}^{t-1}
\left[(\mathbf u_i/ \gamma_i)\mathbf {\bar o}_i^\top
- (\mathbf r_i/ \gamma_i)\mathbf {\bar q}_i^\top
\right]- (\mathbf q_t \gamma_t)^\top \mathbf s_0.
\end{aligned}
$$
我们先考虑$\mathbf v_t$对于梯度的贡献：
$$
\begin{aligned}
\mathbf {d\log \gamma_t}
&= - \left[\mathbf s_0 \mathbf{dv}_t \right]\odot [ \mathbf q_t \odot \gamma_t ]   -
[ \mathbf q_t \odot \gamma_t ]  \odot \sum_{j=1}^{t-1}
\left[ (\mathbf u_j/\gamma_j) \mathbf {\bar o}_j ^\top
-(\mathbf r_j/\gamma_j) \mathbf {\bar q}_j ^\top
\right]
\mathbf{dv}_t   \\
&+ [\mathbf u_t/\gamma_t]  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \gamma_s)\mathbf{dv}_s ^\top \right] \mathbf {\bar o}_t\right)
- [\mathbf r_t/\gamma_t]  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \gamma_s)\mathbf{dv}_s ^\top \right] \mathbf {\bar q}_t\right), \\

\mathbf{d\bar q}_t
&=-[\mathbf s_0 \mathbf {dv}_t]\odot \mathbf \gamma_t
-  \gamma_t\odot \sum_{i=1}^{t-1}
\left[(\mathbf u_i/ \gamma_i)\mathbf {\bar o}_i^\top
- (\mathbf r_i/ \gamma_i)\mathbf {\bar q}_i^\top
\right] \mathbf {dv}_t, \\
&= \mathbf{d q}_t +\lambda_t \mathbf s_{t-1}\mathbf p_t,  \\

\mathbf{du}_t
&=  -(1/\gamma_t)  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \gamma_s)\mathbf{dv}_s ^\top \right] \mathbf {\bar o}_t\right) -
\left[\mathbf{ds}_n \gamma_n / \gamma_t\right] \mathbf {\bar o}_t \\
&\triangleq  \mathbf{d\bar u}_t  -
\left[\mathbf{ds}_n \gamma_n / \gamma_t\right] \mathbf {\bar o}_t,\\


\mathbf{dr}_t
&=  (1/\gamma_t)  \odot \left(\left[ \sum_{s=t+1}^n  (\mathbf q_s \odot \gamma_s)\mathbf{dv}_s ^\top \right] \mathbf {\bar q}_t\right) +
\left[\mathbf{ds}_n \gamma_n / \gamma_t\right]^\top \mathbf {\bar q}_t \\

&\triangleq \mathbf{d\bar r}_t + \left[\mathbf{ds}_n \gamma_n / \gamma_t\right]^\top \mathbf {\bar q}_t,

\\
\mathbf {d\log \gamma_t} &= \mathbf q_t \odot \mathbf {d\bar q}_t -
\mathbf u_t \odot \mathbf {du}_t - \mathbf r_t \odot \mathbf {dr}_t
- \mathbf u_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right]^\top \mathbf h_t, \\

\mathbf {dh}_t &= \sum_{s=t+1}^n\mathbf {dv}_{s} (\mathbf q_{s} \gamma_s)^\top (\mathbf r_t/\gamma_t).

\end{aligned}
$$
接着将下式代入：
$$
\begin{aligned}
\mathbf u_t &= (1-\lambda_t) \odot \mathbf k_t, \\
\mathbf r_t & = \lambda_t \odot \mathbf u_t.
\end{aligned}
$$
我们有：
$$
\begin{aligned}
\mathbf {dk}_t &=
\frac{\partial l}{\partial \mathbf u_t} \frac{\partial \mathbf u_t}{\partial \mathbf k_t} +
\frac{\partial l}{\partial \mathbf r_t} \frac{\partial \mathbf r_t}{\partial \mathbf k_t}  \\
&= (1-\lambda_t)\odot \mathbf {du}_t  + \lambda_t \odot (1-\lambda_t) \odot \mathbf {dr}_t.



\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf u_t \odot \mathbf {du}_t + \mathbf r_t \odot \mathbf {dr}_t
&= (1-\lambda_t) \odot \mathbf k_t \odot \mathbf {du}_t
+ \lambda_t \odot (1-\lambda_t) \odot \mathbf k_t \odot \mathbf {dr}_t \\
&= \mathbf k_t \odot \mathbf {dk}_t.
\end{aligned}
$$
因此：
$$
\mathbf {d\log \gamma_t} = \mathbf q_t \odot \mathbf {d q}_t -
\mathbf k_t \odot \mathbf {dk}_t
+ \lambda_t \mathbf s_{t-1}\mathbf p_t
- \mathbf u_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right]^\top \mathbf h_t.
$$
接着考虑$\mathbf s_n$带来的梯度，参考vector decay/deltanet的结论：
$$
\mathbf{d}\log \bar \gamma_t + \lambda_t \mathbf s_{t-1}\mathbf p_t - \mathbf u_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right] \mathbf o_t
+ \mathbf r_t \odot \left[\mathbf{ds}_n \gamma_n / \gamma_t\right]^\top \mathbf h_t =
\begin{cases}
0, & t < n,\\
[\mathbf s_n \odot \mathbf {ds}_n] \mathbf 1_e, & t=n.
\end{cases}
$$
综上：
$$
\mathbf {d}\log \gamma_t =
\begin{cases}
\mathbf q_t \odot \mathbf {d\bar q}_t -
\mathbf k_t \odot \mathbf {dk}_t , & t < n \\
\mathbf q_t \odot \mathbf {d\bar q}_t -
\mathbf k_t \odot \mathbf {dk}_t  + [\mathbf s_n \odot \mathbf {ds}_n]1_e, & t=n
\end{cases}
$$
最后，我们还需要考虑input部分的decay，即下式带来的梯度：
$$
\mathbf u_t = (1-\lambda_t) \odot \mathbf k_t.
$$
推导可得：
$$
\begin{aligned}
\mathbf {d\log \tilde \lambda_t}&=-  \mathbf {du}_t \odot \lambda_t  \odot \mathbf k_t,  \\
\mathbf {d k}_t &= (1-\lambda_t ) \odot \mathbf {du}_t,  \\
\mathbf {du}_t &= \mathbf {d k}_t / (1-\lambda_t) , \\
\mathbf {d\log \tilde \lambda_t} &= -  (\lambda_t/(1-\lambda_t)) \odot \mathbf {dk}_t  \odot \mathbf k_t.

\end{aligned}
$$
