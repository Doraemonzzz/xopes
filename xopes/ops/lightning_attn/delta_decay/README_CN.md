# Lightning Attention with Delta Decay

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初始state $\mathbf s_0$，以及Decay $\Lambda \in \mathbb R^{n\times d}, \mathbf A\in \mathbb R^{n\times d},\mathbf B \in \mathbb R^{n\times e}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &=  (\mathrm{diag}(\lambda_i)+ \mathbf a_i \mathbf b_i^\top) \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times e}.
$$

在后续讨论中，我们记：
$$
\mathbf b_t = \mathbf c_t \odot \lambda_t.
$$


## 展开式

记：
$$
\begin{aligned}
\mathbf M_i &= \prod_{k=1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top), \\
\mathbf M_i^j &= \mathbf M_j^{-1}  \mathbf M_i  \\
&=\prod_{k=j+1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top).
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf s_i &=   (\mathrm{diag}(\lambda_i) + \mathbf a_i \mathbf b_i^\top) \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top \\
&=   (\mathrm{diag}(\lambda_i) + \mathbf a_i \mathbf b_i^\top)
\left[  (\mathrm{diag}(\lambda_{i-1}) + \mathbf a_{i-1} \mathbf b_{i-1}^\top)  \mathbf s_{i-2}
+ \mathbf k_{i-1} \mathbf v_{i-1}^\top \right] + \mathbf k_i \mathbf v_i^\top \\
&= \mathbf M_{i}^j \mathbf s_j + \sum_{s=j+1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top  \\
&= \mathbf M_{i}\mathbf s_0 + \sum_{s=1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top.

\end{aligned}
$$
下面分别讨论这两项的展开式。

### $\mathbf M_i^j$

$$
\mathbf M_i^j =\prod_{k=j+1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top), i\ge j+1.
$$

我们猜测：
$$
\begin{aligned}
\mathbf M_i^j &=\mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top \\

\mathbf M_i &=\mathrm{diag}(\gamma_i) + \sum_{k=1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top .
\end{aligned}
$$
$i=j+1$时：
$$
\begin{aligned}
\mathbf M_{j+1}^j

&= \mathrm{diag}(\lambda_{j+1}) + \mathbf a_{j+1} \mathbf b_{j+1}^\top \\
& \triangleq \mathrm{diag}(\gamma_{j+1}/\gamma_j) + \mathbf a_{j+1} \mathbf u_{j+1}^\top, \\
\mathbf u_{j+1} &= \mathbf b_{j+1}.
\end{aligned}
$$
假设当$i$时结论成立，那么$i+1$时：
$$
\begin{aligned}
\mathbf M_{i+1}^{j}
& = (\mathrm{diag}(\lambda_{i+1}) + \mathbf a_{i+1} \mathbf b_{i+1}^\top)\mathbf M_i^{j}  \\
& = (\mathrm{diag}(\lambda_{i+1})+ \mathbf a_{i+1} \mathbf b_{i+1}^\top)
\left(
\mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i \mathrm{diag}(\gamma_i/\gamma_k) \mathbf a_k \mathbf u_k^\top
\right)\\
&=  \mathrm{diag}(\lambda_{i+1})\left(
\mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k) \mathbf a_k \mathbf u_k^\top \right)
+ \mathbf a_{i+1} \mathbf b_{i+1}^\top \mathrm{diag}(\gamma_i/\gamma_j)  + \mathbf a_{i+1} \mathbf b_{i+1}^\top \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k) \mathbf a_k \mathbf u_k^\top

  \\
&= \mathrm{diag}(\gamma_{i+1}/\gamma_j) + \sum_{k=j+1}^i \mathrm{diag}(\gamma_{i+1}/\gamma_k) \mathbf a_k \mathbf u_k^\top +  \mathbf a_{i+1} \mathbf b_{i+1}^\top \mathrm{diag}(\gamma_i/\gamma_j)
 +
\mathbf a_{i+1} \mathbf b_{i+1}^\top \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top \\

&=  \mathrm{diag}(\gamma_{i+1}/\gamma_j) + \sum_{k=j+1}^i \mathrm{diag}(\gamma_{i+1}/\gamma_k) \mathbf a_k \mathbf u_k^\top + \mathbf a_{i+1}\left(\mathbf b_{i+1}^\top \mathrm{diag}(\gamma_i/\gamma_j) +\mathbf b_{i+1}^\top \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top  \right).

\end{aligned}
$$
因此，对于上标$j$：
$$
\mathbf u_{i+1}= \mathrm{diag}(\gamma_i/\gamma_j) \mathbf b_{i+1} +

\sum_{k=j+1}^i   \mathbf u_k\mathbf a_k^\top  \mathrm{diag}(\gamma_i/\gamma_k) \mathbf b_{i+1}.
$$

特别的，取$j=0$，将$\mathbf b_t = \mathbf c_t \odot \lambda_t$代入可得：
$$
\begin{aligned}
\mathbf u_{i+1} &= \gamma_{i+1} \odot \mathbf c_{i+1} +

\sum_{k=1}^i   \mathbf u_k\mathbf a_k^\top  \mathrm{diag}(\gamma_{i+1}/\gamma_k) \mathbf c_{i+1}, \\
\mathbf u_1 &= \gamma_{1} \odot \mathbf c_{1}.

\end{aligned}
$$
写成矩阵形式为：
$$
\begin{aligned}
\mathbf {\bar C} &=  \mathbf C \odot \mathbf {\Gamma} , \\
\mathbf {\bar A} &=\mathbf A / \mathbf {\Gamma}, \\
\mathbf U & = \mathbf {\bar C}  + \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1) \mathbf U, \\
[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf U &= \mathbf {\bar C}.
\end{aligned}
$$



### $\sum_{s=j+1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top $

根据之前的推导，猜测：
$$
\sum_{s=j+1}^i \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top
=\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right).
$$
$i=j+1$时：
$$
\sum_{s=j+1}^{j+1} \mathbf M_{j+1}^{s} \mathbf k_s \mathbf v_s^\top
=\sum_{t=j+1}^{j+1}  \mathrm{diag}(\gamma_{j+1}/\gamma_{j+1})

\mathbf k_t \mathbf v_t^\top
= \mathbf k_{j+1} \mathbf v_{j+1}^\top, \\
\mathbf p_{j+1}=\mathbf 0.
$$
假设当$i$时结论成立，那么$i+1$时：
$$
\begin{aligned}
\sum_{s=j+1}^{i+1} \mathbf M_{i+1}^{s} \mathbf k_s \mathbf v_s^\top
&=(\mathrm{diag}(\lambda_{i+1}) + \mathbf a_{i+1}\mathbf b_{i+1}^\top)\sum_{s=j+1}^{i} \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top
+ \mathbf k_{i+1} \mathbf v_{i+1}^\top \\
&= \mathrm{diag}(\lambda_{i+1})\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right)
+  \mathbf a_{i+1}\mathbf b_{i+1}^\top  \sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right)

+ \mathbf k_{i+1} \mathbf v_{i+1}^\top \\
&= \sum_{t=j+1}^i  \mathrm{diag}(\gamma_{i+1}/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right)
+\mathbf k_{i+1} \mathbf v_{i+1}^\top +   \mathbf a_{i+1}\mathbf b_{i+1}^\top  \sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right) \\

&\triangleq \sum_{t=j+1}^i  \mathrm{diag}(\gamma_{i+1}/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right)
+\mathbf k_{i+1} \mathbf v_{i+1}^\top
+ \mathbf a_{i+1} \mathbf p_{i+1}^\top \\

&= \sum_{t=j+1}^{i+1}  \mathrm{diag}(\gamma_{i+1}/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right).

\end{aligned}
$$
其中，对于上标$j$：
$$
\mathbf p_{i+1}=  \sum_{t=j+1}^i
\left(
 \mathbf v_t \mathbf k_t^\top +  \mathbf p_t\mathbf a_t^\top
\right)\mathrm{diag}(\gamma_i/\gamma_t)\mathbf b_{i+1}.
$$

特别的，取$j=0$得到，将$\mathbf b_t = \mathbf c_t \odot \lambda_t$代入可得：
$$
\begin{aligned}
\mathbf p_{i+1} &=  \sum_{t=1}^i
\left(
 \mathbf v_t \mathbf k_t^\top +  \mathbf p_t\mathbf a_t^\top
\right)\mathrm{diag}(\gamma_{i+1}/\gamma_t)\mathbf c_{i+1} , \\

\mathbf p_1 &= 0 .

\end{aligned}
$$
写成矩阵形式为：
$$
\begin{aligned}
\mathbf {\bar C} &=  \mathbf C \odot \mathbf {\Gamma} , \\
\mathbf {\bar A} &=\mathbf A / \mathbf {\Gamma}, \\
\mathbf {\bar K} &= \mathbf K / \mathbf {\Gamma}, \\
\mathbf P & = \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V  + \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1) \mathbf P, \\
[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf P &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V.
\end{aligned}
$$


### 小结

完全展开上式可得：
$$
\begin{aligned}
\mathbf s_t
&=\left[ \mathrm{diag}(\gamma_t) + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)\mathbf a_i\mathbf u_i^\top \right] \mathbf s_0 + \sum_{k=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
\left(
\mathbf k_i \mathbf v_i^\top + \mathbf a_i \mathbf p_i^\top
\right) \\

&=  \mathrm{diag}(\gamma_t)\mathbf s_0   + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
\left(
\mathbf k_i \mathbf v_i^\top + \mathbf a_i (\mathbf p_i + \mathbf s_0^\top \mathbf u_i) ^\top
\right)  \\

&=  \mathrm{diag}(\gamma_t)\mathbf s_0   + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
\left(
\mathbf k_i \mathbf v_i^\top + \mathbf a_i \mathbf r_i^\top
\right),  \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf U &= \mathbf {\bar C}, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf P &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V, \\

\mathbf {\bar C} &=  \mathbf C \odot \mathbf {\Gamma} , \\
\mathbf {\bar A} &=\mathbf A / \mathbf {\Gamma}, \\
\mathbf {\bar K} &= \mathbf K / \mathbf {\Gamma}, \\
\mathbf b_t &= \mathbf c_t \odot \lambda_t.

\end{aligned}
$$
根据上述展开公式，不难得到：
$$
\begin{aligned}
\mathbf s_t &= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf r_t^\top, \\



\end{aligned}
$$
另一方面：
$$
\begin{aligned}
\mathbf u_{i+1} &= \gamma_{i+1} \odot \mathbf c_{i+1} +

\sum_{k=1}^i   \mathbf u_k\mathbf a_k^\top  \mathrm{diag}(\gamma_{i+1}/\gamma_k) \mathbf c_{i+1}, \\
\mathbf u_1 &= \gamma_{1} \odot \mathbf c_{1}, \\


\mathbf p_{i+1} &=  \sum_{t=1}^i
\left(
 \mathbf v_t \mathbf k_t^\top +  \mathbf p_t\mathbf a_t^\top
\right)\mathrm{diag}(\gamma_{i+1}/\gamma_t)\mathbf c_{i+1} , \\

\mathbf p_1 &= 0 .

\end{aligned}
$$


## Fwd

根据上述公式，我们可得如下流程：
$$
\begin{aligned}
 [\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf U &= \mathbf {\bar C}, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf {P} &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V , \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf { P} &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V - \mathbf V + \mathbf V, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf {\bar P} &=  \mathbf V, \\

\mathbf P &= \mathbf {\bar P}  - \mathbf V, \\

\mathbf {\bar C} &=  \mathbf C \odot \mathbf {\Gamma} , \\
\mathbf {\bar A} &=\mathbf A / \mathbf {\Gamma}, \\
\mathbf {\bar K} &= \mathbf K / \mathbf {\Gamma}, \\
\mathbf b_t &= \mathbf c_t \odot \lambda_t, \\

\mathbf r_t & =\mathbf p_t + \mathbf s_0^\top \mathbf u_t, \\
\mathbf s_t &= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf r_t^\top, \\

\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{e}.
\end{aligned}
$$
流程为：

- 计算$\mathbf U, \mathbf P$；
  - 后续讨论这部分如何求解；
- 然后计算double state的lightning attention；



## Bwd

这部分使用lightning attention计算即可：
$$
\begin{aligned}
\mathbf {ds}_{n+1} &= \mathbf {ds}_n , \\
\mathbf {ds}_n &= \mathbf {ds}_{n+1} + \mathbf q_n \mathbf{do}_n^\top , \\

\mathbf {ds}_t &= \mathrm{diag}(\lambda_{t+1}) \mathbf {ds}_{t+1} + \mathbf q_t \mathbf {do}_t^\top, \\

\mathbf {ds}_0 &= \mathrm{diag}(\lambda_1) \mathbf {ds}_1, \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dv}_t& = \mathbf k_t^\top \mathbf {ds}_t, \\

\mathbf{da}_t^\top &=\mathbf r_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dr}_t& = \mathbf a_t^\top \mathbf {ds}_t.

\end{aligned}
$$
给定$\mathbf {dr}_t$以及$\mathbf r_t = \mathbf p_t + \mathbf s_0^\top \mathbf u_t$，那么：
$$
\begin{aligned}
\mathbf {dp}_t &= \mathbf {dr}_t , \\
\mathbf {d{\bar  p}}_t &= \mathbf {dp}_t + \mathbf {dv}_t = \mathbf {dr}_t +  \mathbf {dv}_t, \\
\mathbf {du}_t &= \mathbf s_0 \mathbf {dr}_t, \\

[\mathbf {ds}_0]^t &= \mathbf u_t \mathbf {dr}_t^\top.
\end{aligned}
$$
最终的$\mathbf {ds}_0$为：
$$
\mathbf {ds}_0 = \mathbf {ds}_0 + \sum_t [\mathbf {ds}_0]^t .
$$
最后，剩余的部分为求解$\mathbf {V}, \mathbf {C}$的梯度。
