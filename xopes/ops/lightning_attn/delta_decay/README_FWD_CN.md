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
分chunk，然后写成矩阵形式为：
$$
\begin{aligned}
\Pi_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\mathbf {\bar C}_i &=  \mathbf C_i \odot \mathbf {\Pi}_i , \\
\mathbf {\bar A}_i &=\mathbf A_i / \mathbf {\Pi}_i, \\
\mathbf U_i & = \mathbf {\bar C}_i  + \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1) \mathbf U_i, \\
[\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf U_i &= \mathbf {\bar C}_i.
\end{aligned}
$$
因此第$i$个chunk对应的$\mathbf M_i^j $为（注意这里要和之前的$\mathbf M_i$区分开来）：
$$
\begin{aligned}
\mathbf {\tilde A}_i &= \mathbf A_i \odot \Pi_i, \\
\mathbf M_i &= \Pi_{i } + \mathbf {\tilde A}_i^\top \mathbf U_i.
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

分chunk，然后写成矩阵形式为：
$$
\begin{aligned}
\mathbf {\bar C}_i &=  \mathbf C_i \odot \mathbf {\Pi}_i , \\
\mathbf {\bar A}_i &=\mathbf A_i / \mathbf {\Pi}_i, \\
\mathbf {\bar K}_i &= \mathbf K_i / \mathbf {\Pi}_i, \\
\mathbf P_i & = \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i  + \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1) \mathbf P_i, \\
[\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf P_i &= \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i.

\end{aligned}
$$
因此第$i$个chunk对应的$\sum_{s=j+1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top $为：
$$
\begin{aligned}
\Theta_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s, \\
&= \alpha_{ic}/ \alpha_{(i-1)c+j-1}, \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i, \\

\tilde{\mathbf A}_i &=  \mathbf A_i \odot \mathbf \Theta_i,  \\

\mathbf S_{i, inter} &= \mathbf {\tilde K}_i^\top \mathbf V_i + \mathbf {\tilde A}_i^\top \mathbf P_i.
\end{aligned}
$$



### Forward
另一方面，我们考虑$\mathbf s_t$，我们有：
$$
\begin{aligned}
\mathbf s_i &=   (\mathrm{diag}(\lambda_i) + \mathbf a_i \mathbf b_i^\top) \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top \\



&= \mathbf M_{i}^j \mathbf s_j + \sum_{s=j+1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top  \\

&= \left(\mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top\right) \mathbf s_j +
\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right) \\

&= \mathrm{diag}(\gamma_i/\gamma_j) \mathbf s_j +
\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top + \mathbf a_k \mathbf u_k^\top  \mathbf s_j
\right), \\

\mathbf o_i^\top
&= \mathbf q_i^\top  \mathrm{diag}(\gamma_i/\gamma_j) \mathbf s_j + \mathbf q_i^\top
\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top + \mathbf a_t \mathbf u_t^\top  \mathbf s_j
\right).

\end{aligned}
$$
写成矩阵形式，即为：
$$
\begin{aligned}
\Pi_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Pi_i,  \\
\bar{\mathbf K}_i &=  \mathbf K_i / \mathbf \Pi_i,  \\
\mathbf {\bar A}_i &=\mathbf A_i / \mathbf {\Pi}_i, \\

\Theta_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s, \\
&= \alpha_{ic}/ \alpha_{(i-1)c+j-1}, \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i, \\

\tilde{\mathbf A}_i &=  \mathbf A_i \odot \mathbf \Theta_i,  \\

\mathbf O_i &= \mathbf {\bar Q}_i \mathbf S_{i-1} +
\left[
\mathbf {\bar Q}_i \mathbf {\bar K}_i^\top  \odot \mathbf M
\right] \mathbf V_i + \left[
\mathbf {\bar Q}_i \mathbf {\bar A}_i^\top  \odot \mathbf M
\right] (\mathbf P_i + \mathbf U_i \mathbf S_{i-1}), \\

\mathbf S_i
&= \mathbf M_{i}^j \mathbf S_{i-1} + \sum_{s=j+1}^i \mathbf M_{i}^s \mathbf k_s \mathbf v_s^\top  \\
&= (\Pi_{i, c} + \mathbf {\tilde A}_i^\top \mathbf U_i) \mathbf S_{i-1} + \mathbf {\tilde K}_i^\top \mathbf V_i + \mathbf {\tilde A}_i^\top \mathbf P_i \\
&= \Pi_{i, c}\mathbf S_{i-1} +  \mathbf {\tilde A}_i^\top (\mathbf U_i \mathbf S_{i-1} + \mathbf P_i) +
\mathbf {\tilde K}_i^\top \mathbf V_i.

\end{aligned}
$$
其中：
$$
\begin{aligned}
\mathbf {\bar C}_i &=  \mathbf C_i \odot \mathbf {\Pi}_i , \\
\mathbf {\bar A}_i &=\mathbf A_i / \mathbf {\Pi}_i, \\

\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf U_i &= \mathbf {\bar C}_i, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf P_i &= \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i.

\end{aligned}
$$
