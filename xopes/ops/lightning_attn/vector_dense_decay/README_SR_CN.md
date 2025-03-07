# Lightning Attention with Delta Decay

## Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda \in \mathbb R^{n\times d}, \mathbf A\in \mathbb R^{n\times d},\mathbf B \in \mathbb R^{n\times e}$，我们执行如下递归：
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




### 展开式

记：
$$
\begin{aligned}
\mathbf M_i &= \prod_{k=1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top), \\
\mathbf M_j^i &= \mathbf M_j^{-1}  \mathbf M_i  \\
&=\prod_{k=j+1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top) ,\\
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf s_i &=   (\mathrm{diag}(\lambda_i) + \mathbf a_i \mathbf b_i^\top) \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top \\
&=   (\mathrm{diag}(\lambda_i) + \mathbf a_i \mathbf b_i^\top)
\left[  (\mathrm{diag}(\lambda_{i-1}) + \mathbf a_{i-1} \mathbf b_{i-1}^\top)  \mathbf s_{i-2}
+ \mathbf k_{i-1} \mathbf v_{i-1}^\top \right] + \mathbf k_i \mathbf v_i^\top \\
&= \mathbf M_{j}^i \mathbf s_j + \sum_{s=j+1}^i \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top  \\
&= \mathbf M_{i}\mathbf s_0 + \sum_{s=1}^i \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top.

\end{aligned}
$$
下面分别讨论这两项的展开式。



### $\mathbf M_j^i$

$$
\mathbf M_j^i =\prod_{k=j+1}^i  (\mathrm{diag}(\lambda_k) + \mathbf a_k \mathbf b_k^\top), i\ge j+1.
$$

我们猜测：
$$
\mathbf M_j^i =\mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top.
$$
$i=j+1$时：
$$
\begin{aligned}
\mathbf M_j^{j+1}
& = (\mathrm{diag}(\lambda_{j+1})+ \mathbf a_{j+1} \mathbf b_{j+1}^\top) \\
&= \mathrm{diag}(\lambda_{j+1}) + \mathbf a_{j+1} \mathbf b_{j+1}^\top \\
& \triangleq \mathrm{diag}(\gamma_{j+1}/\gamma_j) + \mathbf a_{j+1} \mathbf u_{j+1}^\top, \\
\mathbf u_{j+1} &= \mathbf b_{j+1}.
\end{aligned}
$$
假设当$i$时结论成立，那么$i+1$时：
$$
\begin{aligned}
\mathbf M_j^{i+1}
& = (\mathbf I + \mathbf a_{i+1} \mathbf b_{i+1}^\top)\mathbf M_j^{i+1}  \\
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
因此：
$$
\mathbf u_{i+1}= \mathrm{diag}(\gamma_i/\gamma_j) \mathbf b_{i+1} +

\sum_{k=j+1}^i   \mathbf u_k\mathbf a_k^\top  \mathrm{diag}(\gamma_i/\gamma_k) \mathbf b_{i+1}.
$$


### $\sum_{s=j+1}^i \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top $

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
\sum_{s=j+1}^{j+1} \mathbf M_{s}^{j+1} \mathbf k_s \mathbf v_s^\top
=\sum_{t=j+1}^{j+1}  \mathrm{diag}(\gamma_{j+1}/\gamma_{j+1})

\mathbf k_t \mathbf v_t^\top
= \mathbf k_{j+1} \mathbf v_{j+1}^\top, \\
\mathbf p_{j+1}=\mathbf 0.
$$
假设当$i$时结论成立，那么$i+1$时：
$$
\begin{aligned}
\sum_{s=j+1}^{i+1} \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top
&=(\mathrm{diag}(\lambda_{i+1}) + \mathbf a_{i+1}\mathbf b_{i+1}^\top)\sum_{s=j+1}^{i} \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top
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
其中：
$$
\mathbf p_{i+1}=  \sum_{t=j+1}^i
\left(
 \mathbf v_t \mathbf k_t^\top +  \mathbf p_t\mathbf a_t^\top
\right)\mathrm{diag}(\gamma_i/\gamma_t)\mathbf b_{i+1}.
$$


### 递推

将之前的公式带入可得：
$$
\begin{aligned}
\mathbf s_i
&= \mathbf M_{j}^i \mathbf s_j + \sum_{s=j+1}^i \mathbf M_{s}^i \mathbf k_s \mathbf v_s^\top  \\
&=\left(  \mathrm{diag}(\gamma_i/\gamma_j) + \sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top\right)\mathbf s_j  +\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\left(
\mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf p_t^\top
\right) \\

&=  \mathrm{diag}(\gamma_i/\gamma_j)\mathbf s_j  +\sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)
\mathbf k_t \mathbf v_t^\top  + \left(\sum_{k=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_k)\mathbf a_k \mathbf u_k^\top\right) \mathbf s_j + \sum_{t=j+1}^i  \mathrm{diag}(\gamma_i/\gamma_t)\mathbf a_t \mathbf p_t^\top.



\end{aligned}
$$
写成矩阵形式，即为：
$$
\begin{aligned}
\Lambda_{i,j}&=\prod_{s=(i-1)c+1}^{ic+j}\lambda_s, \\
&= \gamma_{ic+j}/ \gamma_{(i-1)c},  \\

\Gamma_{i,j}&=\prod_{s=(i-1)c+j}^{ic}\lambda_s, \\
&= \gamma_{ic}/ \gamma_{(i-1)c+j-1},  \\


\mathbf {\bar Q}_i &= \mathbf Q_i \odot  {\Lambda} _i,   \\
\mathbf {\bar K}_i &= \mathbf K_i /  {\Lambda} _i,   \\
\mathbf {\bar A}_i &= \mathbf A_i /  {\Lambda} _i, \\
\mathbf{\tilde K}_i &=\mathbf K_i \odot \Gamma_i,   \\
\mathbf{\tilde A}_i &=\mathbf A_i \odot \Gamma_i,   \\

\mathbf O_i &=  \mathbf {\bar Q}_i \mathbf S_i    + [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ]
{\mathbf V}_i + [ [\bar{\mathbf Q}_i \bar{\mathbf A}_i^\top] \odot \mathbf M ]
\left[{\mathbf V}_i + \mathbf U_{i} \mathbf S_i  \right],  \\
\mathbf S_i  &=  \mathbf \Lambda_{i+1} \mathbf S_i + \mathbf {\tilde K}_{i+1} \mathbf V_{i+1}+ \mathbf {\tilde A}_{i+1}\mathbf P_{i+1}
+ \mathbf {\tilde A}_{i+1 }\mathbf U_{i+1} \mathbf S_i. \\

\end{aligned}
$$

最后，我们讨论$ \mathbf U_i,\mathbf P_i$的计算。
$$
\begin{aligned}
\mathbf u_{i+1} &= \mathrm{diag}(\gamma_i/\gamma_j) \mathbf b_{i+1} +

\sum_{k=j+1}^i   \mathbf u_k\mathbf a_k^\top  \mathrm{diag}(\gamma_i/\gamma_k) \mathbf b_{i+1}, \\

\mathbf p_{i+1} &=  \sum_{t=j+1}^i
\left(
 \mathbf v_t \mathbf k_t^\top +  \mathbf p_t\mathbf a_t^\top
\right)\mathrm{diag}(\gamma_i/\gamma_t)\mathbf b_{i+1}.

\end{aligned}
$$
写成chunk形式即为：
$$
\begin{aligned}
\mathbf {\bar A}_i &= \mathbf A_i \odot \Lambda_i,  \\
\mathbf {\bar B}_i &= \mathbf B_i \odot \Lambda_i,  \\
\mathbf {\tilde B}_i &= \mathbf B_i / \Lambda_i,  \\
\mathbf U_i &= \mathbf {\bar B}_i + \mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) \mathbf U_i,  \\
 &=(\mathbf I-\mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) )^{-1}\mathbf {\bar B}_i,  \\

\mathbf P_i &= \mathrm{tril}(\mathbf {\bar K}_i \mathbf{\tilde B}_i^\top, -1) \mathbf V_i +    \mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) \mathbf P_i, \\

&=(\mathbf I-\mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) )^{-1} \mathrm{tril}(\mathbf {\bar K}_i \mathbf{\tilde B}_i^\top, -1) \mathbf V_i  \\

&= (\mathbf I-\mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) )^{-1}
\left( \mathrm{tril}(\mathbf {\bar K}_i \mathbf{\tilde B}_i^\top, -1) -\mathbf I  +\mathbf I\right) \mathbf V_i \\

&=\mathbf V_i -  (\mathbf I-\mathrm{tril}(\mathbf {\bar A}_i \mathbf{\tilde B}_i^\top, -1) )^{-1} \mathbf V_i.
\end{aligned}
$$


## Backward

注意到反向的递推为（注意，这里第一行的$\mathbf {ds}_n$表示state梯度的输入）：
$$
\begin{aligned}
\mathbf {ds}_{n+1} &= \mathbf {ds}_n ,  \\
\mathbf {ds}_n  &= \mathbf {ds}_{n+1} + \mathbf{q}_n\mathbf {do}^\top_n, \\

\mathbf {ds}_t &= [\mathrm{diag}(\lambda_{i+1}) + \mathbf a_{i+1} \mathbf b_{i+1}^\top]  \mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {ds}_0&= [\mathrm{diag}(\lambda_{1}) + \mathbf a_{1} \mathbf b_{1}^\top] \mathbf {ds}_1,  \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dv}_t& = \mathbf k_t^\top \mathbf {ds}_t.
\end{aligned}
$$
假设我们定义fwd的函数为：
$$
\mathbf O, \bar {\mathbf s}= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \mathbf A, \mathbf B, \mathbf s, \mathrm{reverse}).
$$
其中$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e},\Lambda\in \mathbb R^{n\times d},\mathbf A,  \in \mathbb R^{n\times d}, \mathbf B \in \mathbb R^{n\times d},\mathbf O\in \mathbb R^{n\times e}, \mathbf s\in \mathbb R^{d\times e}$：

如果reverse = false:
$$
\begin{aligned}
\mathbf s_0 &=\mathbf s, \\
\mathbf s_t &= [\mathrm{diag}(\lambda_t)+ \mathbf a_t \mathbf b_t^\top]  \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&\triangleq  \mathbf a_t \odot \mathbf {s}_{t-1} + \mathbf k_t\mathbf v_t^\top, \\
t&=1,\ldots, n, \\
\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{e}, \\
\bar {\mathbf s} &= \mathbf s_n.
\end{aligned}
$$
如果reverse = true:
$$
\begin{aligned}
\mathbf {s}_{n+1} &= \mathbf {s} ,  \\
\mathbf {s}_n  &= \mathbf {s}_{n+1} + \mathbf{k}_n\mathbf {v}^\top_n, \\

\mathbf {s}_t &= [\mathrm{diag}(\lambda_{t+1}) + \mathbf a_{t+1} \mathbf b_{t+1}^\top]  \mathbf{s}_{t+1} + \mathbf{k}_t\mathbf {v}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {s}_0&= [\mathrm{diag}(\lambda_{1}) + \mathbf a_{1} \mathbf b_{1}^\top]\mathbf {s}_1, \\
\bar{\mathbf s}& = \mathbf s_0.

\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf O,  {\mathbf s}_n &= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \mathbf A, \mathbf B, \mathbf s, \mathrm{false}), \\

\mathbf {dQ}, {\mathbf s}_n &= f(\mathbf {dO}, \mathbf V, \mathbf K, \Lambda,  \mathbf A, \mathbf B,\mathbf s, \mathrm{false}), \\

\mathbf {dK},  {\mathbf {ds}_0} &= f(\mathbf {V}, \mathbf {dO}, \mathbf Q,  \Lambda,  \mathbf A, \mathbf B, \mathbf {ds}, \mathrm{true}), \\

\mathbf {dV},  {\mathbf {ds}_0} &= f(\mathbf {K}, \mathbf Q,\mathbf {dO} ,  \Lambda,  \mathbf A, \mathbf B, \mathbf {ds}, \mathrm{true}).
\end{aligned}
$$
所以我们可以用一个函数解决前向反向计算的问题，最后，我们讨论
