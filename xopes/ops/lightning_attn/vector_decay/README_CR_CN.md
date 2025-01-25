# Lightning Attention with Vector Decay(Chunk Recurrence)

## 回顾

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e}$，注意如果Decay为空，我们使用$\Lambda=1-\mathbf K, \Gamma=1-\mathbf V$（此时我们默认$0\le \mathbf K \le 1, 0\le \mathbf V \le 1$），我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_t &= (\lambda_t\gamma_t^\top)\odot   \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top, \\
\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{e}.
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

反向：
$$
\begin{aligned}
\mathbf {ds}_{n+1}& \in \mathbb R^{d\times e}, \\

\lambda_{n+1} &=\mathbf 1_d, \\
\gamma_{n+1} &=\mathbf 1_e,  \\
\mathbf q_0 &=\mathbf 0_d, \\
\mathbf{do}_0 &= \mathbf 0_e , \\

\mathbf {ds}_t
& = [\lambda_{t+1}\gamma_{t+1}^\top] \odot \mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\


\mathbf{dq}_t
&= \mathbf s_t \mathbf {do}_t,  \\

\mathbf{dk}_t
&=  \mathbf {ds}_t \mathbf v_t, \\

\mathbf{dv}_t
 &= \mathbf {ds}_t^\top \mathbf k_t .
\end{aligned}
$$



## Forward

我们记：
$$
\begin{aligned}
\mathbf X &= \left[\begin{matrix}
\mathbf X_1^\top  \\
\vdots \\
\mathbf X_k^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d},  \\

\mathbf X_i &\in \mathbb R^{c\times d},  \\
c &=n/k,  \\
[\mathbf X_i]_j&= x_{(i-1)c+j}, j=1,\ldots, c, \\
\mathbf X &\in \{\mathbf Q, \mathbf K, \mathbf V, \mathbf O,\mathbf \Gamma, \mathbf \Lambda \}, \\
\mathbf S_i &= \mathbf s_{ic}, i=0, \ldots, n/c, \\
[\mathbf M]_{ij}&=
\begin{cases}
1, & \text{if }  i \ge  j \\
0, & \text{if }  i < j
\end{cases}.
\end{aligned}
$$

注意到：
$$
\begin{aligned}
\mathbf s_{t+s}
&= \left(\odot_{i=t+1}^{t+s} \mathbf a_i \right)\odot  \mathbf s_t  + \sum_{j=t+1}^{t+s} \left(\odot_{i=j+1}^{t+s} \mathbf a_i \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \frac{\mathbf A_{t+s}}{\mathbf A_{t}}\odot \mathbf s_t  + \sum_{j=t+1}^{t+s} \frac{\mathbf A_{t+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top, \\

\mathbf o_{t+s}
&=
\mathbf q_{t+s}^\top \mathbf s_{t+s}  \\
&= \mathbf q_{t+s}^\top \left[ \frac{\mathbf A_{t+s}}{\mathbf A_{t}}\odot \mathbf s_t \right]  + \mathbf q_{t+s}^\top
\left( \sum_{j=t+1}^{t+s} \frac{\mathbf A_{t+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \right).
\end{aligned}
$$
以及：
$$
\begin{aligned}

\prod_{j=1}^t \lambda_j & = \alpha_t,  \\
\prod_{j=1}^t \gamma_j & =\beta_t, \\
\mathbf a_t &= \lambda_t \gamma_t^\top,  \\
\mathbf A_t &= \odot_{j=1}^t \mathbf a_j \\
&= \alpha_t \beta_t^\top.
\end{aligned}
$$
那么记：
$$
\begin{aligned}
\Pi_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
&= \alpha_{(i-1)c+j}/ \alpha_{(i-1)c} \\
\Rho_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\gamma_s, \\
&=  \beta_{(i-1)c+j}/ \beta_{(i-1)c}, \\
\frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}
&= \Pi_{i,s}\Rho_{i,s}^\top.

\end{aligned}
$$
取$t=ic$，那么：
$$
\begin{aligned}
\mathbf o_{ic+s}
&=
\mathbf q_{ic+s}^\top \mathbf s_{ic+s}  \\
&= \mathbf q_{ic+s}^\top \left[ \frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}\odot \mathbf s_{ic} \right]  + \mathbf q_{ic+s}^\top
\left( \sum_{j=ic+1}^{ic+s} \frac{\mathbf A_{ic+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \right) \\
&= \mathbf q_{ic+s}^\top \left[ \frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}\odot \mathbf S_{i} \right]  + \mathbf q_{ic+s}^\top
\left( \sum_{j=ic+1}^{ic+s} \frac{\mathbf A_{ic+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \right).
\end{aligned}
$$
对于第一项：
$$
\begin{aligned}
\mathbf q_{ic+s}^\top \left[ \frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}\odot \mathbf S_{i} \right]
&=
\mathbf q_{ic+s}^\top  \left[
(\Pi_{i,s}\Rho_{i,s}^\top)
\odot \mathbf S_i
\right] \\
&= [[\mathbf q_{ic+s}\odot \Pi_{i,s}]^\top \mathbf S_i ] \odot \Rho_{i,s}\\
&=  \left[\left[\left[
\mathbf Q_i \odot \Pi_i
\right] \mathbf S_i \right] \odot \Rho_i \right]_s.
\end{aligned}
$$
对于第二项：
$$
\begin{aligned}
 \mathbf q_{ic+s}^\top
\left( \sum_{j=ic+1}^{ic+s} \frac{\mathbf A_{ic+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \right)

&=  \mathbf q_{ic+s}^\top
\left( \sum_{j=ic+1}^{ic+s}
\left[\frac{\mathbf A_{ic+s}}{\mathbf A_{ic}} /  \frac{\mathbf A_{j}}{\mathbf A_{ic}} \right]\odot  \mathbf k_j\mathbf v_j ^\top \right) \\

&= \mathbf q_{ic+s}^\top
\left( \sum_{j=ic+1}^{ic+s}
\left[[(\Pi_{i,s}\Rho_{i,s}^\top)] /  (\Pi_{i,j-ic}\Rho_{i,j-ic}^\top) \right]\odot  \mathbf k_j\mathbf v_j ^\top \right) \\

&= \left[(\mathbf q_{ic+s} \odot \Pi_{i,s})^\top
\left( \sum_{j=ic+1}^{ic+s}
 (\mathbf k_j / \Pi_{i,j-ic}) (\mathbf v_j /\Rho_{i,j-ic}) ^\top \right) \right] \odot \Rho_{i,s} \\

&= \left[ \left(\left[\left[
\mathbf Q_i \odot \Pi_i
\right]  \left[\mathbf K_i / \Pi_i \right]^\top \odot \mathbf M\right] [\mathbf V_i/ \Rho_i] \right) \odot \Rho_i \right]_s.
\end{aligned}
$$
因此写成矩阵形式即为：
$$
\mathbf O_i=\left[\left[\left[
\mathbf Q_i \odot \Pi_i
\right] \mathbf S_i \right] \odot \Rho_i \right] +\left[ \left(\left[\left[
\mathbf Q_i \odot \Pi_i
\right]  \left[\mathbf K_i / \Pi_i \right]^\top \odot \mathbf M\right] [\mathbf V_i/ \Rho_i] \right) \odot \Rho_i \right].
$$
另一方面，考虑下式：
$$
\begin{aligned}
\mathbf s_{t+s}
&= \left(\odot_{i=t+1}^{t+s} \mathbf a_i \right)\odot  \mathbf s_t  + \sum_{j=t+1}^{t+s} \left(\odot_{i=j+1}^{t+s} \mathbf a_i \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \frac{\mathbf A_{t+s}}{\mathbf A_{t}}\odot \mathbf s_t  + \sum_{j=t+1}^{t+s} \frac{\mathbf A_{t+s}}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top,

\end{aligned}
$$
我们可得：
$$
\begin{aligned}
\Theta_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s, \\
&= \alpha_{ic}/ \alpha_{(i-1)c+j-1}, \\
\Phi_{i,j} &=  \prod_{s=(i-1)c+j}^{ic}\gamma_s, \\
&= \beta_{ic}/ \beta_{(i-1)c+j-1}, \\
\frac{\mathbf A_{ic}}{\mathbf A_{(i-1)c +j}}
&= \Theta_{i,j}\Phi_{i,j}^\top.

\end{aligned}
$$
对递推式取$t=ic, s=c$，
$$
\begin{aligned}
\mathbf s_{(i+1)c}
&= \left(\odot_{i=ic+1}^{(i+1)c} \mathbf a_i \right)\odot  \mathbf s_{ic}  + \sum_{j=ic+1}^{(i+1)c} \left(\odot_{t=j+1}^{(i+1)c} \mathbf a_t \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf s_{ic}  +\sum_{j=1}^{c}  \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic+j}}\odot  \mathbf k_{ic+j}\mathbf v_{ic+j}^\top, \\
\mathbf S_{i+1} &= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} +
\sum_{j=1}^{c}    (\mathbf k_{ic+j} \odot  \Theta_{i+1,j})(\mathbf v_{ic+j}\odot \Phi_{i+1,j}) ^\top \\
&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + [\mathbf K_{i+1} \odot \Theta_{i+1}]^\top [\mathbf V_{i+1} \odot \Phi_{i+1}].
\end{aligned}
$$

根据chunk level的递推公式，我们可以可以考虑如下更简单的递推：
$$
\begin{aligned}

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Pi_i,  \\
\bar{\mathbf K}_i &=  \mathbf K_i / \mathbf \Pi_i,  \\
\bar{\mathbf V}_i &=  \mathbf V_i / \mathbf \Rho_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i,  \\
\tilde{\mathbf V}_i &=  \mathbf V_i \odot \mathbf \Phi_i,  \\

\bar{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ]
\bar{\mathbf V}_i, \\
\mathbf O_i &= \bar{\mathbf O}_i \odot \mathbf\Rho_i, \\
\mathbf S_{i+1}&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + \tilde{\mathbf K}_{i+1} \tilde{\mathbf V}_{i+1}^\top \\
&\triangleq \Tau_i \odot \mathbf S_{i}+ \tilde{\mathbf K}_{i+1}  \tilde{\mathbf V}_{i+1}^\top.
\end{aligned}
$$
后续会在反向传播的时候使用。



## Backward

我们考虑$\mathbf {dS}_i$的递推，
$$
\begin{aligned}
\mathbf {dS}_{n+1} &\in \mathbb R^{d\times e}, \\
\mathbf{d\bar O}_i &= \mathbf{d O}_i \odot \mathbf \Rho_i,  \\
\mathbf {dS}_{i} &= \mathbf P_{i} \odot \mathbf {dS}_{i+1} + \mathbf{\bar Q}_i^\top \mathbf {d\bar O}_i, \\
\mathbf {d Q}_i &= \left[\mathbf{d\bar O}_i \mathbf S_i^\top + [[\mathbf{d\bar O}_i \mathbf{\bar V}_i ^\top]\odot \mathbf M]
\mathbf{\bar K_i} \right] \odot \mathbf \Pi_i, \\

\mathbf {d K}_i &= \left[\mathbf{\tilde V}_i \mathbf {dS}_i^\top \right] \odot \mathbf \Theta_i
+ \left[[[\mathbf{d\bar O}_i \mathbf{\bar V}_i^\top ]\odot \mathbf M]^\top \mathbf {\bar Q}_i\right] / \mathbf \Pi_i
,   \\
\mathbf {dV}_i &= \left[ \mathbf{\tilde K}_i \mathbf {dS}_i \right] \odot \Phi_i +

\left[ [[\mathbf{\bar Q}_i \mathbf {\bar K}_i] \odot \mathbf M ] \mathbf{d\bar O}_i \right] /\mathbf \Rho_i, \\

\mathbf{ds}_0 &= \mathbf {dS}_{0}.


\end{aligned}
$$


## 算法

在后续讨论中，我们将计算中包含$\mathbf M$的称为chunk内。

- 方案1：直接使用上述公式进行分块循环；
  - 问题：chunk内的计算有数值问题；
- 方案2：chunk内用循环；
  - 问题：chunk内比较慢；
- 方案3：chunk内通过并行+循环同时计算，然后做递归；
  - 问题：可能并行粒度不够；
- 方案4：对方案3进行两轮，
  - 第一轮使用$c=16$，大规模并行，然后递推；
  - 第二轮使用$c=128$，并行，然后递归；
