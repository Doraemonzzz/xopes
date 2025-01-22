

# Lightning Attention(scalar decay)

考虑$\lambda_i ,\gamma_i$都是标量的情况，此时$\Lambda\in \mathbb R^{n\times 1},\Gamma \in \mathbb R^{n\times 1}$，另一方面，记：
$$
\begin{aligned}
\ [\mathbf M]_{ij}&=
\begin{cases}
1, & \text{if }  i \ge  j \\
0, & \text{if }  i < j
\end{cases}
\end{aligned}
$$


## Chunk Recurrence

### Forward

回顾定义：
$$
\begin{aligned}
\Gamma_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s \\
&= \alpha_{(i-1)c+j}/ \alpha_{(i-1)c}, \\
\Delta_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\gamma_s, \\
&=  \beta_{(i-1)c+j}/ \beta_{(i-1)c}, \\
\frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}
&= \Gamma_{i,s}\Delta_{i,s}^\top, \\
\Theta_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s \\
&= \alpha_{ic}/ \alpha_{(i-1)c+j-1}, \\
\Phi_{i,j} &=  \prod_{s=(i-1)c+j}^{ic}\gamma_s \\
&= \beta_{ic}/ \beta_{(i-1)c+j-1}, \\
\frac{\mathbf A_{ic}}{\mathbf A_{(i-1)c +j}}
&= \Theta_{i,j}\Phi_{i,j}^\top.

\end{aligned}
$$
以及前向公式：
$$
\begin{aligned}

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Gamma_i,  \\
\bar{\mathbf K}_i &=  \mathbf K_i / \mathbf \Gamma_i,  \\
\bar{\mathbf V}_i &=  \mathbf V_i / \mathbf \Delta_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i,  \\
\tilde{\mathbf V}_i &=  \mathbf V_i \odot \mathbf \Psi_i,  \\

\bar{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ]
\bar{\mathbf V}_i, \\
\mathbf O_i &= \bar{\mathbf O}_i \odot \mathbf\Delta_i, \\
\mathbf S_{i+1}&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + \tilde{\mathbf K}_{i+1} \tilde{\mathbf V}_{i+1}^\top \\
&\triangleq \Rho_i \odot \mathbf S_{i}+ \tilde{\mathbf K}_{i+1}  \tilde{\mathbf V}_{i+1}^\top.
\end{aligned}
$$
注意到此时：
$$
\begin{aligned}
\ [ \bar{\mathbf Q}_i \bar{\mathbf K}_i^\top ] \odot \mathbf M
&= \ [ {\mathbf Q}_i {\mathbf K}_i^\top ] \odot \mathbf  M (\mathbf \Gamma_i) ,  \\
\mathbf M(\mathbf \Gamma_i, \mathbf \Gamma_j) &= \left[\exp( \log\mathbf \Gamma_i[:,\mathrm{None}] - \log\mathbf \Gamma_j[\mathrm{None},:] )\right] , i> j, \\
\mathbf M(\mathbf \Gamma_i) &=
\mathbf M(\mathbf \Gamma_i, \mathbf \Gamma_i) \\
&= \left[\exp( \log\mathbf \Gamma_i[:,\mathrm{None}] - \log\mathbf \Gamma_j[\mathrm{None},:] )\right] \odot \mathbf M , i= j.

\end{aligned}
$$
因此递推公式变成：
$$
\begin{aligned}

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Gamma_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i,  \\
\tilde{\mathbf V}_i &=  \mathbf V_i \odot \mathbf \Psi_i,  \\

\bar{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M(\mathbf \Gamma_i) ]
\bar{\mathbf V}_i, \\
\mathbf O_i &= \bar{\mathbf O}_i \odot \mathbf\Delta_i, \\
\mathbf S_{i+1}&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + \tilde{\mathbf K}_{i+1} \tilde{\mathbf V}_{i+1}^\top \\
&\triangleq \Rho_i \odot \mathbf S_{i}+ \tilde{\mathbf K}_{i+1}  \tilde{\mathbf V}_{i+1}^\top.
\end{aligned}
$$

注意到$\lambda,\gamma$都是标量，因此：
$$
\begin{aligned}
\mathbf O_i &= \bar{\mathbf O}_i \odot \mathbf\Delta_i \\
&= \left[
 \bar{\mathbf Q}_i  \mathbf S_i  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M(\mathbf \Gamma_i) ]
\bar{\mathbf V}_i
\right] \odot \mathbf \Delta_i \\
&=  [\bar{\mathbf Q}_i  \mathbf S_i]\odot \mathbf \Delta_i +
\left[[ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M(\mathbf \Gamma_i)] ( \mathbf V_i / \mathbf \Delta_i)\right] \odot \mathbf \Delta_i \\
&=  \mathrm{diag}(\mathbf \Delta_i) [\bar{\mathbf Q}_i  \mathbf S_i] +
 \mathrm{diag}(\mathbf \Delta_i) [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf M(\mathbf \Gamma_i)]  \mathrm{diag}(\mathbf \Delta_i^{-1})\mathbf V_i \\
 &= [[{\mathbf Q}_i \odot \mathbf \Gamma_i \odot \mathbf \Delta_i] \mathbf S_i] +
[ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M(\mathbf \Gamma_i) \odot \mathbf  M(\Delta_i)] \mathbf V_i.
\end{aligned}
$$
因此（注意此处$\mathbf {\bar Q}_i$的定义修改了）：
$$
\begin{aligned}

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Gamma_i \odot \mathbf \Delta_i ,  \\
\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i,  \\
\tilde{\mathbf V}_i &=  \mathbf V_i \odot \mathbf \Psi_i,  \\
{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf M(\mathbf \Gamma_i)
\odot \mathbf M ( \mathbf \Delta_i)]
{\mathbf V}_i, \\
\mathbf S_{i+1}&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + \tilde{\mathbf K}_{i+1} \tilde{\mathbf V}_{i+1}^\top \\
&\triangleq \Rho_i \odot \mathbf S_{i}+ \tilde{\mathbf K}_{i+1}  \tilde{\mathbf V}_{i+1}^\top.
\end{aligned}
$$


### Backward

回顾前向反向公式：
$$
\begin{aligned}

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot \mathbf \Gamma_i,  \\
\bar{\mathbf K}_i &=  \mathbf K_i / \mathbf \Gamma_i,  \\
\bar{\mathbf V}_i &=  \mathbf V_i / \mathbf \Delta_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \mathbf \Theta_i,  \\
\tilde{\mathbf V}_i &=  \mathbf V_i \odot \mathbf \Psi_i,  \\

\bar{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ]
\bar{\mathbf V}_i, \\
\mathbf O_i &= \bar{\mathbf O}_i \odot \mathbf\Delta_i, \\
\mathbf S_{i+1}&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + \tilde{\mathbf K}_{i+1} \tilde{\mathbf V}_{i+1}^\top \\
&\triangleq \Rho_i \odot \mathbf S_{i}+ \tilde{\mathbf K}_{i+1}  \tilde{\mathbf V}_{i+1}^\top, \\
\mathbf {dS}_{n+1} & = \mathbf 0 \in \mathbb R^{d\times e}, \\
\mathbf{d\bar O}_i &= \mathbf{d O}_i \odot \mathbf \Delta_i,  \\
\mathbf {dS}_{i} &= \mathbf P_{i} \odot \mathbf {dS}_{i+1} + \mathbf{\bar Q}_i^\top \mathbf {d\bar O}_i, \\
\mathbf {d Q}_i &= \left[\mathbf{d\bar O}_i \mathbf S_i^\top + [[\mathbf{d\bar O}_i \mathbf{\bar V}_i ^\top]\odot \mathbf M]
\mathbf{\bar K_i} \right] \odot \mathbf \Gamma_i, \\

\mathbf {d K}_i &= \left[\mathbf{\tilde V}_i \mathbf {dS}_i^\top \right] \odot \mathbf \Theta_i
+ \left[[[\mathbf{d\bar O}_i \mathbf{\bar V}_i^\top ]\odot \mathbf M] \mathbf {\bar Q}_i\right] / \mathbf \Gamma_i
,   \\
\mathbf {dV}_i &= \left[ \mathbf{\tilde K}_i \mathbf {dS}_i \right] \odot \Psi_i +

\left[ [[\mathbf{\bar Q}_i \mathbf {\bar K}_i] \odot \mathbf M ] \mathbf{d\bar O}_i \right] /\mathbf \Delta_i.


\end{aligned}
$$

注意到：
$$
\begin{aligned}
\ [\mathbf{d\bar O}_i \mathbf{\bar V}_i ^\top]\odot \mathbf M
&= [\mathbf{d O}_i \mathbf{V}_i ^\top]\odot \mathbf M(\mathbf \Delta_i), \\

[\mathbf{\bar Q}_i \mathbf {\bar K}_i] \odot \mathbf M
&=  [\mathbf{Q}_i \mathbf{K}_i ^\top]\odot \mathbf M(\mathbf \Gamma_i).
\end{aligned}
$$
因此计算公式变成：
$$
\begin{aligned}
\mathbf {dS}_{n+1} & = \mathbf 0 \in \mathbb R^{d\times e}, \\
\mathbf{d\bar O}_i &= \mathbf{d O}_i \odot \mathbf \Delta_i,  \\
\mathbf {dS}_{i} &= \mathbf P_{i} \odot \mathbf {dS}_{i+1} + \mathbf{\bar Q}_i^\top \mathbf {d\bar O}_i, \\
\mathbf {d Q}_i &= \left[\mathbf{d\bar O}_i \mathbf S_i^\top + [[\mathbf{dO}_i \mathbf{V}_i ^\top]\odot  \mathbf M(\mathbf \Delta_i)]
\mathbf{\bar K_i} \right] \odot \mathbf \Gamma_i, \\

\mathbf {d K}_i &= \left[\mathbf{\tilde V}_i \mathbf {dS}_i^\top \right] \odot \mathbf \Theta_i
+ \left[[[\mathbf{d O}_i \mathbf{V}_i^\top ]\odot  \mathbf M(\mathbf \Delta_i)] \mathbf {\bar Q}_i\right] / \mathbf \Gamma_i
,   \\
\mathbf {dV}_i &= \left[ \mathbf{\tilde K}_i \mathbf {dS}_i \right] \odot \Psi_i +

\left[ [[\mathbf{ Q}_i \mathbf {K}_i] \odot  \mathbf M(\mathbf \Gamma_i)] \mathbf{d\bar O}_i \right] /\mathbf \Delta_i.


\end{aligned}
$$
对于后三项，我们有：
$$
\begin{aligned}
\mathbf {d Q}_i &= \left[\mathbf{d\bar O}_i \mathbf S_i^\top + [[\mathbf{dO}_i \mathbf{V}_i ^\top]\odot  \mathbf M(\mathbf \Delta_i)]
\mathbf{\bar K_i} \right] \odot \mathbf \Gamma_i \\
&= [\mathbf{d\bar O}_i \mathbf S_i^\top ]\odot  \mathbf \Gamma_i +
\left[[[\mathbf{dO}_i \mathbf{V}_i ^\top]\odot  \mathbf M(\mathbf \Delta_i)](\mathbf K_i / \mathbf \Gamma_i)   \right] \odot \mathbf \Gamma_i \\
&= [\mathbf{d O}_i  \odot \mathbf \Gamma_i  \odot \mathbf \Delta_i] \mathbf S_i^\top +
[[\mathbf{dO}_i \mathbf{V}_i ^\top]\odot \mathbf M(
\mathbf \Gamma_i)\odot  \mathbf M(\mathbf \Delta_i)]\mathbf K_i, \\



\mathbf {d K}_i &= \left[\mathbf{\tilde V}_i \mathbf {dS}_i^\top \right] \odot \mathbf \Theta_i
+ \left[[[\mathbf{d O}_i \mathbf{V}_i^\top ]\odot  \mathbf M(\mathbf \Delta_i)] \mathbf {\bar Q}_i\right] / \mathbf \Gamma_i
  \\
&=  [\mathbf{ V}_i\odot \mathbf \Psi_i \odot \Theta_i] \mathbf {dS}_i^\top  +
[[\mathbf{dO}_i \mathbf{V}_i ^\top]\odot \mathbf M(
-\mathbf \Gamma_i)\odot  \mathbf M(\mathbf \Delta_i)]\mathbf K_i,


\\
\mathbf {dV}_i &= \left[ \mathbf{\tilde K}_i \mathbf {dS}_i \right] \odot \Psi_i +

\left[ [[\mathbf{ Q}_i \mathbf {K}_i] \odot  \mathbf M(\mathbf \Gamma_i)] \mathbf{d\bar O}_i \right] /\mathbf \Delta_i\\
&= [\mathbf K_i \odot  \mathbf \Psi_i \odot  \mathbf \Theta_i] \mathbf {dS}_i +
[[\mathbf{ Q}_i \mathbf {K}_i] \odot  \mathbf M(\mathbf \Gamma_i) \odot  \mathbf M(-\mathbf \Delta_i)] \mathbf{dO}_i.

\end{aligned}
$$


## 左乘

回顾公式：
$$
\begin{aligned}
\mathbf o_t
&=\mathbf q_t^\top \mathbf s_t \\
&= \mathbf q_t^\top [ \mathbf A_t\odot \mathbf s_0]
+\mathbf q_t^\top
\left( \sum_{j=1}^t \frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top  \right) \\

&=\mathbf q_t^\top [ \mathbf A_t\odot \mathbf s_0] +
\sum_{j=1}^t \mathbf q_t^\top  \frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \\

&= \mathbf q_t^\top [ \alpha_t \beta_t^\top \odot \mathbf s_0] +
\left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \odot \beta_t \\
&= (\alpha_t \beta_t)\mathbf q_t^\top  \mathbf s_0 +
\left[ \mathbf q_t ^\top \sum_{j=1}^t \mathbf k_j (\alpha_t / \alpha_j)(\beta_t /\beta_j)\mathbf v_j^\top \right].


\end{aligned}
$$
那么：
$$
\begin{aligned}
{\mathbf O}
&=
[[{\mathbf Q} \odot \mathbf \Gamma \odot \mathbf \Delta] \mathbf S_0] +
[ [{\mathbf Q} {\mathbf K}^\top] \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)] \mathbf V \\
&\triangleq \bar {\mathbf Q} \mathbf S_0 + \bar {\mathbf M} \mathbf V.
\end{aligned}
$$
计算梯度可得：
$$
\begin{aligned}
\mathbf {dV}
&= \bar {\mathbf M}^\top \mathbf {dO},  \\

\mathbf{d{\bar M}}
&= \mathbf{dO}\mathbf V^\top ,\\

\mathbf{d{\bar Q}}
&= \mathbf{dO}\mathbf S_0^\top,\\

\mathbf{d[\mathbf Q\mathbf K^\top]}
&= \mathbf{d{\bar M}} \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta),  \\

\mathbf {dK} &= \left[ \mathbf{d{\bar M}} \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)\right]^\top \mathbf Q \\
&= \left[ [\mathbf{dO}\mathbf V^\top ] \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)\right]^\top \mathbf Q, \\

\mathbf {dQ} &= [ \mathbf{dO}\mathbf S_0^\top] \odot \mathbf \Gamma \odot \mathbf \Delta + \left[ [\mathbf{dO}\mathbf V^\top ] \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)\right] \mathbf K, \\

\mathbf {ds}_0 &= [\mathbf Q \odot \Alpha]^\top [\mathbf{dO}\odot \Beta].



\end{aligned}
$$
因此分块算法如下。



### Forwad

$$
\begin{aligned}
{\mathbf O}_i
&=\bar {\mathbf Q}_i \mathbf S_0 + \sum_{j\le i}\bar {\mathbf M}_{ij} \mathbf V_j \\
&= \bar {\mathbf Q}_i \mathbf S_0 + \sum_{j< i}
\left[ [ \mathbf Q_i \mathbf K_j^\top ] \odot
\mathbf M(\mathbf \Gamma_i, \mathbf \Gamma_j)
\odot \mathbf M(\mathbf \Delta_i, \mathbf \Delta_j)
\right] \mathbf V_j  \\

&+
\left[ [ \mathbf Q_i \mathbf K_i^\top ] \odot
\mathbf M(\mathbf \Gamma_i)
\odot  \mathbf M(\mathbf \Delta_i)
\right] \mathbf V_i.

\end{aligned}
$$



### Backward

$$
\begin{aligned}
\mathbf {dV}_i
&= \sum_{j\ge i}[\bar {\mathbf M}^\top]_{ij} \mathbf {dO}_j \\
&= \sum_{j\ge i}[\bar {\mathbf M}_{ji}]^\top \mathbf {dO}_j \\
&= \sum_{j > i}\left[ [ \mathbf Q_j \mathbf K_i^\top ] \odot
\mathbf M(\mathbf \Gamma_j, \mathbf \Gamma_i)
\odot \mathbf M(\mathbf \Delta_j, \mathbf \Delta_i)
\right]^\top \mathbf {dO}_j  \\
&+
\left[ [ \mathbf Q_i \mathbf K_i^\top ] \odot
\mathbf M(\mathbf \Gamma_i)
\odot \mathbf M(\mathbf \Delta_i)
\right]  \mathbf {dO}_i,  \\

\mathbf {dK}_i &=
\sum_{j\ge i}\left[ [\mathbf{dO}\mathbf V^\top ] \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)\right]^\top_{ij} \mathbf Q_j  \\
&=
\sum_{j> i}\left[ [\mathbf{dO}_j\mathbf V^\top_i ]
\odot
\mathbf M(\mathbf \Gamma_j, \mathbf \Gamma_i)
\odot \mathbf M(\mathbf \Delta_j, \mathbf \Delta_i)

\right]^\top\mathbf Q_j  \\
&+
\left[ [\mathbf{dO}_i\mathbf V_i^\top ]
\odot \mathbf M(\mathbf \Gamma_i)
\odot \mathbf M(\mathbf \Delta_i) \right]^\top\mathbf Q_i, \\

\mathbf{dQ}_i
&=  [ \mathbf{dO}_i\mathbf S_0^\top] \odot \mathbf \Gamma_i \odot \mathbf \Delta_i+
\sum_{j\le i}
\left[ [\mathbf{dO}\mathbf V^\top ] \odot \mathbf  M(\mathbf \Gamma) \odot \mathbf  M(\Delta)\right]_{ij} \mathbf K_j \\

&=  [ \mathbf{dO}_i\mathbf S_0^\top] \odot \mathbf \Gamma_i \odot \mathbf \Delta_i+
\sum_{j< i}
\left[ [\mathbf{dO}_i\mathbf V^\top_j ]
\odot
\mathbf M(\mathbf \Gamma_i, \mathbf \Gamma_j)
\odot \mathbf M(\mathbf \Delta_i, \mathbf \Delta_j)

\right]
\mathbf K_j \\
&+

\left[ [\mathbf{dO}_i\mathbf V^\top_i ]
\odot
\mathbf M(\mathbf \Gamma_i)
\odot \mathbf M(\mathbf \Delta_i)
\right]
\mathbf K_j, \\


\mathbf {ds}_0 &= [\mathbf Q \odot \Alpha]^\top [\mathbf{dO}\odot \Beta].


\end{aligned}
$$
