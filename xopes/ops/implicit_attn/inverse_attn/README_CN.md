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

注意到：
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



### Chunk形式

后续记：
$$
\mathbf A_{ts} =  (\mathbf Q_t \mathbf K_s^\top) \odot \mathbf M_{ts}.
$$
回顾之前的符号：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\
\bar{\mathbf K}_i &=  \mathbf K_i / \Lambda_i,  \\


\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_{i-1}  + [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ]
{\mathbf V}_i,  \\
\mathbf S_{i+1}&= \Lambda_{i+1, c} \mathbf S_{i}+ \tilde{\mathbf K}_{i+1} {\mathbf V}_{i+1}^\top.
\end{aligned}
$$
注意到在当前记号下
$$
\mathbf A_{ts} =  (\mathbf Q_t \mathbf K_s^\top) \odot \mathbf M_{ts}= [ [\bar{\mathbf Q}_i \bar{\mathbf K}_i^\top] \odot \mathbf M ].
$$
那么根据之前的公式可得：
$$
\begin{aligned}

\mathbf V_t &= \mathbf A_{tt}^{-1}
\left(
\mathbf O_t - \sum_{s=1}^{t-1} \mathbf A_{ts} \mathbf V_s
\right) \\

&= \mathbf A_{tt}^{-1}\left(
\mathbf O_t -  \mathbf {\bar Q}_t \mathbf S_{t-1}
\right),  \\

\mathbf S_{t} &= \Lambda_{t, c} \mathbf S_{t-1} + \tilde{\mathbf K}_t^\top  \mathbf V_t.

\end{aligned}
$$
对于$\mathbf A_{tt}^{-1}$，使用雅可比迭代并行计算$\mathbf A_{tt}^{-1}$。






## Backward

在反向传播时，给定$\mathbf {dV}$，计算$\mathbf {dQ}, \mathbf {dK}, \mathbf {dO}$。

注意到Forward可以改写为：
$$
\begin{aligned}
 \mathbf O = \left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right] \mathbf V.

\end{aligned}
$$
根据之前结论可得：
$$
\begin{aligned}
\mathbf {dQ} &= [(\mathbf {dO} \mathbf V ^\top )  \odot \mathbf M ] \mathbf {K}, \\

\mathbf {dK} &= [(\mathbf V \mathbf {dO}^\top )  \odot \mathbf M^\top ] \mathbf Q, \\

\mathbf {dV} &=  \left[
\left( \mathbf K \mathbf Q^\top\right) \odot \mathbf M^\top
\right] \mathbf {dO}.
\end{aligned}
$$
所以我们可以先计算：
$$
\mathbf {dO}=  \left[
\left( \mathbf K \mathbf Q^\top\right) \odot \mathbf M^\top
\right]^{-1} \mathbf {dV}.
$$
然后计算$\mathbf {dQ}, \mathbf {dK}$。

另一方面：
$$
\begin{aligned}
\mathbf {d}\log \alpha_t
& =\mathbf q_t \odot  \mathbf {dq}_t -  \mathbf k_t \odot  \mathbf {dk}_t,  \\
\mathbf d \log \lambda_t
&= [\mathbf s_n \odot \mathbf {ds}_n]1_e + \sum_{j\ge t} \mathbf d \log \alpha_j.

\end{aligned}
$$



### 递推形式

回顾之前的递推形式：
$$
\begin{aligned}
\mathbf {ds}_{n+1} &= \mathbf {ds}_n ,  \\
\mathbf {ds}_n  &= \mathbf {ds}_{n+1} + \mathbf{q}_n\mathbf {do}^\top_n, \\

\mathbf {ds}_t &= \lambda_{t+1}\mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {ds}_0&= \lambda_1 \mathbf {ds}_1,  \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dv}_t& = \mathbf k_t^\top \mathbf {ds}_t.
\end{aligned}
$$

现在已知$\mathbf {dV}$，需要计算$\mathbf {dQ}, \mathbf {dK}, \mathbf {dV}$：
$$
\begin{aligned}

\mathbf {ds}_t &= \lambda_{t+1}\mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\

\mathbf{q}_t\mathbf {do}^\top_t &= \mathbf {ds}_t -  \lambda_{t+1}\mathbf{ds}_{t+1}, \\

(\mathbf k_t^\top \mathbf{q}_t)\mathbf {do}^\top_t &=
\mathbf k_t^\top \mathbf {ds}_t - \lambda_{t+1} \mathbf k_t^\top \mathbf{ds}_{t+1}， \\

\mathbf {do}^\top_t &= (\mathbf k_t^\top \mathbf {ds}_t - \lambda_{t+1} \mathbf k_t^\top \mathbf{ds}_{t+1}) /(\mathbf k_t^\top \mathbf{q}_t)  \\
&= ( \mathbf {dv}_t - \lambda_{t+1} \mathbf k_t^\top \mathbf{ds}_{t+1}) /(\mathbf q_t^\top \mathbf{k}_t), \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top.
\end{aligned}
$$



### Chunk

####  $\mathbf {dO}$

记：
$$
\mathbf {dA}_{tt}= [[\mathbf{ K}_t \mathbf{ Q}_t^\top ] \odot \mathbf M(\Lambda_t)^\top ]
$$
回顾之前的计算公式：
$$
\begin{aligned}



\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\
\tilde{\mathbf K}_i &=  \mathbf K_i \odot  \bar  \Lambda_i,  \\


\mathbf {d  V}_i &= \mathbf{\tilde K}_i \mathbf {dS}_{i-1}
+ [[\mathbf{ K}_i \mathbf{ Q}_i^\top ] \odot \mathbf M(\Lambda_i)^\top ] \mathbf {d O}_i,\\


\mathbf {dS}_{i}&=

 \Lambda_{i, c}\mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$
那么：
$$
\begin{aligned}
 \mathbf {d O}_t
 &= \mathbf {dA}_{tt}^{-1}
 \left( \mathbf{dV}_t -  \mathbf{\tilde K}_t \mathbf {dS}_{t-1} \right),  \\

 \mathbf {dS}_{t}&=\Lambda_{t, c}\mathbf {dS}_{t+1}+ {\mathbf {\bar Q}}_{t} {\mathbf {d O}}_{t}^\top.
\end{aligned}
$$



#### $\mathbf {dQ}$

回顾之前的计算公式：
$$
\begin{aligned}

{\mathbf {d \bar O}}_i &=  \mathbf {dO}_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\


{\mathbf {d Q}}_i&= \mathbf{d \bar O}_i  \mathbf S_{i-1}^\top  + [ [\mathbf{d O}_i {\mathbf V}_i^\top] \odot \mathbf M(\Lambda_i) ]
{\mathbf K}_i, \\

\mathbf S_{i+1}&= \Lambda_{i+1, c} \mathbf S_{i}+ \tilde{\mathbf K}_{i+1} {\mathbf V}_{i+1}^\top.
\end{aligned}
$$

此时所有元素都已知，直接求解即可。



#### $\mathbf{dK} $

回顾之前的计算公式：
$$
\begin{aligned}





\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\

{\mathbf {\bar V}_i}  &= \mathbf V_i  \odot \bar \Lambda_i , \\

\mathbf {d  K}_i &= \mathbf{ \bar V}_i \mathbf {dS}_{i-1}^\top
+ [[ \mathbf{ V}_i \mathbf{d O}_i^\top ]\odot \mathbf M(\bar\Lambda_i)^\top] \mathbf { Q}_i,\\

\mathbf {dS}_{i}&=

 \Lambda_{i+1, c}\mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$

此时所有元素都已知，直接求解即可。
