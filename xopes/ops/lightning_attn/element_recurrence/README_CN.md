

# Lightning Attention with Element-wise Recurrence

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times d}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：

$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d}, \\
\mathbf s_i &= \lambda_i \odot \mathbf s_{i-1} + \mathbf k_i\odot \mathbf v_i, \\
\mathbf o_i &= \mathbf q_i \odot \mathbf s_i.
\end{aligned}
$$

返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d}.
$$



## Sequential Recurrence

### Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times d}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：

$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d}, \\
\mathbf s_i &= \lambda_i \odot \mathbf s_{i-1} + \mathbf k_i\odot \mathbf v_i, \\
\mathbf o_i &= \mathbf q_i \odot \mathbf s_i.
\end{aligned}
$$

返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d}.
$$



### Backward

给定$\mathbf {do}_1,\ldots, \mathbf {do}_n, \mathbf {ds}_n$，参考vector decay的结论可得：
$$
\begin{aligned}
\mathbf{ds}_t &= \lambda_{t+1} \odot \mathbf{ds}_{t+1} + \mathbf q_t \odot \mathbf{do}_t, \\
\mathbf{dq}_t

&=\mathbf s_t \odot \mathbf {do}_t,  \\

\mathbf{dk}_t

&=  \mathbf {ds}_t \odot \mathbf v_t, \\

\mathbf{dv}_t

 &= \mathbf {ds}_t \odot \mathbf k_t.
\end{aligned}
$$
这里我们记$\lambda_{n+1}=1, \mathbf {ds}_{n+1}=\mathbf {ds}$，那么：
$$
\begin{aligned}
\mathrm{d}\lambda_n
&= \frac{\partial l}{\partial \lambda_n} \\
&=  \frac{\partial l}{\partial \mathbf s_n} \frac {\partial \mathbf s_n} {\partial \lambda_n} \\
&= \mathbf {ds}_n \odot \mathbf s_{n-1}, \\

\mathrm{d}\log\lambda_n
&= \mathrm{d}\lambda_n \odot \lambda_n \\
&= \mathbf {ds}_n \odot \mathbf s_{n-1}  \odot \lambda_n .

\end{aligned}
$$


## 统一形式

同vector decay，假设我们定义fwd的函数为：
$$
\mathbf O, \bar {\mathbf s}= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \mathbf s, \mathrm{reverse}).
$$
其中$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e},\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e},\mathbf O\in \mathbb R^{n\times e}, \mathbf s\in \mathbb R^{d\times e}$：

如果reverse = false:
$$
\begin{aligned}
\mathbf s_0 &=\mathbf s, \\
\mathbf s_t &= \lambda_t \odot   \mathbf s_{t-1} + \mathbf k_t\odot \mathbf v_t, \\
t&=1,\ldots, n, \\
\mathbf o_t &= \mathbf q_t \odot \mathbf s_t \in \mathbb R^{e}, \\
\bar {\mathbf s} &= \mathbf s_n.
\end{aligned}
$$
如果reverse = true:
$$
\begin{aligned}
\mathbf {s}_{n+1} &= \mathbf {s} ,  \\
\mathbf {s}_n  &= \mathbf {s}_{n+1} + \mathbf{k}_n \odot \mathbf {v}, \\

\mathbf {s}_t &= \lambda_{t+1} \odot \mathbf{s}_{t+1} + \mathbf{k}_t \odot \mathbf {v}_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {s}_0&= \lambda_1 \odot\mathbf {s}_1, \\
\bar{\mathbf s}& = \mathbf s_0.
\end{aligned}
$$

那么：
$$
\begin{aligned}
\mathbf O,  {\mathbf s}_n &= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda,\mathbf s, \mathrm{false}), \\

\mathbf {dQ}, {\mathbf s}_n &= f(\mathbf {dO}, \mathbf V, \mathbf K, \Lambda, \mathbf s, \mathrm{false}), \\

\mathbf {dK},  {\mathbf {ds}_0} &= f(\mathbf {V}, \mathbf {dO}, \mathbf Q,\Lambda, \mathbf {ds}, \mathrm{true}), \\

\mathbf {dV},  {\mathbf {ds}_0} &= f(\mathbf {K}, \mathbf Q,\mathbf {dO} , \Lambda, \mathbf {ds}, \mathrm{true}).
\end{aligned}
$$


## Chunk Recurrence

### Forward

首先使用vector decay下的结论：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_{i-1}  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M (\Lambda_i) ]
{\mathbf V}_i,  \\
\mathbf S_{i}&= \Lambda_{i, c} \mathbf S_{i-1}+ \tilde{\mathbf K}_{i} {\mathbf V}_{i}^\top.
\end{aligned}
$$

接着进行特殊场景下的优化（head dim=1）：
$$
\begin{aligned}
\ [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M (\Lambda_i) ]
{\mathbf V}_i
&= \mathbf Q_i \odot \mathrm{cumsum}(\tilde {\mathbf K}_i \odot \mathbf V_i, \text{reverse = False}) , \\

\mathbf S_{i}&= \Lambda_{i, c} \mathbf S_{i-1}+  \mathrm{sum}(\tilde {\mathbf K}_{i+1}\odot \mathbf V_i).
\end{aligned}
$$
最后结果如下：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf O}_i&=  \bar{\mathbf Q}_i \odot \mathbf S_{i-1}  + \mathbf Q_i \odot \mathrm{cumsum}(\tilde {\mathbf K}_i \odot \mathbf V_i, \text{reverse = False}),  \\
\mathbf S_{i}&= \Lambda_{i, c} \odot \mathbf S_{i-1}+  \mathrm{sum}(\tilde {\mathbf K}_{i}\odot \mathbf V_i).
\end{aligned}
$$




### Backward

#### $\mathbf{dQ}$

首先使用vector decay下的结论：
$$
\begin{aligned}

\bar{\mathbf K}_i &=  \mathbf K_i / \mathbf \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\


{\mathbf {d \bar Q}}_i&= \mathbf{d O}_i  \mathbf S_{i-1}^\top  + [ [\mathbf{d O}_i {\mathbf V}_i^\top] \odot \mathbf M ]
\bar{\mathbf K}_i, \\
{\mathbf {d Q}}_i &={\mathbf {d \bar Q}}_i \odot \mathbf\Lambda_i, \\

\mathbf S_{i}&= \Lambda_{i, c} \mathbf S_{i-1}+ \tilde{\mathbf K}_{i} {\mathbf V}_{i}^\top.
\end{aligned}
$$
接着进行特殊场景下的优化（head dim=1）：
$$
\begin{aligned}

{\mathbf {d \bar O}}_i &=  \mathbf O_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\


{\mathbf {d Q}}_i&= \mathbf{d \bar O}_i \odot  \mathbf S_{i-1}  + \mathbf{dO}_i \odot \mathrm{cumsum}(\tilde {\mathbf K}_i \odot \mathbf V_i, \text{reverse = False}), \\

\mathbf S_{i}&= \Lambda_{i, c}\odot \mathbf S_{i-1}+  \mathrm{sum}(\tilde {\mathbf K}_{i}\odot \mathbf V_i).
\end{aligned}
$$


#### $\mathbf{dK}$

首先使用vector decay下的结论：
$$
\begin{aligned}

\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\

{\mathbf {\bar V}_i}  &= \mathbf V_i  \odot \bar \Lambda_i , \\

\mathbf {d  K}_i &= \mathbf{ \bar V}_i \mathbf {dS}_{i+1}^\top
+ [[ \mathbf{ V}_i \mathbf{d O}_i^\top ]\odot \mathbf M(\bar\Lambda_i)^\top] \mathbf { Q}_i,\\

\mathbf {dS}_{i}&=

 \Lambda_{i, c}\mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$
接着进行特殊场景下的优化（head dim=1）：
$$
\begin{aligned}
\ [[ \mathbf{ V}_i \mathbf{d O}_i^\top ]\odot \mathbf M(\bar\Lambda_i)^\top] \mathbf { Q}_i
&= \mathbf{V}_i \odot \mathrm{cumsum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i, \text{reverse = True})   ,\\

\mathbf {dS}_{i}&= \Lambda_{i, c} \mathbf {dS}_{i+1}+  \mathrm{sum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i).

\end{aligned}
$$
最后结果如下：
$$
\begin{aligned}

\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\

{\mathbf {\bar V}_i}  &= \mathbf V_i  \odot \bar \Lambda_i , \\

\mathbf {d  K}_i &= \mathbf{ \bar V}_i \odot \mathbf {dS}_{i+1}
+ \mathbf{V}_i \odot \mathrm{cumsum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i, \text{reverse = True})  ,\\

\mathbf {dS}_{i}&= \Lambda_{i, c}\odot  \mathbf {dS}_{i+1}+  \mathrm{sum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i).
\end{aligned}
$$




#### $\mathbf{dV}$

首先使用vector decay下的结论：
$$
\begin{aligned}



\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\
\tilde{\mathbf K}_i &=  \mathbf K_i \odot  \bar  \Lambda_i,  \\


\mathbf {d  V}_i &= \mathbf{\tilde K}_i \mathbf {dS}_{i+1}
+ [[\mathbf{ K}_i \mathbf{ Q}_i^\top ] \odot \mathbf M(\Lambda_i)^\top ] \mathbf {d O}_i,\\


\mathbf {dS}_{i}&=

 \Lambda_{i, c}\mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$
接着进行特殊场景下的优化（head dim=1）：
$$
\begin{aligned}



\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\
\tilde{\mathbf K}_i &=  \mathbf K_i \odot  \bar  \Lambda_i,  \\


\mathbf {d  V}_i &= \mathbf{\tilde K}_i \odot \mathbf {dS}_{i+1}
+  \mathbf{K}_i \odot \mathrm{cumsum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i, \text{reverse = True}) ,\\


\mathbf {dS}_{i}&= \Lambda_{i, c}\odot  \mathbf {dS}_{i+1}+  \mathrm{sum}(\mathbf{d O}_i \odot \mathbf {\bar Q}_i).
\end{aligned}
$$
