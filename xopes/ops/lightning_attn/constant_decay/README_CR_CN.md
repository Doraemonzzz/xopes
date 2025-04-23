# Lightning Attention with Decay

## 回顾

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\lambda$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= \lambda  \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
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



## Forward

$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s =\lambda^j, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s= \lambda^{c-j+1} , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M (\Lambda_i) ]
{\mathbf V}_i,  \\
\mathbf S_{i+1}&= \lambda^c\mathbf S_{i}+ \tilde{\mathbf K}_{i+1} {\mathbf V}_{i+1}^\top.
\end{aligned}
$$



## Backward

### $\mathbf{dQ}$

$$
\begin{aligned}

{\mathbf {d \bar O}}_i &=  \mathbf O_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\


{\mathbf {d Q}}_i&= \mathbf{d \bar O}_i  \mathbf S_i^\top  + [ [\mathbf{d O}_i {\mathbf V}_i^\top] \odot \mathbf M(\Lambda_i) ]
{\mathbf K}_i, \\

\mathbf S_{i+1}&= \lambda^c \mathbf S_{i}+ \tilde{\mathbf K}_{i+1} {\mathbf V}_{i+1}^\top.
\end{aligned}
$$



### $\mathbf{dK} $

$$
\begin{aligned}





\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\

{\mathbf {\bar V}_i}  &= \mathbf V_i  \odot \bar \Lambda_i , \\

\mathbf {d  K}_i &= \mathbf{ \bar V}_i \mathbf {dS}_i^\top
+ [[ \mathbf{ V}_i \mathbf{d O}_i^\top ]\odot \mathbf M(\bar\Lambda_i)^\top] \mathbf { Q}_i,\\

\mathbf {dS}_{i}&=

 \lambda^ c \mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$



### $\mathbf{dV}$

所以最终形式为：
$$
\begin{aligned}



\bar {\mathbf Q}_i &=  \mathbf Q_i \odot   \Lambda_i,  \\
\tilde{\mathbf K}_i &=  \mathbf K_i \odot  \bar  \Lambda_i,  \\


\mathbf {d  V}_i &= \mathbf{\tilde K}_i \mathbf {dS}_i
+ [[\mathbf{ K}_i \mathbf{ Q}_i^\top ] \odot \mathbf M(\Lambda_i)^\top ] \mathbf {d O}_i,\\


\mathbf {dS}_{i}&=

 \lambda^c \mathbf {dS}_{i+1}+ {\mathbf {\bar Q}}_{i} {\mathbf {d O}}_{i}^\top.
\end{aligned}
$$
