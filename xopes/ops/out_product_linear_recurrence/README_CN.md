# Todo: Chunk parallel

# Out product linear recurrence

给定输入$\mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf o & = \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf o_i &= \mathrm{diag}(\lambda_i) \mathbf o_{i-1} + \mathbf k_i \mathbf v_i^\top.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times d\times e}.
$$



## Forward

输入：$\mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，以及Decay $\Lambda\in \mathbb R^{n\times d}$，注意如果Decay为空，我们使用$\Lambda=1-\mathbf K$（我们默认$0\le \mathbf K \le 1$）。

计算：
$$
\begin{aligned}
\mathbf o & = \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf o_i &= \mathrm{diag}(\lambda_i) \mathbf o_{i-1} + \mathbf k_i \mathbf v_i^\top.
\end{aligned}
$$



## Backward

输入：$\mathbf {dO}\in \mathbb R^{n\times d\times e}$。

计算：
$$
\begin{aligned}
\mathbf{dkv}_{n+1} &= \mathbf 0\in \mathbb R^{d\times e}, \\
\mathbf{dkv}_{i}&= \mathrm{diag}(\lambda_{i+1})  \mathbf{dkv}_{i+1} + \mathbf{do}_{i}, \\
\mathbf{dk}_i &=\mathbf{dkv}_{i} \mathbf{v}_i, \\
\mathbf{dv}_i &=\mathbf k_i\mathbf{dkv}_{i}^\top. \\


\end{aligned}
$$
关于$\mathbf{d\lambda}_i$，我们首先展开前向式，并且记：
$$
\begin{aligned}
\lambda_i &=\exp(\alpha_i),  \\
\prod_{j=1}^t \lambda_j & = \exp\left(\sum_{j=1}^t\alpha_j  \right) \\
&\triangleq \exp(\beta_t).
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf o_s &= \mathrm{diag}(\lambda_s) \mathbf o_{s-1} + \mathbf k_s \mathbf v_s^\top \\
&= \mathrm{diag}(\lambda_s )
\left( \mathrm{diag}(\lambda_{s-1} ) \mathbf o_{s-2} +\mathbf k_{s-1} \mathbf v_{s-1}^\top  \right)+\mathbf k_s \mathbf v_s^\top  \\
&= \mathrm{diag}(\lambda_s \lambda_{s-1})\mathbf o_{s-2}  + \mathrm{diag}(\lambda_s ) \mathbf k_{s-1} \mathbf v_{s-1}^\top
+ \mathbf k_s \mathbf v_s^\top \\
&=  \ldots \\
&= \sum_{j=1}^s \mathrm{diag} \left(\prod_{i=j+1}^s \lambda_i \right)  \mathbf k_j \mathbf v_j^\top \\
&= \sum_{j=1}^s \mathrm{diag} \left(\exp(\beta_s-\beta_j) \right)  \mathbf k_j\mathbf v_j ^\top\\
&=\mathrm{diag}\left( \exp(\beta_s) \right)\sum_{j=1}^s \mathrm{diag} \left(\exp(-\beta_j) \right)  \mathbf k_j \mathbf v_j^\top.
\end{aligned}
$$
所以：
$$
\begin{aligned}
 \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{k}_t]_r}
 &=[\exp(\beta_s)]_i [\exp(-\beta_t) ]_i[\mathbf v_t]_j \mathbf{1}_{i=r} \\
[\mathbf{dk}_t]_r
&= \sum_{s=t}^n \sum_{i,j} \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{k}_t]_r}  [\mathbf{do_s}]_{ij} \\
&= \sum_{s=t}^n \sum_{i,j} [\exp(\beta_s)]_i [\exp(-\beta_t) ]_i [\mathbf v_t]_j \mathbf{1}_{i=r} [\mathbf{do_s}]_{ij}  \\
&=  \sum_{s=t}^n  [\exp(\beta_s)]_r[\exp(-\beta_t) ]_r\sum_{j}  [\mathbf v_t]_j  [\mathbf{do_s}]_{rj} \\
&=  \sum_{s=t}^n  [\exp(\beta_s)]_r[\exp(-\beta_t) ]_r
[\mathbf{do}_s\mathbf v_t]_r, \\

\mathbf{dk}_t& = \sum_{s=t}^n \mathrm{diag}\left( \exp(\beta_s) \right)\mathrm{diag}\left(\exp(-\beta_t) \right)
[\mathbf{do}_s\mathbf v_t].
\end{aligned}
$$
另一方面：
$$
\begin{aligned}
 \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{\beta}_t]_r}
 &= -[\exp(\beta_s)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=r}, t\neq s,  \\

[\mathbf{d\beta}_t]_r
&= \sum_{s=t+1}^n \sum_{i,j} \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{\beta}_t]_r}  [\mathbf{do_s}]_{ij} \\

&= \sum_{s=t+1}^n \sum_{i,j} -[\exp(\beta_s)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=r} [\mathbf{do_s}]_{ij}  \\
&=  - \sum_{s=t+1}^n [\exp(\beta_s)]_r[\exp(-\beta_t) ]_r\sum_{j}  [\mathbf k_t \mathbf v_t^\top]_{rj} [\mathbf{do_s}]_{rj} \\

&=   -\sum_{s=t+1}^n  [\exp(\beta_s)]_r [\exp(-\beta_t) ]_r \left([\mathbf k_t \mathbf v_t^\top]_r \odot [\mathbf{do_s}]_r  \right) \mathbf 1_{e}  \\
\mathbf{d\beta}_t& = -\sum_{s=t+1}^n \mathrm{diag}\left( \exp(\beta_s) \right) \mathrm{diag}\left(\exp(-\beta_t) \right)
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e}\\
&= -\sum_{s=t}^n \mathrm{diag}\left( \exp(\beta_s) \right)\mathrm{diag}\left(\exp(-\beta_t) \right)
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e}
+ \left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}.
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\left[\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e} \right]_{r}

&= \sum_{k} [\mathbf{k}_t]_r[\mathbf v_{t}]_k[\mathbf{do_s}]_{r, k} \\
&= [\mathbf k_{t}]_r \sum_{k} [\mathbf v_{t}]_k[\mathbf{do_s}]_{r, k} \\
&= [\mathbf k_{t}]_r [\mathbf{do}_s\mathbf v_t]_r,\\
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e}
&= [\mathbf k_{t}] \odot [\mathbf{do}_s\mathbf v_t].
\end{aligned}
$$
因此：
$$
\mathbf{d\beta}_t = - \mathbf k_t \odot  \mathbf{dk}_t + [\mathbf k_{t}] \odot [\mathbf{do}_t\mathbf v_t].
$$
因为：
$$
\beta_t = \sum_{j=1}^t\alpha_j.
$$
所以：
$$
\begin{aligned}
\mathbf{d}\alpha_t
& = \sum_{j=t}^n \mathbf{d}\beta_j,  \\
\mathbf{d}\lambda_t
&= \mathbf{d}\exp(\alpha_t)\\
&= \exp(\alpha_t)\odot\mathbf{d}\alpha_t\\
&= \lambda_t \odot \mathbf{d}\alpha_t.
\end{aligned}
$$



### 补充

如果$\Lambda=1-\mathbf K$。

那么：
$$
\begin{aligned}
\mathbf {dK}
&= \mathbf {dK}- \mathbf {d\Lambda} \\
&= \mathbf {dK}- (\Lambda \odot \mathbf{d\Alpha}) \\
&= \mathbf {dK}- ((1-\mathbf K) \odot \mathbf{d\Alpha}).
\end{aligned}
$$



### 补充1

$$
\begin{aligned}
 \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{\beta}_t]_r}
 &= -[\exp(\beta_s)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=r}, s> t,  \\

 \frac{\partial [\mathbf{do}_t]_{ij}} {\partial [\mathbf{\beta}_t]_r}
 &=[\exp(\beta_t)]_i \sum_{s=1}^t [\exp(-\beta_s) ]_i  [\mathbf k_s \mathbf v_s^\top]_{ij} \mathbf{1}_{i=r},  \\
&= [\mathbf o_t]_{ij}  \mathbf{1}_{i=r}, \\

[\mathbf{d\beta}_t]_r
&= \sum_{i,j}  [\mathbf{do_t}]_{ij}[\mathbf o_t]_{ij}  \mathbf{1}_{i=r}  + \sum_{s=t+1}^n \sum_{i,j} \frac{\partial [\mathbf{do}_s]_{ij}} {\partial [\mathbf{\beta}_t]_r}  [\mathbf{do_s}]_{ij} \\

&=\sum_{i,j}  [\mathbf{do_t}]_{ij}[\mathbf o_t]_{ij}  \mathbf{1}_{i=r}  + \sum_{s=t+1}^n \sum_{i,j} -[\exp(\beta_s)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=r} [\mathbf{do_s}]_{ij}  \\
&=  \sum_{j}  [\mathbf{do_t}]_{rj}[\mathbf o_t]_{rj}    - \sum_{s=t+1}^n [\exp(\beta_s)]_r[\exp(-\beta_t) ]_r\sum_{j}  [\mathbf k_t \mathbf v_t^\top]_{rj} [\mathbf{do_s}]_{rj} \\

&=  [\mathbf o_t]_{r}^\top [\mathbf{do_t}]_{r}   -\sum_{s=t+1}^n  [\exp(\beta_s)]_r [\exp(-\beta_t) ]_r \left([\mathbf k_t \mathbf v_t^\top]_r \odot [\mathbf{do_s}]_r  \right) \mathbf 1_{e}  \\
\mathbf{d\beta}_t
& =\left([ \mathbf o_t] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}  -\sum_{s=t+1}^n \mathrm{diag}\left( \exp(\beta_s) \right) \mathrm{diag}\left(\exp(-\beta_t) \right)
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e}\\

&= -\sum_{s=t}^n \mathrm{diag}\left( \exp(\beta_s) \right)\mathrm{diag}\left(\exp(-\beta_t) \right)
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_s}]  \right) \mathbf 1_{e}
+\left([ \mathbf o_t] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}+ \left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}.
\end{aligned}
$$

所以：
$$
\mathbf{d\beta}_t = - \mathbf k_t \odot  \mathbf{dk}_t + [\mathbf k_{t}] \odot [\mathbf{do}_t\mathbf v_t]
+\left([ \mathbf o_t] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e} .
$$


注意到：
$$
\begin{aligned}
\mathbf o_t
&=\mathrm{diag}\left( \exp(\beta_t) \right)\sum_{j=1}^t \mathrm{diag} \left(\exp(-\beta_j) \right)  \mathbf k_j\mathbf v_j^\top .
\end{aligned}
$$
所以：
$$
\begin{aligned}
\left([ \mathbf o_t] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}
&= \left(
\left[\mathrm{diag}\left( \exp(\beta_t) \right)\sum_{j=1}^t \mathrm{diag} \left(\exp(-\beta_j) \right)  \mathbf k_j \mathbf v_j^\top \right] \odot [\mathbf {do}_t] \right) \mathbf 1_e \\
&= \sum_{j=1}^t  \left(\left[ [ \mathbf k_j \odot \exp(\beta_t-\beta_j)]\mathbf v_j^\top \right] \odot [\mathbf {do}_t]\right) \mathbf{1_e}\\
&= \sum_{j=1}^t   [ \mathbf k_j \odot \exp(\beta_t-\beta_j)] \odot [\mathbf {do}_t \mathbf v_j].

\end{aligned}
$$
