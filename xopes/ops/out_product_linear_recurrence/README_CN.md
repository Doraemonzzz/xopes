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
\mathbf{dkv}_{i}&= \mathrm{diag}(\lambda_i)  \mathbf{dkv}_{i+1} + \mathbf{do}_{i}, \\
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
\mathbf o_t &= \mathrm{diag}(\lambda_t) \mathbf o_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&= \mathrm{diag}(\lambda_t )
\left( \mathrm{diag}(\lambda_t ) \mathbf o_{t-2} +\mathbf k_{t-1} \mathbf v_{t-1}^\top  \right)+\mathbf k_t \mathbf v_t^\top  \\
&= \mathrm{diag}(\lambda_t \lambda_{t-1})\mathbf o_{t-2}  + \mathrm{diag}(\lambda_t ) \mathbf k_{t-1} \mathbf v_{t-1}^\top
+ \mathbf k_t \mathbf v_t^\top \\
&=  \ldots \\
&= \sum_{j=1}^t \mathrm{diag} \left(\prod_{i=j+1}^n \lambda_i \right)  \mathbf k_j^\top \mathbf v_j \\
&= \sum_{j=1}^t \mathrm{diag} \left(\exp(\beta_n-\beta_j) \right)  \mathbf k_j^\top \mathbf v_j \\
&=\mathrm{diag}\left( \exp(\beta_n) \right)\sum_{j=1}^t \mathrm{diag} \left(\exp(-\beta_j) \right)  \mathbf k_j^\top \mathbf v_j.
\end{aligned}
$$
所以：
$$
\begin{aligned}
 \frac{\partial [\mathbf{do}_t]_{ij}} {\partial [\mathbf{k}_t]_s}
 &=[\exp(\beta_n)]_i [\exp(-\beta_t) ]_i[\mathbf v_t]_j \mathbf{1}_{i=s} \\
[\mathbf{dk}_u]_s
&= \sum_{t=u}^n \sum_{i,j} \frac{\partial [\mathbf{do}_t]_{ij}} {\partial [\mathbf{k}_t]_s}  [\mathbf{do_t}]_{ij} \\
&= \sum_{t=u}^n \sum_{i,j} [\exp(\beta_n)]_i [\exp(-\beta_t) ]_i [\mathbf v_t]_j \mathbf{1}_{i=s} [\mathbf{do_t}]_{ij}  \\
&=  [\exp(\beta_n)]_s\sum_{t=u}^n  [\exp(-\beta_t) ]_s\sum_{j}  [\mathbf v_t]_j  [\mathbf{do_t}]_{sj} \\
&=  [\exp(\beta_n)]_s\sum_{t=u}^n  [\exp(-\beta_t) ]_s
[\mathbf{do}_t\mathbf v_t]_s, \\
\mathbf{dk}_u& = \mathrm{diag}\left( \exp(\beta_n) \right)\sum_{t=u}^n \mathrm{diag}\left(\exp(-\beta_t) \right)
[\mathbf{do}_t\mathbf v_t].
\end{aligned}
$$
另一方面：
$$
\begin{aligned}
 \frac{\partial [\mathbf{do}_t]_{ij}} {\partial [\mathbf{\beta}_t]_s}
 &= -[\exp(\beta_n)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=s}, t\neq n,  \\
[\mathbf{d\beta}_u]_s
&= \sum_{t=u}^n \sum_{i,j} \frac{\partial [\mathbf{do}_t]_{ij}} {\partial [\mathbf{\beta}_t]_s}  [\mathbf{do_t}]_{ij} \\
&= \sum_{t=u}^n \sum_{i,j} -[\exp(\beta_n)]_i [\exp(-\beta_t) ]_i  [\mathbf k_t \mathbf v_t^\top]_{ij}\mathbf{1}_{i=s} [\mathbf{do_t}]_{ij}  \\
&=  -[\exp(\beta_n)]_s\sum_{t=u}^n [\exp(-\beta_t) ]_s\sum_{j}  [\mathbf k_t \mathbf v_t^\top]_{sj} [\mathbf{do_t}]_{sj} \\
&=   -[\exp(\beta_n)]_s\sum_{t=u}^n  [\exp(-\beta_t) ]_s \left([\mathbf k_t \mathbf v_t^\top]_s \odot [\mathbf{do_t}]_s  \right) \mathbf 1_{e}  \\
\mathbf{d\beta}_u& = -\mathrm{diag}\left( \exp(\beta_n) \right)\sum_{t=u}^n \mathrm{diag}\left(\exp(-\beta_t) \right)
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}.
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\left[\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e} \right]_{s}

&= \sum_{k} [\mathbf{k}_t]_s[\mathbf v_{t}]_k[\mathbf{do_t}]_{s, k} \\
&= [\mathbf k_{t}]_s \sum_{k} [\mathbf v_{t}]_k[\mathbf{do_t}]_{s, k} \\
&= [\mathbf k_{t}]_s [\mathbf{do}_t\mathbf v_t]_s,\\
\left([\mathbf k_t \mathbf v_t^\top] \odot [\mathbf{do_t}]  \right) \mathbf 1_{e}
&= [\mathbf k_{t}] \odot [\mathbf{do}_t\mathbf v_t].
\end{aligned}
$$
因此：
$$
\mathbf{d\beta}_u = - \mathbf k_u \odot  \mathbf{dk}_u.
$$
因为：
$$
\beta_u = \sum_{j=1}^t\alpha_u.
$$
所以：
$$
\begin{aligned}
\mathbf{d}\alpha_u
& = \sum_{j=u}^n \beta_j,  \\
\mathbf{d}\lambda_u
&= \mathbf{d}\exp(\alpha_u)\\
&= \exp(\alpha_u)\odot\mathbf{d}\alpha_u\\
&= \lambda_u \odot \mathbf{d}\alpha_u.
\end{aligned}
$$


### 补充

如果$\Lambda=1-\mathbf K$。

那么：
$$
\mathbf {dK} = \mathbf {dK}- \mathbf {d\Lambda}.
$$
