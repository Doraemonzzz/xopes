# Lightning Attention

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= (\lambda_i\gamma_i^\top)\odot   \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
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





## Sequential Recurrence

### Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e}$，注意如果Decay为空，我们使用$\Lambda=1-\mathbf K, \Gamma=1-\mathbf V$（此时我们默认$0\le \mathbf K \le 1, 0\le \mathbf V \le 1$）。

我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= [\lambda_i\gamma_i^\top]\odot   \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top \\
&\triangleq  \mathbf a_i \odot \mathbf {s}_{i-1} + \mathbf k_i\mathbf v_i^\top, \\
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



### Backward

输入：$\mathbf {dO}\in \mathbb R^{n\times e}, \mathbf{ds}_{n+1}\in \mathbb R^{d\times e}$。

计算：
$$
\begin{aligned}
\mathbf{ds}_{n+1} &\in \mathbb R^{d\times e}, \\
 \lambda_{n+1} & = \mathbf 1_d, \\
  \gamma_{n+1} & = \mathbf 1_e, \\
\mathbf{ds}_{i}&=  [\lambda_{i+1}\gamma_{i+1}^\top] \odot \mathbf{ds}_{i+1} + \mathbf{do}_{i}, \\
\mathbf{dq}_i &=[\mathbf{s}_{i}] \mathbf{do}_i, \\
\mathbf{dk}_i &=[\mathbf{ds}_{i}] \mathbf{v}_i, \\
\mathbf{dv}_i &=\mathbf{ds}_{i}^\top \mathbf k_i , \\
\mathbf{d}\mathbf a_i &= \frac{\partial l}{\partial \mathbf a_i } \\
&=  \left[ \mathbf{ds}_i \odot\frac{\partial \mathbf{s}_i}{\partial \mathbf a_i }  \right] \\
&= \left[ \mathbf{ds}_i \odot {\mathbf{o}_{i-1}}  \right], \\
\mathbf{d\lambda}_i
&= \mathbf{d}\mathbf a_i  \gamma_i, \\
\mathbf{d\gamma}_i
&= \mathbf{d}\mathbf a_i ^\top \lambda_i.
\end{aligned}
$$



#### 解析计算

我们记：
$$
\begin{aligned}

\prod_{j=1}^t \lambda_j & = \alpha_t,  \\
\log \alpha_ t&= \sum_{j=1}^t \log \lambda_j,  \\
\prod_{j=1}^t \gamma_j & =\beta_t, \\
\log \beta_ t&= \sum_{j=1}^t \log \gamma_j,  \\
\mathbf a_t &= \lambda_t \gamma_t^\top,  \\
\mathbf A_t &= \odot_{j=1}^t \mathbf a_j \\
&= \alpha_t \beta_t^\top.
\end{aligned}
$$
其中：
$$
\odot_{j=1}^t \mathbf a_j =\mathbf a_1 \odot \mathbf a_2 \odot \ldots \odot \mathbf a_t.
$$
那么：
$$
\begin{aligned}
\mathbf s_t &= \mathbf a_t \odot  \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&=  \mathbf a_{t} \odot
\left( \mathbf a_{t-1} \odot  \mathbf s_{t-2} +\mathbf k_{t-1} \mathbf v_{t-1}^\top  \right)+\mathbf k_t \mathbf v_t^\top  \\
&= \mathbf a_{t} \odot  \mathbf a_{t-1} \odot \mathbf s_{t-2}  + \mathbf a_{t} \odot  \mathbf k_{t-1} \mathbf v_{t-1}^\top
+ \mathbf k_t \mathbf v_t^\top \\
&=  \ldots \\
&= \mathbf A_t\odot  \mathbf s_0  + \sum_{j=1}^t \left(\odot_{i=j+1}^t \mathbf a_i \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \mathbf A_t\odot \mathbf s_0  + \sum_{j=1}^t \frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top.
\end{aligned}
$$

注意到：
$$
\begin{aligned}
\frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top
&= (\alpha_t\beta_t^\top /\alpha_j\beta_j^\top) \odot  \mathbf k_j\mathbf v_j ^\top \\
&= (\mathbf k_j \odot \alpha_t /\alpha_j) (\mathbf v_j \odot \beta_t /\beta_j)^\top ,\\

\mathbf q_t^\top \frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top
&= \mathbf q_t^\top \left[(\mathbf k_j \odot \alpha_t /\alpha_j) (\mathbf v_j \odot \beta_t /\beta_j)^\top
\right] \\
&= \mathbf q_t^\top (\mathbf k_j \odot \alpha_t /\alpha_j)(\mathbf v_j \odot \beta_t /\beta_j)^\top \\
&= (\mathbf q_t \odot \alpha_t)^\top (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \mathrm{diag}\{ \beta_t \} \\
&= \left[ (\mathbf q_t \odot \alpha_t)^\top (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \odot \beta_t.
\end{aligned}
$$
那么：
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
&\triangleq \mathbf q_t^\top [ \alpha_t \beta_t^\top \odot \mathbf s_0] +  \mathbf {\bar o}_t.


\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf{dq}_t
&=[ \alpha_t \beta_t^\top \odot \mathbf s_0] \mathbf{do}_t
+   \alpha_t\odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t), \\
&= [ \alpha_t \beta_t^\top \odot \mathbf s_0] \mathbf{do}_t  + \mathbf {d\bar{q}}_t,\\
\mathbf{dk}_t
&=
(1/\alpha_t) \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right) ,\\

\mathbf{dv}_t
&=(1/\beta_t) \odot  \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right),\\

\mathbf {d\log \alpha_t}
&= \alpha_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]\beta_t \right]  +
\alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t) \\
&- (1/\alpha_t)\odot \mathbf k_t \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right),\\
&= \alpha_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]\beta_t \right]
+\mathbf q_t \odot \mathbf {d\bar{q}}_t - \mathbf k_t \odot \mathbf{dk}_t,\\

\mathbf {d\log \beta_t}
&= \beta_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]^\top \alpha_t \right]
+ \mathbf{do}_t \odot \left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \\

&- (1/\beta_t) \odot \mathbf v_t \odot   \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right)
\\
&= \beta_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]^\top \alpha_t \right]  +
  \mathbf {\bar{o}}_t \odot \mathbf{do}_t - \mathbf v_t \odot \mathbf{dv}_t .
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\log \alpha_ t&= \sum_{j=1}^t \log \lambda_j,  \\
\log \beta_ t&= \sum_{j=1}^t \log \gamma_j.
\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf d\log \lambda_t &= \sum_{j\ge t} \mathbf d\log \alpha_j, \\
\lambda_t &= \lambda_t \odot \log \lambda_t, \\
\mathbf d\log \gamma_t &= \sum_{j\ge t} \mathbf d\log \beta_j, \\
\gamma_t &= \gamma_t \odot \log \gamma_t.

\end{aligned}
$$




## Chunk Recurrence

### Forward

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
\mathbf S_i &= \mathbf s_{ic}, i=0, \ldots, n/c.
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
\Gamma_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
&= \alpha_{(i-1)c+j}/ \alpha_{(i-1)c} \\
\Delta_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\gamma_s, \\
&=  \beta_{(i-1)c+j}/ \beta_{(i-1)c}, \\
\frac{\mathbf A_{ic+s}}{\mathbf A_{ic}}
&= \Gamma_{i,s}\Delta_{i,s}^\top.

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
(\Gamma_{i,s}\Delta_{i,s}^\top)
\odot \mathbf S_i
\right] \\
&= [[\mathbf q_{ic+s}\odot \Gamma_{i,s}]^\top \mathbf S_i ] \odot \Delta_{i,s}\\
&=  \left[\left[\left[
\mathbf Q_i \odot \Gamma_i
\right] \mathbf S_i \right] \odot \Delta_i \right]_s.
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
\left[[(\Gamma_{i,s}\Delta_{i,s}^\top)] /  (\Gamma_{i,j-ic}\Delta_{i,j-ic}^\top) \right]\odot  \mathbf k_j\mathbf v_j ^\top \right) \\

&= \left[(\mathbf q_{ic+s} \odot \Gamma_{i,s})^\top
\left( \sum_{j=ic+1}^{ic+s}
 (\mathbf k_j / \Gamma_{i,j-ic}) (\mathbf v_j /\Delta_{i,j-ic}) ^\top \right) \right] \odot \Delta_{i,s} \\

&= \left[\left[\left[
\mathbf Q_i \odot \Gamma_i
\right]  \left[\mathbf K_i / \Gamma_i \right]^\top \odot \mathbf M\right] / \Delta_i \right]_s
\end{aligned}
$$
因此写成矩阵形式即为：
$$
\mathbf O_i=\left[\left[\left[
\mathbf Q_i \odot \Gamma_i
\right] \mathbf S_i \right] \odot \Delta_i \right] + \left[\left[\left[
\mathbf Q_i \odot \Gamma_i
\right]  \left[\mathbf K_i / \Gamma_i \right]^\top \odot \mathbf M\right] / \Delta_i \right].
$$
另一方面，对下式取$t=ic, s=c$
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
\mathbf s_{(i+1)c}
&= \left(\odot_{i=ic+1}^{(i+1)c} \mathbf a_i \right)\odot  \mathbf s_{ic}  + \sum_{j=ic+1}^{(i+1)c} \left(\odot_{t=j+1}^{(i+1)c} \mathbf a_t \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf s_{ic}  +\sum_{j=1}^{c}  \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic+j}}\odot  \mathbf k_{ic+j}\mathbf v_{ic+j}^\top, \\
\mathbf S_{i+1} &= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} +
\sum_{j=1}^{c}    (\mathbf k_{ic+j} \odot  \Gamma_{i,j})(\mathbf v_{ic+j}\odot \Delta_{i,j}) ^\top \\
&= \frac{\mathbf A_{(i+1)c}}{\mathbf A_{ic}}\odot \mathbf S_{i} + [\mathbf K_i \odot \Gamma_i]^\top [\mathbf V_i \odot \Delta_i].
\end{aligned}
$$


### Backward

$$
\begin{aligned}
\mathbf{dq}_i &=[\mathbf{s}_{i}] \mathbf{do}_i, \\
\mathbf{ds}_{i}&=  [\lambda_{i+1}\gamma_{i+1}^\top] \odot \mathbf{ds}_{i+1} + \mathbf{do}_{i}, \\
\mathbf{dk}_i &=[\mathbf{ds}_{i}] \mathbf{v}_i, \\
\mathbf{dv}_i &=\mathbf{ds}_{i}^\top \mathbf k_i , \\

\end{aligned}
$$

关于第一项，代入前向公式可得：
$$
\mathbf {dQ}_i=\left[\left[\left[
\mathbf {dO}_i \odot \Delta_i
\right] \mathbf S_i^\top \right] \odot \Gamma_i \right] + \left[\left[\left[
\mathbf {dO}_i \odot \Delta_i
\right]  \left[\mathbf V_i / \Delta_i \right]^\top \odot \mathbf M\right] / \Gamma_i \right].
$$
对于另外两项，我们记：
$$
\mathbf u_{i}=\mathbf{ds}_{n-i}, \\
\rho_{i}=\mathbf{\alpha}_{n-i}, \\
\tau_{i}=\mathbf{\beta}_{n-i}, \\
i=0,\ldots, n.
$$
那么：
$$
\mathbf{u}_{i}=  [\lambda_{i+1}\gamma_{i+1}^\top] \odot \mathbf{ds}_{i+1} + \mathbf{do}_{i}
$$
