# Flash Attention w/wo scalar/constant decay

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，Decay $\log\Lambda\in \mathbb R^{n}$，Scale $\alpha\in \mathbb R^n$，我们计算如下结果：
$$
\begin{aligned}
\mathbf M_{ij} &=
\begin{cases}
\sum_{t=j+1}^i \log\Lambda_t, & i \ge j, \\
-\infty, & i < j.
\end{cases} \\
\mathbf Q &= \mathbf Q / \alpha, \\
\mathbf S &= \mathbf Q\mathbf K^\top + \mathbf M, \\
\mathbf P &= \mathrm{Softmax}\left(\mathbf S \right) ,\\
\mathbf O &=  \mathbf P\mathbf V.
\end{aligned}
$$



## Forward

假设chunk size为$c$，初始化$\mathbf m_i=-\infty \times  \mathbf 1_c \in \mathbb R^{c}, \mathbf O_i = \mathbf 0\in \mathbb R^{c\times e}$，我们迭代计算：
$$
\begin{aligned}
\mathbf S_{ij} &=\mathbf Q_i^\top \mathbf K_j + \mathbf M_{ij} \in \mathbb R^{c\times c}, \\
\mathbf {m}_{ij} &= \mathrm{lse}( \mathbf S_{ij})\in \mathbb R^{c}, \\
\mathbf { O}_i^j &= \exp(\mathbf S_{ij} - \mathbf m_{ij})  \mathbf V_j \in \mathbb R^{c\times e}, \\

\mathbf {\bar m}_i &= \mathrm{lse}([\mathbf m_i, \mathbf m_i^j]) \in \mathbb R^{c}, \\

\mathbf p_i &= \exp(\mathbf m_i - \mathbf {\bar m}_i) \in \mathbb R^c, \\

\mathbf O_i &= \mathbf p_i \odot \mathbf {O}_i + (1-\mathbf p_i) \odot \mathbf O_i^j \in \mathbb R^{c\times e}, \\

\mathbf m_i &= \mathbf {\bar m}_i, \\

j&=1,\ldots i.
\end{aligned}
$$

注意我们会存储$\mathbf m_i$用以反向传播。



## Backward

$$
\begin{aligned}
\mathbf {dV} &= \mathrm{Softmax}([\mathbf Q\mathbf K^\top] \odot \mathbf M )^\top \mathbf {dO}.
\end{aligned}
$$

所以迭代计算即可：
$$
\begin{aligned}
\mathbf S_{ij} &=\mathbf Q_i^\top \mathbf K_j + \mathbf M_{ij} , \\
\mathbf P_{ij} &=  \exp(\mathbf S_{ij} - \mathbf m_i),  \\
\mathbf {dV}_{ji} &= \mathbf P_{ij}^\top   \mathbf {dO}_i , \\


\mathbf {dV}_j &=  \mathbf {dV}_j +  \mathbf {dV}_{ji}, \\
i&= j, \ldots, n.
\end{aligned}
$$
另一方面：
$$
\mathbf {dP}= \mathbf {dO} \mathbf V^\top.
$$
注意到：
$$
\mathbf P= \mathrm{Softmax}\left(\mathbf S \right).
$$
以及：
$$
\begin{aligned}
\frac{\partial p_{ij}}{ \partial s_{tk}}
&= \mathbf 1_{i=t}
\left [\mathbf 1_{j=k}  \frac{\exp(s_{ij})}{\sum_j \exp(s_{ij})}
-  \frac{\exp(s_{ij})\exp(s_{ik})}{\left( \sum_j \exp(s_{ij})\right)^2}
\right] \\

&= \mathbf 1_{i=t}
\left [\mathbf 1_{j=k} p_{ij}
-  p_{ij}p_{ik}
\right] \\

\frac{\partial \mathbf p_{i}}{ \partial \mathbf s_{t}}
&= \mathbf 1_{i=t} \left[  \mathrm{diag}(\mathbf p_i) -\mathbf p_{i} \mathbf p_i^\top   \right].
\end{aligned}
$$
所以：
$$
\begin{aligned}
\mathbf {ds}_t^\top  &= \mathbf  {dp}_t^\top \frac{\partial \mathbf p_{t}}{ \partial \mathbf s_{t}} \\
&=  \mathbf  {dp}_t^\top  \left[  \mathrm{diag}(\mathbf p_t) - \mathbf p_{t} \mathbf p_t^\top    \right] \\
&= (\mathbf p_t \odot \mathbf {dp}_t)^\top - ( \mathbf  {dp}_t^\top  \mathbf p_{t} )\mathbf p_t^\top .


\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf  {dp}_s^\top  \mathbf p_{s}
&= \sum_{t=1}^s \mathbf  {dp}_{st} \mathbf p_{st} \\
&= \sum_{t=1}^s {\mathbf {do}}_s^\top \mathbf v_t \exp(\mathbf q_s^\top \mathbf k_t- \mathbf m_s) \\
&= {\mathbf {do}}_s^\top \sum_{t=1}^s \mathbf v_t \exp(\mathbf q_s^\top \mathbf k_t- \mathbf m_s) \\
&= {\mathbf {do}}_s^\top \mathbf o_s \\
&\triangleq \mathbf d_s.
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf {ds}_t^\top
&= (\mathbf p_t \odot \mathbf {dp}_t)^\top - \mathbf d_t\mathbf p_t^\top .


\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf {d Q} &= \mathbf {dS} \mathbf K , \\
\mathbf {dK} &= \mathbf {dS}^\top  \mathbf Q.
\end{aligned}
$$
所以我们可以得出如下算法：
$$
\begin{aligned}
\mathbf D_i &= \mathrm{sum}(\mathbf {dO}_i \odot \mathbf O_i) ,\\
\mathbf {dP}_{ij} &= \mathbf {dO}_i \mathbf V_j^\top , \\
\mathbf S_{ij} &=\mathbf Q_i^\top \mathbf K_j + \mathbf M_{ij} , \\
\mathbf P_{ij} &=  \exp(\mathbf S_{ij} - \mathbf m_i),\\
\mathbf {dS}_{ij} &=\mathbf P_{ij} \odot \mathbf {dP}_{ij}- \mathrm{diag}(\mathbf D_i)  \mathbf {dP}_{ij}  , \\

\mathbf Q_i &=  \mathbf {Q}_i + \mathbf {dQ}_{ij} \mathbf K_j, \\
\mathbf K_j &=  \mathbf {K}_j + \mathbf {dQ}_{ij}^\top \mathbf Q_i, \\

j&=1,\ldots i.
\end{aligned}
$$
最后求decay部分的梯度，注意到：
$$
\begin{aligned}
\mathbf M_{ij} &=
\begin{cases}
\sum_{t=j+1}^i \log\Lambda_t\triangleq \alpha_i - \alpha_j, & i \ge j, \\
-\infty, & i < j.
\end{cases}  \\
\mathbf {dM}_{ij} &= \mathbf {d S}_{ij}.
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf {d \alpha}_t  &=  \sum_{j=1}^t \mathbf {\mathbf {dM}}_{tj} -
 \sum_{i=t}^n \mathbf {\mathbf {dM}}_{it} \\
 &= \mathrm{sum}(\mathbf p_t \odot \mathbf {dp}_t - \mathbf d_t\mathbf p_t, \mathrm{dim}=-1 )
 - \sum_i \left(\mathbf p_{it} \odot \mathbf {dp}_{it}  - \mathbf d_t\mathbf p_{it}\right).
\end{aligned}
$$
写成chunk形式即为：
$$
\mathbf {d\Alpha}_i = \sum_{j\le i} \mathbf {dS}_{ij} -  \sum_{j\ge i}  \mathbf {dS}_{ji}.
$$
