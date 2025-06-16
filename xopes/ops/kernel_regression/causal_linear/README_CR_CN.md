## Forward

### tril

注意到上式等价于：
$$
[\mathbf I + \mathrm{tril}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ] \mathbf O = \mathbf V.
$$

记：
$$
\mathbf R\triangleq [\mathbf I + \mathrm{tril}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ] .
$$
首先通过雅可比迭代计算得到$\mathbf R_{ii}^{-1}$，然后计算得到：
$$
\mathbf U_i = \mathbf R_{ii}^{-1} \mathbf V_i.
$$
记：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\


\mathbf {\bar K}_i & = \mathbf K_i / \mathbf \Alpha_i.
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\sum_{j=1}^i \mathbf R_{ij} \mathbf O_j &= \mathbf V_i, \\

\mathbf R_{ii}\mathbf O_i + \sum_{j=1}^{i-1} \mathbf R_{ij} \mathbf O_j &= \mathbf V_i, \\

\mathbf O_i & = \mathbf R_{ii}^{-1}\mathbf V_i - \mathbf R_{ii}^{-1}\left( \sum_{j=1}^{i-1} \mathbf R_{ij} \mathbf O_j  \right), \\

&=\mathbf R_{ii}^{-1} \left( \mathbf V_i -  \left( \sum_{j=1}^{i-1} [[\mathbf Q_i \mathbf K_j^\top] \odot \mathbf M_{ij}]  \mathbf O_j  \right)  \right)\\

&\triangleq  \mathbf R_{ii}^{-1} (\mathbf U_i - \mathbf P_{i}).

\end{aligned}
$$
根据linear atttention的结论，我们有：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf P}_i&=  \bar{\mathbf Q}_i  \mathbf S_{i-1} ,  \\
\mathbf S_{i}&= \Lambda_{i, c} \mathbf S_{i-1}+ \tilde{\mathbf K}_{i} {\mathbf O}_{i}^\top.
\end{aligned}
$$
我们记改function为：
$$
f(\mathbf Q, \mathbf K, \mathbf V, \Lambda)= [\mathbf I + \mathrm{tril}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ]^{-1} \mathbf V.
$$


### triu

triu情况对应于：
$$
[\mathbf I + \mathrm{triu}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ] \mathbf O = \mathbf V.
$$
记：
$$
\mathbf R\triangleq [\mathbf I + \mathrm{triu}([(\mathbf Q\mathbf K^\top)\odot \mathbf M], -1) ] .
$$
首先通过雅可比迭代计算得到$\mathbf R_{ii}^{-1}$，然后计算得到：
$$
\mathbf U_i = \mathbf R_{ii}^{-1} \mathbf V_i.
$$
记：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\


\mathbf {\bar K}_i & = \mathbf K_i / \mathbf \Alpha_i, \\


\end{aligned}
$$

注意到：
$$
\begin{aligned}
\sum_{j=i}^{n} \mathbf R_{ij} \mathbf O_j &= \mathbf V_i, \\

\mathbf R_{ii}\mathbf O_i + \sum_{j=i+1}^{n} \mathbf R_{ij} \mathbf O_j &= \mathbf V_i, \\

\mathbf O_i & = \mathbf R_{ii}^{-1}\mathbf V_i - \mathbf R_{ii}^{-1}\left( \sum_{j=i+1}^{n} \mathbf R_{ij} \mathbf O_j  \right), \\

&=\mathbf R_{ii}^{-1} \left( \mathbf V_i -  \left( \sum_{j=i+1}^{n} [[\mathbf Q_i \mathbf K_j^\top] \odot \mathbf M_{ij}]  \mathbf O_j  \right)  \right)\\

&\triangleq  \mathbf R_{ii}^{-1} (\mathbf V_i - \mathbf P_{i}).

\end{aligned}
$$
根据linear atttention的结论，我们有：
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\

\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf P}_i&=  \bar{\mathbf Q}_i  \mathbf S_{i+1} ,  \\

\mathbf S_{i}&= \Lambda_{i+1, c} \mathbf S_{i+1}+ \tilde{\mathbf K}_{i} {\mathbf O}_{i}^\top.
\end{aligned}
$$




## Backward

### $\mathbf {ds}$的递推

$$
\begin{aligned}
\mathbf{ds}_{t}
&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {dv}_{t+1}^\top \\

&\triangleq \lambda_{t+1} \mathbf{ds}_{t+1} + \mathbf {q}_{t+1} \mathbf {d\bar v}_{t+1}^\top,    \\

\mathbf{ds}_{t}&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1}(\mathbf {do}_{t+1} + \mathbf p_{t+1})^\top \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1}(\mathbf {do}_{t+1} + \mathbf {ds}_{t+1}^\top \mathbf k_{t+1})^\top \\

&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {do}_{t+1}^\top \ - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf{k}_{t+1}^\top \mathbf {ds}_{t+1}\\

&= \lambda_{t+1}(\mathbf I - \mathbf {q}_{t+1} \mathbf{k}_{t+1}^\top ) \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {do}_{t+1}^\top \\

&\triangleq  \lambda_{t+1}(\mathbf I - \mathbf {q}_{t+1} \mathbf{k}_{t+1}^\top ) \mathbf{ds}_{t+1} + \mathbf {q}_{t+1} \mathbf {d\bar o}_{t+1}^\top , \\

\mathbf {d\bar o}_{t}^\top & = - \lambda_{t}\mathbf {d o}_{t}^\top.
\end{aligned}
$$

注意到此时：
$$
\begin{aligned}
\mathbf{ds}_{t}
&= \lambda_{t+1} \mathbf{ds}_{t+1} - \lambda_{t+1}\mathbf {q}_{t+1} \mathbf {dv}_{t+1}^\top \\


&= \lambda_{t+1}(\lambda_{t+2} \mathbf{ds}_{t+2} -\lambda_{t+2} \mathbf {q}_{t+2} \mathbf {dv}_{t+2}^\top ) - \lambda_{t+1} \mathbf {q}_{t+1} \mathbf {dv}_{t+1}^\top  \\

&=  \lambda_{t+1} \lambda_{t+2}\mathbf{ds}_{t+2} - \lambda_{t+1}\lambda_{t+2}  \mathbf {q}_{t+2} \mathbf {d v}_{t+2}^\top -  \lambda_{t+1} \mathbf {q}_{t+1} \mathbf {d v}_{t+1}^\top \\

&= \ldots \\

&=  \alpha_{n} / \alpha_t \mathbf{ds}_{n} - \sum_{j=t+1}^n \alpha_{j} / \alpha_t \mathbf q_j \mathbf {d v}_{j}^\top.

\end{aligned}
$$
在实际中，我们不会使用$\mathbf {ds}_n$，所以后续的讨论中忽略这点。



### $\mathbf {dQ}$

注意到：
$$
\mathbf {dq}_t^\top= -\lambda_t  \left[\mathbf {s}_{t-1}\mathbf {dv}_t\right]^\top
=-\lambda_t \mathbf {dv}_t^\top \mathbf {s}_{t-1}^\top
.
$$
那么可以恢复成并行版本：
$$
\begin{aligned}
\mathbf {dQ} +  \mathrm{tril}([\mathbf {dV} \mathbf O^\top ]\odot \mathbf M)   \mathbf K &= \mathbf 0, \\
(\mathbf I + \mathrm{tril}([\mathbf {dV} \mathbf O^\top ]\odot \mathbf M)  ) \mathbf K &= \mathbf {dQ}  + \mathbf {K}.
\end{aligned}
$$
那么：
$$
\mathbf {dQ}= f(\mathbf {dV}, \mathbf O, \mathbf K, \Lambda)-\mathbf {K}。
$$


### $\mathbf {dK}$

注意到：
$$
\mathbf {dk}_t^\top=  \mathbf o_t^\top \mathbf {ds}_t  ^\top.
$$
那么可以恢复成并行版本：
$$
\mathbf {dK} = -\mathrm{triu}([\mathbf O \mathbf Q^\top]\odot \mathbf M, -1) \mathbf {dV}.
$$


### $\mathbf {dV}$

注意到：
$$
\begin{aligned}
\mathbf p_t^\top  &=\mathbf k_t ^\top \mathbf {ds}_{t} ,\\
\mathbf {dv}_t^\top - \mathbf {do}_t^\top &= \mathbf k_t ^\top \mathbf {ds}_{t}.


\end{aligned}
$$
那么可以恢复成并行版本：
$$
\begin{aligned}
\mathbf {dV} - \mathbf {dO} & = -\mathrm{triu}([\mathbf K \mathbf Q^\top]\odot \mathbf M, -1) \mathbf {dV}, \\
(\mathbf I + \mathrm{triu}([\mathbf K \mathbf Q^\top]\odot \mathbf M, -1)) \mathbf {dV} & = \mathbf {dO}.

\end{aligned}
$$
