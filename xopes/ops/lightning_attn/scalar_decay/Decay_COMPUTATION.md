注意到
$$
\begin{aligned}
\Lambda_{i,j} &= \prod_{s=(i-1)c+1}^{(i-1)c+j}\lambda_s, \\
\bar \Lambda_{i,j} &= \prod_{s=(i-1)c+j}^{ic}\lambda_s , \\

\bar{\mathbf Q}_i &=  \mathbf Q_i \odot  \Lambda_i,  \\

\tilde{\mathbf K}_i &=  \mathbf K_i \odot \bar \Lambda_i,  \\

{\mathbf O}_i&=  \bar{\mathbf Q}_i  \mathbf S_i  + [ [{\mathbf Q}_i {\mathbf K}_i^\top] \odot \mathbf  M (\Lambda_i) ]
{\mathbf V}_i\\
&= \mathbf O_{\mathrm{inter}} + \mathbf O_{\mathrm{intra}}.
\end{aligned}
$$
注意intra和inter部分都对于$\log \lambda$有梯度贡献，所以：
$$
\begin{aligned}
\mathbf d \log\Lambda_i
&= \mathbf d\bar {\mathbf  Q}_i \odot \frac{\partial \bar{\mathbf Q}_i}{\partial \log \Lambda_i}
+
 f\left(\mathbf d\mathbf M(\exp(\log \Lambda_i)), \frac{\partial \mathbf M(\exp(\log \Lambda_i))}
 {\partial \log \Lambda_i}\right) \\

 &=\left[ [\mathbf {dO}_{i} \mathbf S_i^\top] \odot   \bar{\mathbf Q}_i\right] \mathbf 1_d +

 f\left([\mathbf {dO}_{i} \mathbf V_i^\top] \odot [\mathbf Q_i \mathbf K_i^\top]\odot \mathbf M
, \frac{\partial \mathbf M(\exp(\log \Lambda_i))}
 {\partial \log \Lambda_i} \right).
\end{aligned}
$$
下面讨论第二项的维度如何规约：：
$$
\begin{aligned}
\mathbf M(\mathbf \Lambda_i, \mathbf \Lambda_i)
&= \left[\exp( \log\mathbf \Lambda_i[:,\mathrm{None}] - \log\mathbf \Lambda_i[\mathrm{None},:] )\right] \odot \mathbf M ,  \\

f\left([\mathbf {dO}_{\mathrm{intra}} \mathbf V_i^\top] \odot [\mathbf Q_i \mathbf K_i^\top]\odot \mathbf M
, \frac{\partial \mathbf M(\exp(\log \Lambda_i))}
 {\partial \log \Lambda_i} \right)
 &= \left[[\mathbf {dO}_{\mathrm{intra}} \mathbf V_i^\top] \odot [\mathbf Q_i \mathbf K_i^\top]\odot \mathbf M \right]  \mathbf 1 -\left[[\mathbf {dO}_{\mathrm{intra}} \mathbf V_i^\top] \odot [\mathbf Q_i \mathbf K_i^\top]\odot \mathbf M \right]^\top \mathbf 1.

\end{aligned}
$$
