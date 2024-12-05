#### 前向反向

其中$\mathbf D$表示$\mathrm{diag}$符号。

##### 前向

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\log \mathbf r_{0}=\mathbf 0, \mathbf u_{0}=\mathbf 0,\mathbf h_{0} =\mathbf 0,\\
\log \mathbf  r_{t} &=\log \mathbf r_{t-1}+\mathbf e_t, \\

  \log \mathbf s_{t} &= \log \mathbf s_{t-1} +  \log \mathbf r_{t}, \\
\mathbf u_t &= \mathbf r_t-\mathbf r_0, \\

\textbf{additive decay}:
\mathbf s_{0}& =\mathbf 0,\mathbf u_{0}=\mathbf 0,\mathbf h_{0} =\mathbf 0, \\
\mathbf  u_{t} &=\mathbf u_{t-1}+\mathbf e_t, \\
\mathbf  s_{t} &=\mathbf s_{t-1} +\mathbf u_{t}, \\
\textbf{compute}:\mathbf p_{t}&=\mathbf D(\mathbf u_{t-1} / \mathbf u_t)\mathbf p_{t-1} +  ((1-\mathbf u_{t-1} /\mathbf u_{t}) \odot \mathbf k_t) \mathbf{v}_{t}^\top,   \\
\mathbf h_{t}&=\mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf h_{t-1} + \mathbf D(1-\mathbf s_{t-1} /\mathbf s_{t}) \mathbf p_t,\\
\mathbf o_t&= \mathbf h_{t}^\top \mathbf q_t.
\end{aligned}
$$



##### 反向

compute part:
$$
\begin{aligned}
\mathbf {dq}_t & =\mathbf h_t\mathbf {do}_t \\
\mathbf {dh}_{t-1}&=\mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf {dh}_{t} + \mathbf q_{t-1}\mathbf{do}_{t-1}^{\top}  \\
\mathbf {dp}_{t-1}&= (\mathbf u_{t-1} / \mathbf u_t)\mathbf {dp}_{t}+\mathbf D(1-\mathbf s_{t-1} /\mathbf s_{t})\mathbf {dh}_{t-1}   \\
\mathbf {dv}_{t}&= \mathbf {dp}_{t}^{\top} ((1-\mathbf u_{t-1} /\mathbf u_{t}) \odot \mathbf k_t) \\
\mathbf {dk}_{t}&= (\mathbf {dp}_{t} \mathbf v_t) \odot (1-\mathbf u_{t-1} /\mathbf u_{t})
 \\


\end{aligned}
$$
记
$$
\begin{aligned}
\gamma_t &=\mathbf u_{t-1}/\mathbf u_t=\exp(\log \mathbf u_{t-1}- \log \mathbf u_t )\\
\lambda_t &=\mathbf s_{t-1}/\mathbf s_t=\exp(\log \mathbf s_{t-1}- \log \mathbf s_t )
\end{aligned}
$$




#### Naive

$$
\begin{aligned}
\mathbf {d\lambda}_t
& = \frac{\partial \mathcal L}{\partial \lambda_t}   \\
& = \frac{\partial \mathcal L}{\partial \mathbf h_t} \frac{\partial \mathbf h_t}{\partial \lambda_t}   \\
&= (\mathbf {dh_t}\odot \left( \mathbf h_{t-1}-\mathbf p_t \right))\mathbf 1 \\

\mathbf {d s}_t
&= \frac{\partial \mathcal L}{\mathbf s_t}   \\
&= \frac{\partial \mathcal L}{\partial \lambda_t} \frac{\partial \lambda_t}{\mathbf s_t}
+ \frac{\partial \mathcal L}{\partial \lambda_{t+1}} \frac{\partial \lambda_{t+1}}{\mathbf s_t}  \\
&=((\mathbf {dh_t}\odot \left( \mathbf h_{t-1}-\mathbf p_t \right))\mathbf 1) \odot \left( -\frac{\mathbf s_{t-1}}{\mathbf s_t^2}\right)
+((\mathbf {dh_{t+1}}\odot \left( \mathbf h_{t}-\mathbf p_{t+1} \right))\mathbf 1)\odot \left( \frac{1}{\mathbf s_{t+1}}\right)   \\

\mathbf {d \gamma}_t
& = \frac{\partial \mathcal L}{\partial \gamma_t}   \\
& = \frac{\partial \mathcal L}{\partial \mathbf p_t} \frac{\partial \mathbf p_t}{\partial \lambda_t}   \\
&= (\mathbf {dp_t}\odot \left( \mathbf p_{t-1}-\mathbf k_t \mathbf v_t^\top \right))\mathbf 1 \\

\mathbf {d \bar u}_t
&= \frac{\partial \mathcal L}{\mathbf u_t}   \\
&= \frac{\partial \mathcal L}{\partial \gamma_t} \frac{\partial \gamma_t}{\mathbf u_t}
+ \frac{\partial \mathcal L}{\partial \gamma_{t+1}} \frac{\partial \gamma_{t+1}}{\mathbf u_t}  \\
&=((\mathbf {dp_t}\odot \left( \mathbf p_{t-1}-\mathbf k_t\mathbf v_t^\top \right))\mathbf 1) \odot \left( -\frac{\mathbf p_{t-1}}{\mathbf p_t^2}\right)
+((\mathbf {dp_{t+1}}\odot \left( \mathbf p_{t}-\mathbf k_{t+1}\mathbf v_{t+1}^\top \right))\mathbf 1)  \odot \left( \frac{1}{\mathbf p_{t+1}}\right)   \\
\mathbf {d u}_t &=
\mathbf {d \bar u}_t  + \sum_{k\ge t}\mathbf {ds}_k\\
\mathbf {de}_t &= \sum_{k\ge t}\mathbf {du}_k
\end{aligned}
$$





展开$\mathbf h_t$的计算式可得：
$$
\begin{aligned}
\mathbf p_t
&=\mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+ \sum_{k=1}^t\left(\frac{\mathbf u_{k}-\mathbf u_{k-1}}{\mathbf u_t} \odot \mathbf k_k \right) \mathbf{v}_{t}^\top  \\
&=  \mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+\mathbf D\left( \frac{1}{\mathbf u_t} \right)\sum_{k=1}^t\left(({\mathbf u_{k}-\mathbf u_{k-1}}) \odot \mathbf k_k \right) \mathbf{v}_{k}^\top
\\
\mathbf h_t &= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right) \mathbf p_k   \\
&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)

\end{aligned}
$$


#### Additive

如果additive decay，那么
$$
\mathbf  s_{t} =\mathbf s_{t-1} +\mathbf u_{t}
$$
梯度1:
$$
\begin{aligned}
\mathbf h_t
&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}
\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\

\mathbf o_t^{\top}&= \mathbf q_t^\top \mathbf h_t \\
&=\mathbf q_t^\top \left(\mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+ \mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&=(\mathbf q_t/ \mathbf s_t)^\top \left(\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}\sum_{j=1}^k\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&= (\mathbf q_t/ \mathbf s_t)^\top\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}
\left(
(\mathbf q_t/ \mathbf s_t)^\top \mathbf{D}(\mathbf u_0)\mathbf p_0 +
\sum_{j=1}^k (\mathbf q_t/ \mathbf s_t)^\top \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\

\mathbf {d q}_t&=  \frac{1}{\mathbf s_t} \odot \left(
\mathbf D(\mathbf s_0) \mathbf h_0 \mathbf {do}_t+
\sum_{k=1}^{t}
\left(
\mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t+
\sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \mathbf {do}_t  \right) \right)\\
&\triangleq   \mathbf {dq}_t^- +  \mathbf {dq}_t^+\\


\mathbf {ds}_t &= -\frac{\mathbf q_t}{\mathbf s_t^2} \odot \left( \sum_{k=1}^{t}
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \mathbf {do}_t    \right) \right)  \\
&= -\frac{\mathbf q_t\odot \mathbf {dq}_t^+ }{\mathbf s_t}   \\
\mathbf {ds}_0 &= \sum_{t}\frac{\mathbf q_t}{\mathbf s_t}\odot  (\mathbf h_0 \mathbf {do}_t)
=\sum_{t} {\mathbf {dq}_t^- \odot  \mathbf q_t}

\end{aligned}
$$
梯度2:(no use)
$$
\begin{aligned}
\frac{\mathbf {do}_t}{\mathbf {dk}_j}
&= \sum_{k=j}^t k \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1}) \\
&=\frac{(t+j)(t-j+1)}{2} \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1}) \\

\mathbf {dk}_j
&= \sum_{t\ge j}\frac{\mathbf {do}_t}{\mathbf {dk}_j} \\
& =\sum_{t\ge j} \frac{(t+j)(t-j+1)}{2} \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1})  \\

\frac{\partial \mathbf{do}_t}{\partial \mathbf {du}_j}
&= \sum_{k=j}^t k \mathbf {do}_t^\top \mathbf v_j \frac{\partial\left( (\mathbf q_t/ \mathbf s_t)^\top \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right)  \right)}{\partial \mathbf u_j}
-\sum_{k=j+1}^t k \mathbf {do}_t^\top \mathbf v_{j+1} \frac{\partial\left( (\mathbf q_t/ \mathbf s_t)^\top \left(({\mathbf u_{j+1}-\mathbf u_{j}}) \odot \mathbf k_{j+1} \right)  \right)}{\partial \mathbf u_j} \\
&=\sum_{k=j}^t k (\mathbf {do}_t^\top \mathbf v_j)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j - \sum_{k=j+1}^t k (\mathbf {do}_t^\top \mathbf v_{j+1})
\mathbf q_t/ \mathbf s_t \odot \mathbf k_{j+1}  \\
&=\frac{(j+t)(t-j+1)}{2} (\mathbf {do}_t^\top \mathbf v_j)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j - \frac{(t+j+1)(t-j)}{2} (\mathbf {do}_t^\top \mathbf v_{j+1})
\mathbf q_t/ \mathbf s_t \odot \mathbf k_{j+1}\\
&=\frac{\partial \mathbf{do}_t^+}{\partial \mathbf {du}_j} -\frac{\partial \mathbf{do}_t^-}{\partial \mathbf {du}_j} \\

\mathbf {du_j} &=\sum_{k\ge j}\frac{\partial \mathbf{do}_k^+}{\partial \mathbf {du}_j} -\sum_{k\ge j+1}\frac{\partial \mathbf{do}_k^-}{\partial \mathbf {du}_j}  \\
&= \sum_{k\ge j} \frac{(k+j)(k-j+1)}{2} (\mathbf {do}_k^\top \mathbf v_j)
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j} -\sum_{k\ge j+1} \frac{(k+j+1)(k-j)}{2} (\mathbf {do}_k^\top \mathbf v_{j+1})
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j+1} \\
&= \sum_{k\ge j} \frac{(k+j)(k-j+1)}{2} (\mathbf {do}_k^\top \mathbf v_j)
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j} -\frac{(k+j+1)(k-j)}{2} (\mathbf {do}_k^\top \mathbf v_{j+1})
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j+1}  \\
&=  \frac{\mathbf {dk}_j \odot \mathbf k_j}{\mathbf u_j - \mathbf u_{{j-1}}}- \frac{\mathbf {dk}_{j+1} \odot \mathbf k_{j+1}}{\mathbf u_{j+1} - \mathbf u_{{j}}}

\end{aligned}
$$
总结：
$$
\mathbf {du}_t =\mathbf {du}_t+ \sum_{j\ge t} \mathbf {d  s}_j, \mathbf {de}_j = \sum_{j\ge t} \mathbf {du}_j
= \sum_{j\ge t}  \mathbf {du}_j + \sum_{j\ge t} \sum_{k\ge j} \mathbf {d  s}_k
= \frac{\mathbf {dk}_t \odot \mathbf k_t}{\mathbf e_t } + \sum_{j\ge t} \sum_{k\ge j} \mathbf {d  s}_k
$$
梯度2另一种方案(use this)：
$$
\begin{aligned}
\mathbf o_t^{\top}&= \mathbf q_t^\top \mathbf h_t \\
&=\mathbf q_t^\top \left( \mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}
\left(
\mathbf{D}(\mathbf u_0)\mathbf p_0+
\sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)
\right)\\
&=(\mathbf q_t/ \mathbf s_t)^\top \left( \sum_{k=1}^{t}
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0+  + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \right) \\
&= \sum_{k=1}^{t}
\left( (\mathbf q_t/ \mathbf s_t)^\top \mathbf{D}(\mathbf u_0)\mathbf p_0+  \sum_{j=1}^k (\mathbf q_t/ \mathbf s_t)^\top \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\
&= \sum_{k=1}^{t}
\left((\mathbf q_t/ \mathbf s_t)^\top \mathbf{D}(\mathbf u_0)\mathbf p_0+ \sum_{j=1}^k (\mathbf q_t/ \mathbf s_t)^\top \left({\mathbf e_{j}} \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \right)  \\



\end{aligned}
$$
那么：
$$
\begin{aligned}
\frac{\mathbf {do}_t}{\mathbf {dk}_j}
&= \sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1}) \\
&=(t-j+1)\mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1}) \\
&=(t-j+1) \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot \mathbf e_j  \\


\mathbf {dk}_j
&= \sum_{t\ge j}\frac{\mathbf {do}_t}{\mathbf {dk}_j} \\
& =\sum_{t\ge j}(t-j+1)\mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1})  \\
& =\sum_{t\ge j}(t-j+1) \mathbf {do}_t^\top \mathbf v_j (\mathbf q_t/\mathbf s_t) \odot \mathbf e_j   \\

\frac{\partial \mathbf{do}_t}{\partial \mathbf {de}_j}
&= \sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j \frac{\partial\left( (\mathbf q_t/ \mathbf s_t)^\top \left({\mathbf e_{j}} \odot \mathbf k_j \right)  \right)}{\partial \mathbf u_j}
 \\
&=\sum_{k=j}^t  (\mathbf {do}_t^\top \mathbf v_j)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j   \\
&=(t-j+1)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j \\


\mathbf {de_j} &=\sum_{k\ge j}\frac{\partial \mathbf{do}_k}{\partial \mathbf {de}_j}  \\
&= \sum_{k\ge j}(k-j+1) (\mathbf {do}_k^\top \mathbf v_j)
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j}  \\
&=  \frac{\mathbf {dk}_j \odot \mathbf k_j}{\mathbf e_j }  \\

\mathbf{du}_0 &= \sum_{t}\ t \mathbf p_0 \mathbf {do}_t\\
&=  \sum_{t} \frac{t(t+1)}{2} \mathbf p_0 \mathbf {do}_t
\end{aligned}
$$
那么：
$$
\mathbf {du}_t = \sum_{j\ge t} \mathbf {d  s}_j, \mathbf {de}_j = \mathbf {de}_j+\sum_{j\ge t} \mathbf {du}_j
=\mathbf {de}_j+\sum_{j\ge t} \sum_{k\ge j}  \mathbf {d  s}_k .
$$



### Multiplicative

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\log \mathbf u_{0}=\mathbf 0, \mathbf u_{0}=\mathbf 0,\mathbf h_{0} =\mathbf 0,\\
\log \mathbf  u_{t} &=\log \mathbf u_{t-1}+\mathbf e_t, \\

  \log \mathbf s_{t} &= \log \mathbf s_{t-1} +  \log \mathbf u_{t}, \\

\textbf{compute}:\mathbf p_{t}&=\mathbf D(\mathbf u_{t-1} / \mathbf u_t)\mathbf p_{t-1} +  ((1-\mathbf u_{t-1} /\mathbf u_{t}) \odot \mathbf k_t) \mathbf{v}_{t}^\top,   \\
\mathbf h_{t}&=\mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf h_{t-1} +\mathbf p_t,\\
\mathbf o_t&= \mathbf h_{t}^\top \mathbf q_t.
\end{aligned}
$$



##### 反向

compute part:
$$
\begin{aligned}
\mathbf {dq}_t & =\mathbf h_t\mathbf {do}_t \\
\mathbf {dh}_{t-1}&=\mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf {dh}_{t} + \mathbf q_{t-1}\mathbf{do}_{t-1}^{\top}  \\
\mathbf {dp}_{t-1}&= (\mathbf u_{t-1} / \mathbf u_t)\mathbf {dp}_{t}+\mathbf {dh}_{t-1}   \\
\mathbf {dv}_{t}&= \mathbf {dp}_{t}^{\top} ((1-\mathbf u_{t-1} /\mathbf u_{t}) \odot \mathbf k_t) \\
\mathbf {dk}_{t}&= (\mathbf {dp}_{t} \mathbf v_t) \odot (1-\mathbf u_{t-1} /\mathbf u_{t})
 \\


\end{aligned}
$$
记
$$
\begin{aligned}
\gamma_t &=\mathbf u_{t-1}/\mathbf u_t=\exp(\log \mathbf u_{t-1}- \log \mathbf u_t )\\
\lambda_t &=\mathbf s_{t-1}/\mathbf s_t=\exp(\log \mathbf s_{t-1}- \log \mathbf s_t )
\end{aligned}
$$


$$
\begin{aligned}


\mathbf p_t
&=\mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+ \sum_{k=1}^t\left(\frac{\mathbf u_{k}-\mathbf u_{k-1}}{\mathbf u_t} \odot \mathbf k_k \right) \mathbf{v}_{t}^\top  \\
&=  \mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+\mathbf D\left( \frac{1}{\mathbf u_t} \right)\sum_{k=1}^t\left(({\mathbf u_{k}-\mathbf u_{k-1}}) \odot \mathbf k_k \right) \mathbf{v}_{k}^\top
\\

\mathbf h_t &= \mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf h_{t-1} +\mathbf p_t  \\
&=\mathbf D(\mathbf s_{t-1} / \mathbf s_t)
\left(\mathbf D(\mathbf s_{t-2} / \mathbf s_{t-1}) \mathbf h_{t-2} +\mathbf p_{t-1} \right) +\mathbf p_t  \\
&=\mathbf D(\mathbf s_{t-2} / \mathbf s_t) \mathbf h_{t-2}+\mathbf D(\mathbf s_{t-1} / \mathbf s_t) \mathbf p_{t-1}  +\mathbf p_t  \\
&=\mathbf D(\mathbf s_{0} / \mathbf s_t) \mathbf h_{0}+
\sum_{k=1}^t\mathbf D(\mathbf s_{k} / \mathbf s_t) \mathbf p_{k}   \\

&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}} \right)

\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)

\end{aligned}
$$
梯度1:

$$
\begin{aligned}
\mathbf h_t
&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D (\mathbf s_k)
\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\

\mathbf o_t^{\top}&= \mathbf q_t^\top \mathbf h_t \\
&=\mathbf q_t^\top \left(\mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+ \mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D (\mathbf s_k)\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&=(\mathbf q_t/ \mathbf s_t)^\top \left(\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}\mathbf D (\mathbf s_k)\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&= (\mathbf q_t/ \mathbf s_t)^\top \mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}
\left(
(\mathbf q_t/ \mathbf s_t)^\top \mathbf D (\mathbf s_k) \mathbf{D}(\mathbf u_0)\mathbf p_0 +
(\mathbf q_t/ \mathbf s_t)^\top  \mathbf D (\mathbf s_k)\sum_{j=1}^k \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\

\mathbf {d q}_t&=  \frac{1}{\mathbf s_t} \odot \left(
\mathbf D(\mathbf s_0) \mathbf h_0 \mathbf {do}_t+
\sum_{k=1}^{t}
\left(
\mathbf D (\mathbf s_k)\mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t+
 \mathbf D (\mathbf s_k) \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \mathbf {do}_t  \right) \right)\\
&\triangleq   \mathbf {dq}_t^- +  \mathbf {dq}_t^+

\end{aligned}
$$



注意到：
$$
\begin{aligned}
\mathbf {ds}_t =& -\frac{\mathbf q_t}{\mathbf s_t^2} \odot \left(  \left(
\mathbf D(\mathbf s_0) \mathbf h_0 \mathbf {do}_t+
\sum_{k=1}^{t}
\left(
\mathbf D (\mathbf s_k)\mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t+
 \mathbf D (\mathbf s_k) \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \mathbf {do}_t  \right) \right)\\ \right) \\
+& \sum_{k\ge t} \frac{\mathbf q_k}{\mathbf s_k \odot \mathbf u_t}
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \mathbf {do}_k    \right)
\end{aligned}
$$


注意到：
$$
\begin{aligned}
\mathbf h_t
&=\mathbf D(\mathbf s_{0} / \mathbf s_t) \mathbf h_{0}+
\sum_{k=1}^t\mathbf D(\mathbf s_{k} / \mathbf s_t) \mathbf p_{k} \\

\mathbf o_t^{\top}&= \mathbf q_t^\top \mathbf h_t \\
&= \mathbf q_t^\top \left( \mathbf D(\mathbf s_{0} / \mathbf s_t) \mathbf h_{0}+
\sum_{k=1}^t\mathbf D(\mathbf s_{k} / \mathbf s_t) \mathbf p_{k} \right)\\
&= (\mathbf q_t/\mathbf s_t)^\top  \left( \mathbf D(\mathbf s_{0} ) \mathbf h_{0}+
\sum_{k=1}^t\mathbf D(\mathbf s_{k}) \mathbf p_{k} \right)\\
&\triangleq (\mathbf q_t/\mathbf s_t)^\top {\mathbf {\bar  h}}_t
\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf {dq}_t &= \frac{1}{\mathbf s_t}\odot ( {\mathbf {\bar  h}}_t \mathbf {do}_t)\\
\mathbf {dk}_t &= \sum_{j\ge t}
(\mathbf u_j -\mathbf u_{j-1})\odot
\sum_{k=j}^t (\mathbf q_t/\mathbf s_t)^\top \mathbf D(\mathbf s_k)\mathbf v_j^\top \mathbf {do}_t

\\
\mathbf {ds}_t &= -\frac{\mathbf q_t}{\mathbf s_t^2} \odot ( {\mathbf {\bar  h}}_t \mathbf {do}_t)
+ (\mathbf p_t \mathbf {do}_t)\odot \sum_{j\ge t} \frac{\mathbf q_j}{\mathbf s_j}   \\
&= -\frac{\mathbf q_t \odot \mathbf {dq}_t}{\mathbf s_t}+ (\mathbf p_t \mathbf {do}_t)\odot \sum_{j\ge t} \frac{\mathbf q_j}{\mathbf s_j}  \\

\end{aligned}
$$
另一方面：
$$
\begin{aligned}
\frac{\mathbf {do}_t}{\mathbf {dk}_j}
&= \sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j \mathbf D(\mathbf s_k)(\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1}) \\



\mathbf {dk}_j
&= \sum_{t\ge j}\frac{\mathbf {do}_t}{\mathbf {dk}_j} \\
& =\sum_{t\ge j}\sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j \mathbf D(\mathbf s_k)(\mathbf q_t/\mathbf s_t) \odot (\mathbf u_j - \mathbf u_{j-1})   \\

\frac{\partial \mathbf{do}_t}{\partial (\mathbf u_j - \mathbf u_{j-1})}
&= \sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j \frac{\partial\left( (\mathbf q_t/ \mathbf s_t)^\top \mathbf D(\mathbf s_k) \left((\mathbf u_j - \mathbf u_{j-1}) \odot \mathbf k_j \right)  \right)}{\partial (\mathbf u_j - \mathbf u_{j-1})}
 \\
&=\sum_{k=j}^t  (\mathbf {do}_t^\top \mathbf v_j)
\mathbf D(\mathbf s_k)\mathbf q_t/ \mathbf s_t\odot \mathbf k_j   \\


\mathbf {d}(\mathbf u_j - \mathbf u_{j-1} ) &=\sum_{k\ge j}\frac{\partial \mathbf{do}_k}{\partial (\mathbf u_j - \mathbf u_{j-1})}  \\
&=  \frac{\mathbf {dk}_j \odot \mathbf k_j}{\mathbf u_j - \mathbf u_{j-1}}  \\

\mathbf {d}\mathbf u_j  &=
\sum_{k=1}^j\frac{\mathbf {dk}_k \odot \mathbf k_k}{\mathbf u_k - \mathbf u_{k-1}}
\end{aligned}
$$









#### Multiplicative(no read, need check, update later)

$$
\mathbf  s_{t} =\mathbf s_{t-1}\mathbf r_{t}=\mathbf s_{t-1}(\mathbf u_{t} +\mathbf r_0),
(\mathbf s_k - \mathbf s_{k-1})/\mathbf u_k = (\mathbf s_{k-1}\mathbf r_k - \mathbf s_{k-1})/\mathbf u_k
$$

记：
$$
\alpha_t =\sum_{j} \mathbf e_j, \\
\mathbf r_t = \mathbf r_0 \exp(\alpha_t), \mathbf u_t =  \mathbf r_t - \mathbf r_0=\mathbf r_0(\exp(\alpha_t ) - 1), \\
\mathbf s_t = \mathbf s_{t-1}\mathbf r_t = \mathbf s_{t-1}\mathbf r_0  \exp(\alpha_t), \\
(\mathbf s_t - \mathbf s_{t-1})/\mathbf u_t =\mathbf s_{t-1}(\mathbf r_0 \exp(\alpha_t) - 1)/ \mathbf r_0(\exp(\alpha_t ) - 1)
$$


那么：
$$
\begin{aligned}
\mathbf p_t
&=\mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+ \sum_{k=1}^t\left(\frac{\mathbf u_{k}-\mathbf u_{k-1}}{\mathbf u_t} \odot \mathbf k_k \right) \mathbf{v}_{t}^\top  \\
&=  \mathbf D\left(\frac{\mathbf u_0}{\mathbf u_t}\right) \mathbf p_0+\mathbf D\left( \frac{1}{\mathbf u_t} \right)\sum_{k=1}^t\left(({\mathbf u_{k}-\mathbf u_{k-1}}) \odot \mathbf k_k \right) \mathbf{v}_{k}^\top
\\
\mathbf h_t &= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right) \mathbf p_k   \\
&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)

\end{aligned}
$$
梯度1:
$$
\begin{aligned}
\mathbf h_t
&= \mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+\mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t} \mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\

\mathbf o_t^{\top}&= \mathbf q_t^\top \mathbf h_t \\
&=\mathbf q_t^\top \left(\mathbf D \left(\frac{\mathbf s_0}{{\mathbf s_t}} \right) \mathbf h_0+ \mathbf D \left(\frac{1}{{\mathbf s_t}} \right)\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&=(\mathbf q_t/ \mathbf s_t)^\top \left(\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)\sum_{j=1}^k\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
&= (\mathbf q_t/ \mathbf s_t)^\top\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ (\mathbf q_t/ \mathbf s_t)^\top \sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left(
\mathbf{D}(\mathbf u_0)\mathbf p_0 +
\sum_{j=1}^k  \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\
&\triangleq (\mathbf q_t/ \mathbf s_t)^\top\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ (\mathbf q_t/ \mathbf s_t)^\top \sum_{k=1}^{t}\mathbf A_k \left(
\mathbf{D}(\mathbf u_0)\mathbf p_0 +
\sum_{j=1}^k  \left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right) \\


\mathbf {d q}_t&=  \frac{1}{\mathbf s_t} \odot \left(
\mathbf D(\mathbf s_0) \mathbf h_0 \mathbf {do}_t+
\sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left(
\mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t+
\sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \mathbf {do}_t  \right) \right)\\
&\triangleq   \mathbf {dq}_t^- +  \mathbf {dq}_t^+\\


\mathbf {ds}_t &= -\frac{\mathbf q_t}{\mathbf s_t^2} \odot \left( \sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \mathbf {do}_t    \right) \right)  \\
&= -\frac{\mathbf q_t\odot \mathbf {dq}_t^+ }{\mathbf s_t}   \\
\mathbf {ds}_0 &= \sum_{t}\frac{\mathbf q_t}{\mathbf s_t}\odot  (\mathbf h_0 \mathbf {do}_t)
=\sum_{t} {\mathbf {dq}_t^- \odot  \mathbf q_t}

\end{aligned}
$$




1
$$
\begin{aligned}
\mathbf {ds}_t =& -\frac{\mathbf q_t}{\mathbf s_t^2} \odot \left( \sum_{k=1}^{t}\mathbf D\left( {\mathbf s_{k}-\mathbf s_{k-1}} \right)
\mathbf D\left( \frac{1}{\mathbf u_k} \right)
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \mathbf {do}_t    \right) \right) \\
+& \sum_{k\ge t} \frac{\mathbf q_k}{\mathbf s_k \odot \mathbf u_t}
\left( \mathbf{D}(\mathbf u_0)\mathbf p_0 \mathbf{do}_t + \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top \mathbf {do}_k    \right)
\end{aligned}
$$






记：
$$
\begin{aligned}
 o_t^\top &=c+\sum_{k=1}^t \bar q_t^\top f(k)\sum_{j=1}^k \bar k_{j} v_j^\top
\end{aligned}
$$
那么：

在$t$时刻：
$$
d \bar k_j=\sum_{k=j}^t  \bar q_t^\top f(k) v_j^\top do_t, \\
d k_j=(u_j-u_{j-1})\odot \sum_{k=j}^t \bar q_t^\top f(k) v_j^\top do_t
$$
累加全部时刻可得：
$$
dk_t = \sum_{j\ge t} (u_j-u_{j-1})\odot \sum_{k=1}^t  (q_t/s_t)^\top D((s_k-s_{k-1})/u_k)) v_j^\top do_t
$$
