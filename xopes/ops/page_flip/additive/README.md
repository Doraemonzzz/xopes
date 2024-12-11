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
&=(\mathbf q_t/ \mathbf s_t)^\top \left(\mathbf D \left({\mathbf s_0} \right) \mathbf h_0+ \sum_{k=1}^{t}\left( \mathbf {D}(\mathbf u_0) \mathbf p_0+ \sum_{j=1}^k\left(({\mathbf u_{j}-\mathbf u_{j-1}}) \odot \mathbf k_j \right) \mathbf{v}_{j}^\top  \right)\right) \\
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
=\sum_{t} \frac{{\mathbf {dq}_t^- \odot  \mathbf q_t}}{\mathbf s_0}

\end{aligned}
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

\frac{\mathbf{do}_t}{\mathbf {de}_j}
&= \sum_{k=j}^t  \mathbf {do}_t^\top \mathbf v_j \frac{\partial\left( (\mathbf q_t/ \mathbf s_t)^\top \left({\mathbf e_{j}} \odot \mathbf k_j \right)  \right)}{\partial \mathbf e_j}
 \\
&=\sum_{k=j}^t  (\mathbf {do}_t^\top \mathbf v_j)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j   \\
&=(t-j+1)  (\mathbf {do}_t^\top \mathbf v_j)
\mathbf q_t/ \mathbf s_t\odot \mathbf k_j \\


\mathbf {de}_j &=\sum_{k\ge j}\frac{ \mathbf{do}_k}{ \mathbf {de}_j}  \\
&= \sum_{k\ge j}(k-j+1) (\mathbf {do}_k^\top \mathbf v_j)
\mathbf q_k/ \mathbf s_k \odot \mathbf k_{j}  \\
&=  \frac{\mathbf {dk}_j \odot \mathbf k_j}{\mathbf e_j }  \\

\mathbf{du}_0 &= \mathbf p_0  \odot\sum_{t}(\mathbf q_t / \mathbf s_t)
\end{aligned}
$$
