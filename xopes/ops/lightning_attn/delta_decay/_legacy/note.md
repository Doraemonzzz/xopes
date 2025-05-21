# 求逆部分加速

## Forward

$$
\begin{aligned}
\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf U &= \mathbf {\bar C}, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf {P} &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V , \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf { P} &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V - \mathbf V + \mathbf V, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf {\bar P} &=  \mathbf V, \\

\mathbf P &= \mathbf {\bar P}  - \mathbf V, \\

\mathbf {\bar C} &=  \mathbf C \odot \mathbf {\Gamma} , \\
\mathbf {\bar A} &=\mathbf A / \mathbf {\Gamma}, \\
\mathbf {\bar K} &= \mathbf K / \mathbf {\Gamma}, \\
\mathbf b_t &= \mathbf c_t \odot \lambda_t, \\

\mathbf r_t & =\mathbf p_t + \mathbf s_0^\top \mathbf u_t, \\
\mathbf s_t &= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf r_t^\top, \\

\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{e}.
\end{aligned}
$$

记：
$$
\mathbf G_{ts} =\mathbf I_c \mathbf 1_{t=s} - \mathrm{tril}(\mathbf {\bar C}_t \mathbf {\bar A}_s^\top
\odot \mathbf M_{ts}, -1).
$$
考虑方程：
$$
\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf X = \mathbf Y.
$$
记：
$$
\begin{aligned}
\mathbf X_t &= \mathbf G_{tt}^{-1} \left( \mathbf Y_t- \sum_{s=1}^{t-1} \mathbf G_{ts} \mathbf X_s  \right) \\
&= \mathbf G_{tt}^{-1}  \left( \mathbf Y_t- \sum_{s=1}^{t-1} \mathbf {\bar C}_t   \mathbf {\bar A}_s^\top \mathbf X_s  \right) \\
&= \mathbf G_{tt}^{-1}  \left( \mathbf Y_t- \mathbf {\bar C}_t  \sum_{s=1}^{t-1}   \mathbf {\bar A}_s^\top \mathbf X_s  \right)  \\
&\triangleq \mathbf G_{tt}^{-1}  \left( \mathbf Y_t- \mathbf {\bar C}_t  \mathbf H_{t-1}  \right),  \\

\mathbf H_t &= \mathbf H_{t-1} +   \mathbf {\bar A}_t^\top \mathbf X_t
\end{aligned}
$$

那么我们考虑如下算法：
$$
\begin{aligned}
\mathbf T_t &= \mathbf G_{tt}^{-1}, \\

\mathbf {\bar c}_t &= \gamma_t \odot \mathbf c_t, \\

\mathbf h_t & = \mathrm{diag}(\lambda_t) \mathbf h_{t-1}  + \mathbf a_t [\mathbf {u}_t^\top, \mathbf {\bar p}_t^\top], \\


\mathbf U_t &= \mathbf T_t \left(  \mathbf {\bar C}_t - \mathbf {\bar C}_t \mathbf H_{t-1,1} \right), \\

\mathbf {\bar P}_t &= \mathbf T_t \left( \mathbf V_t - \mathbf {\bar C}_t  \mathbf H_{t-1,2}\right).
\end{aligned}
$$

### 特殊情况
假设$\mathbf a_t = \mathbf k_t$，此时：
$$
\begin{aligned}
\mathbf s_t &= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top + \mathbf a_t \mathbf r_t^\top\\

&= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t (\mathbf v_t+  \mathbf r_t)^\top \\

&= \mathrm{diag}(\lambda_t)\mathbf s_{t-1} + \mathbf k_t ( \mathbf {\bar p}_t + \mathbf s_0^\top \mathbf u_t)^\top.


\end{aligned}
$$

注意到上述形式计算量大增，所以这个思路在实际中用处不大，仅作参考。
