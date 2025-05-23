# Backward

递推式为：
$$
\begin{aligned}
\mathbf O_i &= \mathbf {\bar Q}_i \mathbf S_{i-1} +
\left[
\mathbf {\bar Q}_i \mathbf {\bar K}_i^\top  \odot \mathbf M
\right] \mathbf V_i + \left[
\mathbf {\bar Q}_i \mathbf {\bar A}_i^\top  \odot \mathbf M
\right] (\mathbf P_i + \mathbf U_i \mathbf S_{i-1}), \\

\mathbf S_i
&= \Pi_{i, c}\mathbf S_{i-1} +  \mathbf {\tilde A}_i^\top (\mathbf U_i \mathbf S_{i-1} + \mathbf P_i) +
\mathbf {\tilde K}_i^\top \mathbf V_i,  \\

\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf U_i &= \mathbf {\bar C}_i, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf P_i &= \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i.
\end{aligned}
$$
为了方便讨论，记：
$$
\mathbf D_i = \mathbf U_i \mathbf S_{i-1} + \mathbf P_i.
$$
那么：
$$
\begin{aligned}
\mathbf O_i &= \mathbf {\bar Q}_i \mathbf S_{i-1} +
\left[
\mathbf {\bar Q}_i \mathbf {\bar K}_i^\top  \odot \mathbf M
\right] \mathbf V_i + \left[
\mathbf {\bar Q}_i \mathbf {\bar A}_i^\top  \odot \mathbf M
\right] \mathbf D_i , \\

\mathbf S_i
&= \Pi_{i, c}\mathbf S_{i-1} +  \mathbf {\tilde A}_i^\top \mathbf D_i +
\mathbf {\tilde K}_i^\top \mathbf V_i,  \\

\mathbf D_i &= \mathbf U_i \mathbf S_{i-1} + \mathbf P_i, \\

\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf U_i &= \mathbf {\bar C}_i, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf P_i &= \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i.
\end{aligned}
$$
我们将上述四个部分分别称为$o, s, d, u, p$，后续讨论中会使用到。



## $\mathbf {dQ}$

$\mathbf {Q}$的计算只涉及到$o$，类似GLA的思路，不难得到：
$$
\mathbf {dQ}_i = [\mathbf {dO}_i \mathbf S_{i-1}^\top] \odot \Pi_i
+ [[\mathbf {dO}_i \mathbf V_i^\top  \odot \mathbf M] \mathbf {\bar K}_i] \odot \Pi_i
+ [[\mathbf {dO}_i \mathbf D_i^\top  \odot \mathbf M] \mathbf {\bar A}_i] \odot \Pi_i.
$$
我们将前一部分称：
$$
\begin{aligned}
\mathbf {dQ}_{i, 1} &= [\mathbf {dO}_i \mathbf S_{i-1}^\top] \odot \Pi_i
+ [[\mathbf {dO}_i \mathbf V_i^\top  \odot \mathbf M] \mathbf {\bar K}_i] \odot \Pi_i, \\
\mathbf {dQ}_{i, 2} &=  [[\mathbf {dO}_i \mathbf D_i^\top  \odot \mathbf M] \mathbf {\bar A}_i] \odot \Pi_i, \\
\mathbf {dQ} &=  \mathbf {dQ}_{i, 1} + \mathbf {dQ}_{i, 2}.


\end{aligned}
$$


## $\mathbf {dS}_i$

$\mathbf {S}_i$的计算涉及到$o, s,d$，所以不难得到：
$$
\begin{aligned}
\mathbf {dS}_{i-1} &= \Pi_{i, c}   \mathbf {dS}_i+
\mathbf {\bar Q}_i^\top \mathbf {d \bar O}_i  + \mathbf {U}_i^\top \mathbf {dD_i}.
\end{aligned}
$$


## $\mathbf{dD}, \mathbf {dU}, \mathbf {dP}$

$\mathbf D$涉及到$o, s$，所以：
$$
\mathbf {dD}_i = \left[
\mathbf {\bar Q}_i \mathbf {\bar A}_i^\top  \odot \mathbf M
\right]^\top \mathbf {dO}_i  + \mathbf {\tilde A}_i \mathbf {dS}_i.
$$
根据$\mathbf D, \mathbf U, \mathbf P$的关系，可以得到：
$$
\begin{aligned}
\mathbf {dP}_i &= \mathbf {dD}_i, \\
\mathbf {dU}_i &= \mathbf {dD}_i \mathbf S_{i-1}^\top.

\end{aligned}
$$


## $\mathbf {dK}$ part 1

$\mathbf K$的计算涉及到$o, s, p$，我们先求$o, s$部分：
$$
\mathbf {dK}_{i, 1} = \left[\mathbf {dO}_i \mathbf V_i^\top \odot \mathbf M \right]^\top \mathbf {\bar Q}_i / \Pi_i + \left[ \mathbf V_i \mathbf {dS}_i^\top  \right]
\odot \Theta_i.
$$


## $\mathbf {dV}$ part 1

$\mathbf V$的计算涉及到$o, s, p$，我们先求$o, s$部分：
$$
\mathbf {dV}_{i, 1} = \left[
\mathbf {\bar Q}_i \mathbf {\bar K}_i^\top  \odot \mathbf M
\right]^\top \mathbf {dO}_i  + \mathbf {\tilde K}_i \mathbf {dS}_i.
$$


## $\mathbf {dA}$ part 1

$\mathbf A$的计算涉及到$o, s, u, p$，我们先求$o, s$部分：
$$
\mathbf {dA}_{i, 1} = \left[\mathbf {dO}_i \mathbf D_i^\top \odot \mathbf M \right]^\top \mathbf {\bar Q}_i / \Pi_i +  \left[ \mathbf D_i \mathbf {dS}_i^\top  \right]
\odot \Theta_i.
$$


## $\mathbf {dK}, \mathbf {dV}, \mathbf {dA}, \mathbf {dC}$ part2

回顾公式：
$$
\begin{aligned}
\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf P_i &= \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i, \\
\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)] \mathbf U_i &= \mathbf {\bar C}_i.
\end{aligned}
$$
对于第一个公式，我们有：
$$
\mathbf P_i =  \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1) \mathbf P_i + \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1) \mathbf V_i.
$$
所以给定$\mathbf {dP}_i$，我们有：
$$
\begin{aligned}
\mathbf {dC}_i &=
\left[
\mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top, -1 )\mathbf {\bar A}_i
+ \mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )\mathbf {\bar K}_i
\right] \odot \Pi_i,  \\

\mathbf {dA}_i &=
\left[
\mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top, -1 )^\top \mathbf {\bar C}_i
\right] / \Pi_i, \\

\mathbf {dK}_i &= \left[\mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )^\top \mathbf {\bar C}_i \right]/ \Pi_i, \\

\mathbf {dV}_i &= \left[
\mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
\odot \mathbf M, -1)
\right]^\top \mathbf P_i.
\end{aligned}
$$
对于第二个公式：
$$
\mathbf U_i = \mathbf {\bar C}_i + \mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar A}_i^\top
\odot \mathbf M, -1)\mathbf U_i.
$$
所以给定$\mathbf {dU}_i$，我们有：
$$
\begin{aligned}
\mathbf {dC}_i &= \mathbf {dU}_i + [\mathrm{tril}(\mathbf {dU}_i \mathbf U_i, -1) \mathbf {\bar A}_i ] \odot \Pi_i, \\

\mathbf {dA}_i &=[\mathrm{tril}(\mathbf {dU}_i \mathbf U_i, -1)^\top  \mathbf {\bar C}_i ]/ \Pi_i.

\end{aligned}
$$
综上：
$$
\begin{aligned}
\mathbf {dC}_i &=  \left[
\mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top, -1 )\mathbf {\bar A}_i
+ \mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )\mathbf {\bar K}_i
+ \mathrm{tril}(\mathbf {dU}_i \mathbf U_i, -1) \mathbf {\bar A}_i
\right] \odot \Pi_i + \mathbf {dU}_i,  \\

\mathbf {dA}_{i, 2} &=
\left[ \mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top + \mathbf {dU}_i \mathbf U_i, -1 )^\top \mathbf {\bar C}_i
\right] / \Pi_i,  \\

\mathbf {dK}_{i, 2} &= \left[\mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )^\top \mathbf {\bar C}_i \right]/ \Pi_i, \\

\mathbf {dV}_{i, 2} &= \left[
\mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
, -1)
\right]^\top \mathbf P_i.
\end{aligned}
$$


## $\mathbf {d}\log \lambda_t$

记：
$$
\pi_i = \prod_{t=1}^i\lambda_t.
$$
那么：
$$
\begin{aligned}
\mathbf O &= \mathbf {\bar Q} \mathbf S_{0} +
\left[
\mathbf {\bar Q} \mathbf {\bar K}^\top  \odot \mathbf M
\right] \mathbf V + \left[
\mathbf {\bar Q} \mathbf {\bar A}^\top  \odot \mathbf M
\right] (\mathbf P+ \mathbf U \mathbf S_{0}), \\

\mathbf S_n
&= \pi_n \mathbf S_{0} +  \mathbf {\tilde A}^\top (\mathbf U \mathbf S_{0} + \mathbf P) +
\mathbf {\tilde K}^\top \mathbf V, \\

\ [\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf U &= \mathbf {\bar C}, \\

[\mathbf I - \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar A}^\top
\odot \mathbf M, -1)] \mathbf P &= \mathrm{tril}(\mathbf {\bar C} \mathbf {\bar K}^\top
\odot \mathbf M, -1) \mathbf V, \\

\mathbf {\bar Q} &= \mathbf Q \odot \Alpha =\mathbf Q \odot \exp(\log \Pi) ,  \\
\mathbf {\bar K} &= \mathbf K / \Alpha =  \mathbf K\odot \exp(-\log\Pi) ,  \\
\mathbf {\bar A} &= \mathbf A / \Alpha =  \mathbf K\odot \exp(-\log\Pi).
\end{aligned}
$$
注意到：
$$
\begin{aligned}
 \frac{\partial l}{\partial \mathbf {q}_i} \frac{\partial \mathbf {q}_i}{\partial \log \pi_i}

&= \mathbf {dq}_i \odot \left(
\frac{\partial \mathbf {q}_i}{\partial \mathbf {\bar q}_i}
\frac {\partial \mathbf {\bar q}_i} {\partial \log \pi_i}
\right) \\

&= \mathbf {dq}_i \odot \left(
\frac{1}{\alpha_i} \odot
\mathbf q_i \odot \alpha_i
\right) \\

& =\mathbf {dq}_i \odot \mathbf q_i.

\end{aligned}
$$
我们首先考虑$\mathbf O $部分对于梯度的贡献：
$$
\begin{aligned}
\mathbf {d}\log \pi_i
&= \frac{\partial l}{\partial \log \pi_i} \\
&= \frac{\partial l}{\partial \mathbf {q}_i} \frac{\partial \mathbf {q}_i}{\partial \log \pi_i}
+  \frac{\partial l}{\partial \mathbf {k}_i} \frac{\partial \mathbf {k}_i}{\partial \log \pi_i}
+  \frac{\partial l}{\partial \mathbf {a}_i} \frac{\partial \mathbf {a}_i}{\partial \log \pi_i}
+ \frac{\partial l}{\partial \mathbf {c}_i} \frac{\partial \mathbf {c}_i}{\partial \log \pi_i} \\


&= \mathbf {dq}_i \odot \mathbf q_i + \mathbf {dc}_i \odot \mathbf c_i - \mathbf {dk}_i \odot \mathbf k_i - \mathbf {da}_i \odot \mathbf a_i.
\end{aligned}
$$
然后考虑$\mathbf s_n$部分贡献的梯度，这部分使用vector decay的结论为（只有$t=n$时才有梯度）：
$$
[\mathbf {ds}_n \odot \mathbf s_n]1_e.
$$
综上，这部分的梯度为：
$$
\mathbf {d}\log \pi_i=
\begin{cases}
\mathbf {dq}_i \odot \mathbf q_i + \mathbf {dc}_i \odot \mathbf c_i - \mathbf {dk}_i \odot \mathbf k_i - \mathbf {da}_i \odot \mathbf a_i, & i < n \\
\mathbf {dq}_i \odot \mathbf q_i + \mathbf {dc}_i \odot \mathbf c_i - \mathbf {dk}_i \odot \mathbf k_i - \mathbf {da}_i \odot \mathbf a_i + [\mathbf s_n \odot \mathbf {ds}_n]1_e, & i=n
\end{cases}
$$
因此：
$$
\mathbf d{\log}\lambda_t = [\mathbf s_n \odot \mathbf {ds}_n]1_e + \sum_{j\ge t}
\mathbf {d}\log \pi_j.
$$




## 总结

$\mathbf {dQ}$：
$$
\begin{aligned}
\mathbf {dQ}_{i, 1} &= [\mathbf {dO}_i \mathbf S_{i-1}^\top] \odot \Pi_i
+ [[\mathbf {dO}_i \mathbf V_i^\top  \odot \mathbf M] \mathbf {\bar K}_i] \odot \Pi_i, \\
\mathbf {dQ}_{i, 2} &=  [[\mathbf {dO}_i \mathbf D_i^\top  \odot \mathbf M] \mathbf {\bar A}_i] \odot \Pi_i, \\
\mathbf {dQ} &=  \mathbf {dQ}_{i, 1} + \mathbf {dQ}_{i, 2}.


\end{aligned}
$$
$\mathbf {dK}$：
$$
\begin{aligned}
\mathbf {dK}_{i, 1}  &= \left[\mathbf {dO}_i \mathbf V_i^\top \odot \mathbf M \right]^\top \mathbf {\bar Q}_i / \Pi_i + \left[ \mathbf V_i \mathbf {dS}_i^\top  \right]
\odot \Theta_i, \\
\mathbf {dK}_{i, 2} &= \left[\mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )^\top \mathbf {\bar C}_i \right]/ \Pi_i,  \\

\mathbf {dK} &=\mathbf {dK}_{i, 1}  + \mathbf {dK}_{i, 2}.
\end{aligned}
$$
$\mathbf {dV}$：
$$
\begin{aligned}
\mathbf {dV}_{i, 1} &= \left[
\mathbf {\bar Q}_i \mathbf {\bar K}_i^\top  \odot \mathbf M
\right]^\top \mathbf {dO}_i  + \mathbf {\tilde K}_i \mathbf {dS}_i, \\


\mathbf {dV}_{i, 2} &= \left[
\mathrm{tril}(\mathbf {\bar C}_i \mathbf {\bar K}_i^\top
, -1)
\right]^\top \mathbf P_i, \\

\mathbf {dK} &= \mathbf {dV}_{i, 1} + \mathbf {dV}_{i, 2}.

\end{aligned}
$$
$\mathbf {dA}$：
$$
\begin{aligned}
\mathbf {dA}_{i, 1} &= \left[\mathbf {dO}_i \mathbf D_i^\top \odot \mathbf M \right]^\top \mathbf {\bar Q}_i / \Pi_i +  \left[ \mathbf D_i \mathbf {dS}_i^\top  \right]
\odot \Theta_i, \\

\mathbf {dA}_{i, 2} &=
\left[ \mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top + \mathbf {dU}_i \mathbf U_i, -1 )^\top \mathbf {\bar C}_i
\right] / \Pi_i, \\

\mathbf {dA} &= \mathbf {dA}_{i, 1} + \mathbf {dA}_{i, 2}.

\end{aligned}
$$
$\mathbf {dC}$：
$$
\mathbf {dC}_i =
\left[
\mathrm{tril}( \mathbf {dP}_i \mathbf P_i^\top, -1 )\mathbf {\bar A}_i
+ \mathrm{tril}( \mathbf {dP}_i \mathbf V_i^\top, -1 )\mathbf {\bar K}_i
\right] \odot \Pi_i
$$
$\mathbf {d}\log \Lambda$：
$$
\begin{aligned}
\mathbf {d}\log \pi_i &=
\begin{cases}
\mathbf {dq}_i \odot \mathbf q_i + \mathbf {dc}_i \odot \mathbf c_i - \mathbf {dk}_i \odot \mathbf k_i - \mathbf {da}_i \odot \mathbf a_i, & i < n \\
\mathbf {dq}_i \odot \mathbf q_i + \mathbf {dc}_i \odot \mathbf c_i - \mathbf {dk}_i \odot \mathbf k_i - \mathbf {da}_i \odot \mathbf a_i + [\mathbf s_n \odot \mathbf {ds}_n]1_e, & i=n
\end{cases} \\

\mathbf d{\log}\lambda_t &= [\mathbf s_n \odot \mathbf {ds}_n]1_e + \sum_{j\ge t}
\mathbf {d}\log \pi_j.
\end{aligned}
$$

注意到实际使用时，我们会使用如下参数化方案：
$$
\mathbf c_t = -\beta_t \mathbf a_t.
$$
那么：
$$
\begin{aligned}
\mathbf {dA} &= \mathbf {dA} - \Beta \mathbf {dC}, \\
\mathbf {d\Beta} &= -\mathrm{sum}(\mathbf {dC} \odot \mathbf a).
\end{aligned}
$$
如果使用参数共享，即$\mathbf A = \mathbf K$，那么需要进一步化简梯度。
