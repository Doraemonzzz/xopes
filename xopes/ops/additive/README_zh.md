

# 加法形式的递推

## 朴素实现

前向：
$$
\begin{aligned}
g&=\max(g_t) \\
\bar k_t &=\exp(g_t-g) \odot k_t, \\
s_t &= s_{t-1} +\bar k_t^T v_t,\\
\Delta_t &=\Delta_{t-1} + \exp(g_t-g), \\
\bar {s}_t &=diag\{\Delta_t\}^{-1} s_t, \\
o_t &=q_t \bar {s}_t. \\
\end{aligned}
$$

另一种前向计算方式：
$$
\begin{aligned}
\bar {s}_t &= diag\{\Delta_{t-1}/\Delta_t\}  \bar {s}_{t-1} +diag\{1/\Delta_t\}\bar k_t^T v_t, \\
o_t &=q_t \bar {s}_t. \\

\end{aligned}
$$
反向：
$$
dq_t=(do_t \bar{s}_t^T),\\
d\bar{s}_t=d \bar{s}_{t-1} + q_{t-1}^T do_{t-1},\\
d  {s}_t=diag\{\Delta_t\}^{-1}  d \bar{s}_t,\\
d {\Delta}_t=\sum_{v} (d {s}_t \odot \left(diag\{\Delta_t\}^{-2} d\bar s_t \right))\\
d k_t =v_t d {s}_t^T \\
\bar g_t = \exp(g_t) \\
d\bar g_t =\sum_{j\ge t} d\Delta_j\\
dg_t =d\bar g_t\odot \bar g_t \\
dv_t = \bar k_t d {s}_t
$$
上式有数值爆炸的问题，考虑下式：
$$
\bar q_i =q_i\bar g_i/ \Delta_i \\

d {s}_t=d {s}_{t-1} + \bar q_{t-1}^T do_{t-1}\\

\bar dg_t = d\bar q_t \odot (-1/s_t-\bar g_t/s_t^2) +\sum_{i>t} d\bar q_i \odot (-\bar g_i/s_i^2)  \\
d g_t =d\bar  g_t  \odot \bar g_t\\
$$



## 数值稳定版本

$$
\begin{aligned}
m_0 &= -\infty, \\
m_t  &= \max(m_{t-1}, g_t), \\
\lambda_t &= \exp(m_{t-1}-m_t)  \\
\bar k_t &=\exp(g_t-m_t) \odot k_t, \\
s_t &= \lambda_t s_{t-1} +\bar k_t^T v_t,\\
\Delta_t &=\lambda_t \Delta_{t-1} + \exp(g_t-m_t), \\
\bar {s}_t &=diag\{\Delta_t\}^{-1} s_t, \\
o_t &=q_t \bar {s}_t. \\
\end{aligned}
$$





## 数值稳定版本(gla版本)

$$
\begin{aligned}
\Delta_0 &= 0 \\
f_t &=\log \left( \exp(f_{t-1}) + \exp(g_t) \right), \\
\Delta_t &=\exp(f_t)=\Delta_{t-1} + \exp(g_t) ,\\
i_t &=g_t - f_t,  \\
\exp(i_t) & = \exp(g_t) / \Delta_t ,\\
\bar k_t &= \exp(i_t) \odot k_t, \\
s_t &=\mathrm{diag}\{\Delta_{t-1}/\Delta_t\} s_{t-1} +\mathrm{diag}\{ \exp(g_t)/\Delta_t\}  k_t^T v_t,\\
&=\mathrm{diag}\{\exp(f_{t-1}-f_t)\} s_{t-1} + \mathrm{diag}\{\exp(i_t)\} k_t^Tv_t\\
&=\mathrm{diag}\{\exp(f_{t-1}-f_t)\} s_{t-1} + \bar k_t^Tv_t,
o_t &=q_t  {s}_t. \\
\end{aligned}
$$
步骤：

1. 先计算$f_t$，注意$f_0=-\infty$
2. 然后loop计算;



反向：

首先定义虚拟的前向（不用于实际前向计算）：
$$
\begin{aligned}
\bar k_t &= \exp(g_t) \odot k_t, \\
\bar s_t &= \bar s_{t-1} + \bar k_t^T v_t,\\
\Delta_{t}&=\Delta_{t-1}+\exp(g_t), \\
s_t &=\mathrm{diag}\{\Delta_t^{-1} \} \bar s_t, \\
o_t &= q_t s_t \\
&= q_t \mathrm{diag}\{\Delta_t^{-1} \} \bar s_t \\
&= (q_t/\Delta_t) \bar s_t\\
&=\bar q_t \bar s_t

\end{aligned}
$$
先在考虑反向：
$$
dq_t=do_t s_t^T,
$$
然后考虑$d\bar s_{t}$，定义：
$$
d {\bar s}_t=d {\bar s}_{t-1} + \bar q_{t-1}^T do_{t-1},\\
d\bar k_t = v_t d\bar s_t^T, \\
dk_t =d\bar k_t \odot \exp(u_t),   \\
dv_t = \bar k_t  d \bar s_t.
$$
展开此式：(check this)
$$
d {\bar s}_{t-1}=d {\bar s}_{t} +\mathrm{diag}\{\Delta_t^{-1}\}q_t^T do_{t},\\
d\bar k_t = v_t d\bar s_t^T, \\
dk_t =d\bar k_t \odot \exp(g_t)=(v_t d\bar s_t^T)  \mathrm{diag}\{\exp(g_t) \}=v_t  ds_t^T,   \\
dv_t = \bar k_t  d \bar s_t =k_t  \mathrm{diag}\{\exp(g_t) \}d\bar s_t =k_t ds_t \\
\mathrm{diag}\{\Delta_t - \Delta_{t-1}\}^{-1} ds_{t-1} =\mathrm{diag}\{\Delta_{t-1} - \Delta_{t-2}\}^{-1} ds_{t}
+\mathrm{diag}\{\Delta_t^{-1}\}q_t^T do_{t-1}
$$


最后考虑$d\bar g_t = d\exp(g_t)$：
$$
d \bar g_t = \sum_{i\ge t} d\bar q_i \frac{d\bar q_i }{d\bar g_t } +  \sum_{i\ge t} d\bar k_i \frac{d\bar k_i }{d\bar g_t } \\
=\sum_{i\ge t} d\bar q_i \odot (-\bar q_i /\Delta_i^2) + d\bar k_t \odot k_t\\

 d\bar q_t =   d q_t \odot \Delta_t,d\bar k_t =   d k_t / \bar g_t    \\

   d\bar g_t =  -\sum_{i\ge t} d\bar q_i \odot (q_i /\Delta_i^2)+ d\bar k_t\odot k_t  \\
d g_t =d\bar  g_t  \odot \bar g_t\\
dg_t = d k_t /\bar g_t\odot k_t\odot \bar g_t -\sum_{i\ge t} d\bar q_i \odot (q_i /\Delta_i^2) \odot \bar g_t \\
= d k_t\odot k_t  - \bar g_t   \sum_{i\ge t} d q_i \odot ( q_i /s_i  ) \\
$$



## block level实现

### 数值稳定版本

考虑递推：
$$
s_t = \mathrm{diag}\{\lambda_t\} s_{t-1} + k_t v_t^T \in \mathbb R^{d\times e}, \\
o_t^T=q_t^T  s_t \in \mathbb R^{1\times e}.  \\
o_t=s_t^T q_t \in \mathbb R^{e}.
$$
为了讨论方便，省略diag符号：
$$
s_t =\lambda_t  s_{t-1} + k_t v_t^T \in \mathbb R^{d\times e}, \\
o_t=s_t^T q_t \in \mathbb R^{e}.
$$
假设$s_0=0$，展开上式可得：
$$
\begin{aligned}
s_t
&=\lambda_t  s_{t-1} + k_t v_t^T  \\
& =\lambda_t (\lambda_{t-1} s_{t-2} + k_{t-1} v_{t-1}^T)+  k_t v_t^T  \\
&=  \sum_{s\le t} \left(\prod_{j=s+1}^t \lambda_j  \right)  k_sv_s^T  \\
&=\sum_{s\le t} \left(\Lambda_t /\Lambda_s  \right)  k_sv_s^T \\
\Lambda_t & = \prod_{s=0}^t \lambda_s, \\
\lambda_0 & = 1.
\end{aligned}
$$
因此：
$$
\begin{aligned}
o_t^T
& = q_t ^T s_t  \\
& =q_t ^T\sum_{s\le t} \left(\Lambda_t /\Lambda_s  \right)  k_sv_s^T  \\
&= (q_t \odot \Lambda_t)   \sum_{s\le t}\left( \frac{k_s}{\Lambda_s }   \right)v_s^T
\end{aligned}
$$
写成矩阵形式可得：
$$
O = \left(\left( \left(Q\odot \Lambda\right) \left(\frac{K}{\Lambda} \right)^T  \right)
\odot M
\right) V
$$
考虑chunk size为$B$，
$$
S_t =s_{tB}=\sum_{s\le tB} \left(\Lambda_{tB} /\Lambda_s  \right)  k_sv_s^T
$$
考虑位置$tB+r, 1\le r \le B$：
$$
\begin{aligned}
o_{tB+r}^T
&=q_{tB+r}^T \sum_{s\le tB +r} \left(\Lambda_{tB +r} /\Lambda_s  \right)  k_sv_s^T   \\
&=q_{tB+r}^T
\left(
\sum_{s=tB+1}^{tB+r} \left(\Lambda_{tB +r} /\Lambda_s  \right)  k_sv_s^T +
(\Lambda_{tB+r} /\Lambda_{tB})\sum_{s=1}^{tB} \left(\Lambda_{tB} /\Lambda_s  \right)  k_sv_s^T
\right) \\
&= q_{tB+r}^T \sum_{s=tB+1}^{tB+r} \left(\Lambda_{tB +r} /\Lambda_s  \right)  k_sv_s^T
+ q_{tB+r}^T (\Lambda_{tB+r} /\Lambda_{tB})\sum_{s=1}^{tB} \left(\Lambda_{tB} /\Lambda_s  \right)  k_sv_s^T
\end{aligned}
$$
写成矩阵：
$$
O_{t+1}=\left[ \underbrace {\left( \left(Q_{t+1} \odot \Lambda_{t+1}\right) \left(\frac{K_{t+1}}{\Lambda_{t+1}} \right)^T  \right)}_{P}
\odot M \right] V_{t+1}
+ \left[Q_{t+1} \odot \Tau_{t+1}\right] S_{t}\\
 \Tau_{t+1,r} =\Lambda_{tB+r} /\Lambda_{tB}, \Gamma_{t+1, r}= \Lambda_{(t+1)B} /\Lambda_{tB+r}, \\
 S_{t+1}=\sum_{s=tB+1}^{tB+B} \left(\Lambda_{(t+1)B} /\Lambda_s  \right)  k_sv_s^T
+  (\Lambda_{(t+1)B} /\Lambda_{tB})\sum_{s=1}^{tB} \left(\Lambda_{tB} /\Lambda_s  \right)  k_sv_s^T \\
= \Tau_{t+1,B} S_t +  (\Tau_{t+1}\odot K_{t+1})^T V_{t+1}.
$$
最后讨论$P$：
$$
P_{ij}= \sum_{k=1}^d Q_{ik} K_{jk} (\Lambda_{ik}/\Lambda_{jk})
$$
