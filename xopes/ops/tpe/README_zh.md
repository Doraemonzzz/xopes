​

# 简化情形

## Forward

$$
h_t = \lambda_t h_{t-1} + k_t v_t^T \in \mathbb R^{e\times d}, \\
o_t^T=1_e h_t \triangleq  q_t h_t \in \mathbb R^{1\times d} .
$$



## Backward

$$
dh_n = 0, \\
dh_{t-1} =\lambda_t d h_t + k_t do_t^T \in \mathbb R^{e\times d}, \\
dk_t^T =v_t^Tdh_{t}^T \in \mathbb R^{1\times e},
dv_t^T = k_t^Tdh_t \in \mathbb R^{1\times d}, \\
dk=\sum d k_t,
$$

另一方面，展开公式1可得：

记：
$$
\gamma_t = \sum_{j=1}^t \log \lambda_j,\\
$$
那么：
$$
\begin{aligned}
o_t^T
&=\sum_{i=1}^t  q_t^T\mathrm{diag}\left\{\prod_{j=i+1}^{t}\lambda_j \right \} k_i v_i^T \\
&= \sum_{i=1}^t (q_t \odot \exp(\gamma_t))^T(k_i \odot \exp(-\gamma_i)) v_i^T.
\end{aligned}
$$
关于$q_t$求导可得：
$$
\begin{aligned}
dq_t&=\frac{\partial l}{\partial o_t} \frac{\partial o_t}{\partial
\left( q_t  \odot \exp(\gamma_t) \right)} \frac {\partial
\left( q_t  \odot \exp(\gamma_t) \right)} {\partial q_t }\\
&= (v_i^T do_t) \sum_{i=1}^t(k_i \odot \exp(-\gamma_i))  \odot \exp(\gamma_t)\\
&= \sum_{i=1}^t (v_i^T do_t ) k_i \odot \exp(-\gamma_i)  \odot \exp(\gamma_t ), \\

dk_t&=\sum_{i=t}^n\frac{\partial l}{\partial o_i} \frac{\partial o_i}{\partial
\left( k_t  \odot \exp(-\gamma_t) \right)} \frac {\partial
\left( k_t  \odot \exp(-\gamma_t) \right)} {\partial k_t }\\
&=  (v_t^T do_i) q_i \odot \exp(\gamma_i)\odot \exp(-\gamma_t), \\

d\gamma_t &=\frac{\partial l}{\partial o_t} \frac{\partial o_t}{\partial
\left( q_t  \odot \exp(\gamma_t) \right)} \frac {\partial
\left( q_t  \odot \exp(\gamma_t) \right)} {\partial \gamma_t }

+ \sum_{i=t}^n\frac{\partial l}{\partial o_i} \frac{\partial o_i}{\partial
\left( k_t  \odot \exp(-\gamma_t) \right)} \frac {\partial
\left( k_t  \odot \exp(-\gamma_t) \right)} {\partial \gamma_t }
\\

&= q_t \odot \left((v_i^T do_t) \sum_{i=1}^t(k_i \odot \exp(-\gamma_i))  \odot \exp(\gamma_t) \right)
-k_t \odot \left( (v_t^T do_i) q_i \odot \exp(\gamma_i)\odot \exp(-\gamma_t) \right) \\
&= q_t\odot dq_t- k_t \odot dk_t
\end{aligned}
$$
另一方面：
$$
d\log \lambda_j = \sum_{t=j}^n d\gamma_t
$$
注意此时第一项为0，所以：
$$
d\log \lambda_j  = -\sum_{t=j}^n k_t \odot dk_t=-\sum_{t=j}^n k \odot dk_t=-k\odot \sum_{t=j}^n dk_t,
$$


# 非简化情形

## Forward

$$
h_t = \lambda_t h_{t-1} + k_t \odot v_t^T \in \mathbb R^{e\times d}, \\
o_t^T=1_e h_t \triangleq  q_t h_t \in \mathbb R^{1\times d} .
$$



## Backward

##

$$
dh_n = 0, \\
dh_{t-1} =\lambda_t d h_t + k_t \odot do_t^T \in \mathbb R^{e\times d}, \\
dk_t^T =v_t^Tdh_{t}^T \in \mathbb R^{1\times e},
dv_t^T = k_t^Tdh_t \in \mathbb R^{1\times d}, \\
dk=\sum d k_t,
$$

另一方面：
$$
d\lambda_t =\frac{\partial l}{\partial h_t}\frac{\partial h_t}{\partial \lambda_t}
=dh_t h_{t-1}^T 1_d
$$
