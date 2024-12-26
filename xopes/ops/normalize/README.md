# Normalize

对于输入$\mathbf x\in \mathbb R^d$，本节讨论各种normalize的算法，算法定义为：
$$
\mathbf o=f(\mathbf x)= c\times \frac{\mathbf x_1 \odot \mathbf w}{\sqrt{\mathbf x_1^\top \mathbf x_1}} + \mathbf b.
$$
其中$c\in\mathbb R, \mathbf w, \mathbf b\in \mathbb R^d$，$\mathbf x_1 = \mathbf x$或者$\mathbf x_1 =\mathbf x- \bar {\mathbf x},\bar {\mathbf x}=(\sum_{i=1}^d x_i)/d$。特别的，我们考虑如下算子：
$$
\mathbf o= f(\mathbf x + \mathbf y).
$$
其中$\mathbf y$为残差分支，这在Transformer中很常见。当不使用$\mathbf w, \mathbf b$时，等价于$\mathbf w=\mathbf b=\mathbf 1$；当不使用$\mathbf y$时，等价于$\mathbf y=\mathbf 0$。



## Forward

输入：$\mathbf x, \mathbf y, \mathbf w, \mathbf b, c$，其中$c$是常数，不可学。

计算：
$$
\begin{aligned}
\mathbf p  & =\mathbf x+\mathbf y, \\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o&=c\times\mathbf r \odot \mathbf w + \mathbf b.

\end{aligned}
$$


## Backward

输入：$\mathbf {do}$。

计算：
$$
\begin{aligned}
\mathbf {db}&= \mathbf {do},\\
\mathbf {dw}&= \mathbf {do} \odot (c\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do} \odot (c\times \mathbf w),\\
\mathbf {dq}&= \mathbf {dr}\odot \left(1/\sigma- \mathbf q/\sigma^2 \odot \frac{\partial \sigma}{\partial \mathbf q}  \right)\\
&=  \mathbf {dr}\odot \left(1/\sigma- \mathbf q/\sigma^2 \odot \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2\mathbf q \right)  \right) \\
&=  \mathbf {dr}\odot \left(1/\sigma- \mathbf q/\sigma^2 \odot \left( \mathbf q /\sigma \right)  \right) \\
&=  \mathbf {dr}\odot \left(1/\sigma- \mathbf r^2 /\sigma  \right) \\
&= (\mathbf {dr}/\sigma)\odot (1-\mathbf r^2), \\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp},\\
\mathbf {dy}& = \mathbf {dp}.

\end{aligned}
$$


