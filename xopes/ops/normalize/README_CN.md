# Normalize with residual or gate

对于输入$\mathbf x\in \mathbb R^d$，本节讨论各种normalize的算法，算法定义为：
$$
\begin{aligned}
\mathbf o
&=f(\mathbf x)  \\
&= c\times \frac{\mathbf x_1 \odot \mathbf w}{\sqrt{\mathbf x_1^\top \mathbf x_1}} + \mathbf b \\
&= (c/\sqrt{d})\times  \frac{\mathbf x_1 \odot \mathbf w}{\sqrt{\mathbf x_1^\top \mathbf x_1/d}} + \mathbf b.
\end{aligned}
$$
（其中最后一个等式是为了减少数值精度带来的影响。）

其中$c\in\mathbb R, \mathbf w, \mathbf b\in \mathbb R^d$，$\mathbf x_1 = \mathbf x$或者$\mathbf x_1 =\mathbf x- \bar {\mathbf x},\bar {\mathbf x}=(\sum_{i=1}^d x_i)/d$。



我们考虑两类算子，其中一类包含残差链接，我们称为residual with residual：
$$
\mathbf o= f(\mathbf x + \mathbf y).
$$
其中$\mathbf y$为残差分支，这在Transformer中很常见。当不使用$\mathbf w, \mathbf b$时，等价于$\mathbf w=\mathbf b=\mathbf 1$；当不使用$\mathbf y$时，等价于$\mathbf y=\mathbf 0$。

另一类为residual with gate：
$$
\mathbf o = f(\mathbf x  \odot g(\mathbf y)) \text{ or }
\mathbf o = f(\mathbf x)  \odot g(\mathbf y) .
$$


## Normalize with residual

### Forward

输入：$\mathbf x, \mathbf y, \mathbf w, \mathbf b, c$，其中$c$是常数，不可学。

计算：
$$
\begin{aligned}
\mathbf p  & =\mathbf x+\mathbf y, \\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b.

\end{aligned}
$$



### Backward

输入：$\mathbf {do}$。

计算：
$$
\begin{aligned}
\mathbf {db}&= \mathbf {do},\\
\mathbf {dw}&= \mathbf {do} \odot (c/\sqrt d\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do} \odot (c /\sqrt d\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt d\times q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3 /d   \\
&=1/\sigma  (1_{i=j}-r_i r_j /d)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r /d   \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp},\\
\mathbf {dy}& = \mathbf {dp}.

\end{aligned}
$$



### Fuse Normalize and Residual

下面考虑对于Transformer layer，Fuse Normalize and Residual应该如何实现。

Naive实现，假设输入为$\mathbf x$：
$$
\begin{aligned}
\mathbf x_0& = \mathbf x, \\
\mathbf y_k &= \mathrm{norm}(\mathbf x_{k-1}), \\
\mathbf x_k &= f_k(\mathbf y_k) +\mathbf x_{k-1},  \\
\mathbf o&= \mathbf x_n, \\
k&=1,\ldots, n,
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf y_k &= \mathrm{norm}(\mathbf x_{k-1}) \\
&= \mathrm{norm}(\mathbf x_{k-2}+ \mathbf z_{k-1}), \\
 \mathbf z_k &\triangleq f_k(\mathbf y_k), \\
 \mathbf x_k &= \mathbf x_{k-1}+\mathbf z_k. \\


\end{aligned}
$$
根据上述观察，我们得到Fuse实现：
$$
\begin{aligned}
\mathbf p_0& = \mathbf x, \\
\mathbf r_0& = \mathbf 0, \\
\mathbf r_{k}&= \mathbf p_{k-1}+ \mathbf r_{k-1} , \\
\mathbf q_k &=   \mathrm{norm}( \mathbf r_k), \\
\mathbf p_k &= f_k(\mathbf q_k), \\

\mathbf o&= \mathbf p_n + \mathbf r_n, \\
k&=1,\ldots, n,
\end{aligned}
$$
下面用数学归纳法证明两种计算方法的结果相同。

当$n=1$时，
$$
\begin{aligned}
\mathbf o_{\mathrm{naive}}&= \mathbf x_1 \\
&= f_1(\mathbf y_1) +\mathbf x_{0}\\
&=  f_1(\mathrm{norm}(\mathbf x_0)) +\mathbf x_{0}\\
&=  f_1(\mathrm{norm}(\mathbf x)) +\mathbf x  \\

\mathbf o_{\mathrm{fuse}}&= \mathbf p_1 +\mathbf r_1 \\
&=f_1(\mathbf q_1)+\mathbf r_1\\
&= f_1(\mathrm{norm}(\mathbf p_0 + \mathbf r_0))+\mathbf p_0 \\
&= f_1(\mathrm{norm}(\mathbf x))+\mathbf x.
\end{aligned}
$$
假设$n-1$时结论成立，那么$n$时：
$$
\begin{aligned}
\mathbf o_{\mathrm{naive}}&= \mathbf x_n \\
&= f_n(\mathbf y_n) +\mathbf x_{n-1}\\
&=  f_n(\mathrm{norm}(\mathbf x_{n-1})) +\mathbf x_{n-1}\\


\mathbf o_{\mathrm{fuse}}&= \mathbf p_n +\mathbf r_n \\
&=f_n(\mathbf q_n)+\mathbf p_{n-1} + \mathbf r_{n-1}\\
&=f_n(\mathrm{norm}(\mathbf p_{n-1} + \mathbf r_{n-1}))+\mathbf p_{n-1} + \mathbf r_{n-1}.
\end{aligned}
$$
根据归纳假设，我们有：
$$
\mathbf x_{n-1}=\mathbf p_{n-1} + \mathbf r_{n-1}.
$$
所以：
$$
\begin{aligned}

\mathbf o_{\mathrm{fuse}}
&=f_n(\mathrm{norm}(\mathbf p_{n-1} + \mathbf r_{n-1}))+\mathbf p_{n-1} + \mathbf r_{n-1}\\
&=  f_n(\mathrm{norm}(\mathbf x_{n-1})) +\mathbf x_{n-1} \\
&= \mathbf o_{\mathrm{naive}}.
\end{aligned}
$$
所以结论成立。



#### 反向传播更新

注意到此时函数的输入为输出为：
$$
\mathbf o =\mathrm{norm}(\mathbf x+ \mathbf y), \mathbf r=\mathbf x+\mathbf y.
$$
所以：
$$
\begin{aligned}
\mathbf {dx} & = \mathbf {dx} + \mathbf {dr}, \\
\mathbf {dy}  &= \mathbf {dx} .
\end{aligned}
$$


## Normalize with residual

### Forward

输入：$\mathbf x, \mathbf y, \mathbf w, \mathbf b, c$，其中$c$是常数，不可学。

计算：

pre-gate:
$$
\begin{aligned}
\mathbf p  & =\mathbf x\odot  g(\mathbf y),\\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o_1&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b, \\
\mathbf o& = \mathbf o_1.

\end{aligned}
$$

post-gate:
$$
\begin{aligned}
\mathbf p  & =\mathbf x,\\
\mathbf q &= \mathbf p, \mathrm{or}, \\
\mathbf q&=\mathbf p-\left(\sum_{i=1}^d p_i\right)/d,\\
\sigma&= \sqrt{\mathbf q^\top \mathbf q/d}, \\
\mathbf r&= \mathbf q /\sigma, \\
\mathbf o_1&=(c/\sqrt d)\times\mathbf r \odot \mathbf w + \mathbf b, \\
\mathbf o& = \mathbf o_1\odot g(\mathbf y).

\end{aligned}
$$


### Backward

输入：$\mathbf {do}$。

计算：

pre-gate:
$$
\begin{aligned}
\mathbf {do}_1 & = \mathbf {do} ,  \\
\mathbf {db}&= \mathbf {do}_1,\\
\mathbf {dw}&= \mathbf {do}_1 \odot (c/\sqrt d\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do}_1 \odot (c /\sqrt d\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt d\times q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3 /d   \\
&=1/\sigma  (1_{i=j}-r_i r_j /d)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r /d   \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp} \odot g(\mathbf y) ,\\
\mathbf {dy}& = \mathbf {dp} \odot \mathbf x, \\

\mathbf {dy} &= g'(\mathbf y) \odot \mathbf {dy}.

\end{aligned}
$$

post-gate:
$$
\begin{aligned}
\mathbf {do}_1 & = \mathbf {do} \odot g(\mathbf y) ,  \\
\mathbf {db}&= \mathbf {do}_1,\\
\mathbf {dw}&= \mathbf {do}_1 \odot (c/\sqrt d\times \mathbf r),  \\
\mathbf {d r}&= \mathbf {do}_1 \odot (c /\sqrt d\times \mathbf w),\\
\frac{\partial r_i}{\partial q_j}
&= 1_{i=j}/\sigma - q_i /\sigma^2 \frac{\partial \sigma}{\partial q_j}  \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left(1/2 \times  (\mathbf q^\top \mathbf q)^{-1/2}\times 2 q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_i /\sigma^2 \left( \sigma^{-1}/\sqrt d\times q_j /\sqrt d \right)   \\
&= 1_{i=j}/\sigma - q_iq_j /\sigma^3 /d   \\
&=1/\sigma  (1_{i=j}-r_i r_j /d)   \\

\frac{\partial \mathbf r}{\partial \mathbf q}
&= 1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \\


\mathbf {dq}
&= \left(\frac{\partial \mathbf r}{\partial \mathbf q} \right)^\top \mathbf {dr}  \\
&=1/\sigma (\mathbf I- \mathbf r \mathbf r^\top / d) \mathbf {dr}  \\
&=1/\sigma  \left( \mathbf {dr}  - (\mathbf r^\top \mathbf {dr})\mathbf r /d   \right)\\
\mathbf {dp} &= \mathbf {dq}, \mathrm{or}, \\
\mathbf {d}p_k& = \sum_{i=1}^d \mathbf {d}q_i \frac{\partial q_i }{\partial p_k} \\
& = \sum_{i=1}^d \mathbf {d}q_i (\mathbf 1_{i=k}-1/d) \\
&=  \mathbf d q_k-1/d \left( \sum_{i=1}^d \mathbf {d}q_i  \right)\\
\mathbf {dp}&=\mathbf {dq}-\bar{\mathbf {dq}},\\
\mathbf {dx}& = \mathbf {dp},\\
\mathbf {dy}& = \mathbf {do} \odot \mathbf o_1, \\

\mathbf {dy} &= g'(\mathbf y) \odot \mathbf {dy}.

\end{aligned}
$$
