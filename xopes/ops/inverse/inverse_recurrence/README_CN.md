# Mesa Inverse Recurrence

考虑如下递归：
$$
\begin{aligned}
\mathbf s_0 & = \lambda_0 \mathbf I_d, \\
\mathbf s_t^{-1} &= \lambda_t  \mathbf s_{t-1}^{-1} + \mathbf u_t \mathbf u_t^\top, \\
\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{d}.
\end{aligned}
$$

根据Sherman-Morrison公式，我们有：
$$
\begin{aligned}
\mathbf s_t

&= ( \mathbf s_{t-1}^{-1} + \mathbf u_t \mathbf u_t^\top)^{-1} \\

&=  \lambda_t^{-1} \mathbf s_{t-1} -\frac{ \lambda_t^{-1} \mathbf s_{t-1} \mathbf u_t \mathbf u_t^\top  \lambda_t^{-1} \mathbf s_{t-1}}{1+\mathbf u_t^\top \lambda_t^{-1} \mathbf s_{t-1} \mathbf u_t} \\

&= \lambda_t^{-1}
\left( \mathbf s_{t-1} - \frac{ \mathbf s_{t-1} \mathbf u_t \mathbf u_t^\top  \mathbf s_{t-1}}{\lambda_t+\mathbf u_t^\top \mathbf s_{t-1} \mathbf u_t} \right) .
\end{aligned}
$$

另一方面，对原始公式递推可得：
$$
\begin{aligned}
\ \alpha_t &= \prod_{i=0}^t \lambda_i, \\

\mathbf s_t &= \left( \alpha_t \mathbf I_d + \sum_{i=1}^t (\alpha_t / \alpha_i \mathbf u_i) (\alpha_t / \alpha_i \mathbf u_i)^\top \right)^{-1} \\

&\triangleq \left(
\alpha_t \mathbf I_d + \mathbf {\tilde U}_t^\top \mathbf {\tilde U}_t

\right)^{-1}
\end{aligned}
$$

注意到：
$$
\begin{aligned}
\mathbf X^{-1} &= (\mathbf I - (\mathbf I - \mathbf X))^{-1} \\
&= \sum_{i=0}^\infty (\mathbf I - \mathbf X)^i,  \\

\mathbf X_k^{-1} &=  \mathbf X_{k-1}^{-1}(\mathbf I - \mathbf X) + \mathbf I, \\

\mathbf y^{\top} \mathbf X^{-1}

&\approx \mathbf y^{\top}\mathbf X_k^{-1}  \\

&\triangleq \mathbf y_k \\
&=  \mathbf y^{\top} \mathbf X_{k-1}^{-1}(\mathbf I - \mathbf X)+ \mathbf y^\top \\
&=  \mathbf y_{k-1}^\top (\mathbf I - \mathbf X) + \mathbf y^\top \\
&=  \mathbf y_{k-1}^\top - \mathbf y_{k-1}^\top\mathbf X+ \mathbf y^\top \\
\end{aligned}
$$

将之前的公式代入可得：
$$
\begin{aligned}
[\mathbf o_t^{(i)}]^\top
&= [\mathbf o_t^{(i-1)}]^\top -  [\mathbf o_t^{(i-1)}]^\top
\left(

\alpha_t \mathbf I_d + \mathbf {\tilde U}_t^\top \mathbf {\tilde U}_t
\right) + \mathbf q_t^\top \\

&= \mathbf q_t^\top + [\mathbf o_t^{(i-1)}]^\top (1-\alpha_t \mathbf I_d)
- [\mathbf o_t^{(i-1)}]^\top \mathbf {\tilde U}_t^\top \mathbf {\tilde U}_t.

\end{aligned}
$$
定义Linear Attention算子 $f$:
$$
\begin{aligned}
\mathbf s_0 & = \lambda_0 \mathbf I_d, \\
\mathbf s_t &= \lambda_t  \mathbf s_{t-1} + \mathbf u_t \mathbf u_t^\top, \\
\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{d}.
\end{aligned}
$$
那么上述迭代公式可以表示为：
$$
\begin{aligned}
\mathbf O^{(i)} &= \mathbf Q^{(i-1)} + \mathbf O^{(i-1)}(1 - \Alpha^{(i-1)})
+ f(\mathbf O^{(i-1)}, \mathbf K^{(i-1)}, \mathbf K^{(i-1)}, \Lambda^{(i-1)}, \mathrm{false}), \\
\mathbf Q^{(i-1)} & =\mathbf Q, \\
\mathbf K^{(i-1)} & =\mathbf K, \\
\mathbf \Lambda^{(i-1)} & =\mathbf \Lambda, \\
\Alpha^{(i-1)} &= \Alpha, \\
i&=1,\ldots, k. \\

\mathbf O&= \mathbf O^{(k)}.
\end{aligned}
$$

对于反向传播，计算公式为：
$$
\begin{aligned}
\mathbf{dO} ^{(i-1)}

&= \mathbf{dO} ^{(i)} (1-\Alpha_t^{(i-1)}) + f (\mathbf{dO}^{(i)}, \mathbf{K}^{(i-1)}, \mathbf{K}^{(i-1)}, \mathbf{\Lambda}^{(i-1)}, \mathrm{false}), \\

\mathbf{dQ} ^{(i-1)} &= \mathbf{dO}^{(i)}, \\

\mathbf{dK} ^{(i-1)} &= f(\mathbf K^{(i-1)}, \mathbf{dO}^{(i)}, \mathbf{O}^{(i-1)}, \mathbf{\Lambda}^{(i-1)}, \mathrm{true}) +
f(\mathbf K^{(i-1)}, \mathbf{O}^{(i-1)},\mathbf{dO}^{(i)},  \mathbf{\Lambda}^{(i-1)}, \mathrm{true}), \\

\mathbf{d\log A}^{(i-1)} &= -\mathbf{dO}^{(i)} \odot \mathbf {O}^{(i-1)} \odot \mathbf A^{(i-1)}+
 \mathbf {O}^{(i-1)} \odot \mathbf{dQ} ^{(i-1)} - \mathbf K^{(i-1)} \odot \mathbf{dK} ^{(i-1)}.

\end{aligned}

$$
最终：
$$
\begin{aligned}
\mathbf{dQ} &= \sum_{i=1}^k \mathbf{dQ} ^{(i-1)}, \\
\mathbf{dK} &= \sum_{i=1}^k \mathbf{dK} ^{(i-1)}, \\
\mathbf{d\log A} &= \sum_{i=1}^k \mathbf{d\log A} ^{(i-1)}, \\
\mathbf{dO} &= \sum_{i=1}^k \mathbf{dO} ^{(i-1)}, \\
\mathbf{d\log \Lambda} &= \mathrm{revcumsum}(\mathbf{d\log A}), \\
\end{aligned}
$$

简化版本，注意到梯度的计算涉及到$\mathbf O^{(i-1)}$，因此需要缓存这些值，但是完全换成会造成大量的activation，所以一个近似的方案是用$\mathbf O$来代替$\mathbf O^{(i-1)}$。
