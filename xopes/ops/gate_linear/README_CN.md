# Gate Linear

## 前向传播

输入$\mathbf x_1,\mathbf x_2 \in \mathbb R^{n\times d_1}, \mathbf W\in \mathbb R^{d_1\times d_2}$，残差$\mathbf r\in \mathbb R^{n\times d_2}$，激活函数$f$，计算：
$$
\mathbf o = [f(\mathbf x_1)\odot \mathbf x_2] \mathbf W + \mathbf r.
$$
前向传播缓存$\mathbf x_1, \mathbf x_2, \mathbf W$。



## 反向传播

输入$\mathbf {do}\in \mathbb R^{n\times r_2}$，计算：
$$
\begin{aligned}
\mathbf {dr}&= \mathbf {do}, \\
\mathbf {y} &= f(\mathbf x_1)\odot \mathbf x_2, \\
\mathbf {dW}&= \mathbf {y}^\top \mathbf W, \\
\mathbf {dy} &= \mathbf {do}\mathbf W^\top, \\
\mathbf {dx}_2 & = f(\mathbf {x}_1)  \odot \mathbf {dy}, \\
\mathbf {dx}_1 & = f(\mathbf {x}_1)'  \odot \mathbf {x}_2  \odot \mathbf {dy}.

\end{aligned}
$$
