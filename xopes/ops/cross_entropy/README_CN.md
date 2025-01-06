# Cross Entropy

对于输入$\mathbf z\in \mathbb R^v$，以及one hot label $\mathbf y\in \mathbb R^v,y_k =1$，smooth参数$\lambda\in[0, 1]$，其中：
$$
\begin{aligned}
\mathbf {\bar y} &= (1-\lambda) \mathbf y+\lambda /v  \mathbf 1,\\
\mathbf 1^\top \mathbf {\bar y}&=(1-\lambda)  + \lambda=1.
\end{aligned}
$$
输出：
$$
\begin{aligned}
r&=\log\left(
\sum_{j=1}^v \exp(z_j)
\right)   \\


o &= - \sum_{i=1}^v \bar y_i \left(z_i -r \right) \\
&= - \sum_{i=1}^v ( (1-\lambda) y_i+\lambda /v  ) \left(z_i - r\right) \\
&= - \sum_{i=1}^v (1-\lambda) y_i\left(z_i - r\right)
-\lambda/v  \sum_{i=1}^e \left(z_i -r\right)
\\

&=- (1-\lambda) \left(z_k- r\right) -\lambda/v \left( \sum_{i=1}^v z_i -v r \right)\\
&= - (1-\lambda)z_k + (1-\lambda) r - \lambda/v \left( \sum_{i=1}^v z_i\right) + \lambda r \\
&= - (1-\lambda)z_k + r - \lambda/v \left( \sum_{i=1}^v z_i\right).
\end{aligned}
$$


## Forward

输入：$\mathbf z\in \mathbb R^v$，以及one hot label $\mathbf y\in \mathbb R^v,y_k =1$，smooth参数$\lambda\in[0, 1]$，ignore index $ig$。

计算：
$$
\begin{aligned}
r&= \log\sum_{j=1}^v \exp(z_j),\\
n&= \mathbf 1_{z_k \neq i_g}, \\
s&= \sum_{j=1}^v z_j,\\
o&= -(1-\lambda) z_k +r -\lambda /vs.
\end{aligned}
$$
其中$c$是在多个样本时进行reduce所需要的量（例如mean-reduce时候返回$o/n$）。



## Backward

输入：$\mathbf {do}\in \mathbb R$。

计算：
$$
\begin{aligned}
p_{k}&= \exp(z_k - r) , \\
\frac{\partial o}{\partial z_i}
&= -(1-\lambda)\frac{\partial z_k}{\partial z_i} + \frac{\partial r}{\partial z_i}-
\lambda/ v \frac{\partial \left( \sum_{i=1}^v z_i \right)}{\partial z_i}\\
&=  -(1-\lambda) 1_{i=k}+p_i -\lambda /v, \\
\frac{\partial \mathbf o}{\partial \mathbf  z}& =  -(1-\lambda) \mathbf y + \mathbf p - \lambda/ v,     \\
\mathbf {dz}&= {\mathbf {do}}\frac{\partial \mathbf o}{\partial \mathbf  z}\in \mathbb R^{v}.
\end{aligned}
$$

推广到带批量的版本，计算公式为：
$$
\begin{aligned}
\frac{\partial \mathbf o}{\partial \mathbf  Z}& =  -(1-\lambda) \mathbf Y + \mathbf P - \lambda/ v\in \mathbb R^{b\times v}, \\
\mathbf{dZ}&= \mathbf{dO} \odot \frac{\partial \mathbf o}{\partial \mathbf  Z}.

\end{aligned}
$$
根据上式，我们可以在前向中直接计算出$\frac{\partial \mathbf o}{\partial \mathbf  Z}$，并缓存下来即可，然后在Backward时，根据输入$\mathbf {dO}$计算元素乘法$\mathbf{dO} \odot \frac{\partial \mathbf o}{\partial \mathbf  Z}$即可。
