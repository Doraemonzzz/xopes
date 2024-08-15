# Naive

前向：
$$
\begin{aligned}
x &=[x_1,\ldots, x_n], \\
o_i & = \log \sum_{j=1}^{i} \exp(x_j).
\end{aligned}
$$
反向：
$$
\begin{aligned}
dx_i
& = \frac{\partial l}{\partial x_i}  \\
& = \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \frac{\partial o_j}{\partial x_i}  \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \left[ \frac{\exp(x_i)}
{\sum_{k=1}^{j} \exp(x_k)} \right] \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \left[ \frac{\exp(x_i)}
{\sum_{k=1}^{j} \exp(x_k)} \right] \\
&=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j)
\end{aligned}
$$


上式前向不稳定，可以改为下式：

前向：
$$
\begin{aligned}
x &=[x_1,\ldots, x_n], \\
m & =\max_{i=1,\ldots, n}(x_i), \\
o_i & = m+ \log \sum_{j=1}^{i} \exp(x_j-m).
\end{aligned}
$$
# 递推式

前向：

注意到：
$$
\begin{aligned}
o_i
&= \log \sum_{j=1}^{i} \exp(x_j) \\
&= \log \sum_{j=1}^{i-1} \exp(x_j) +  \exp(x_i) \\
&= \log \left(\exp(o_{i-1})+  \exp(x_i) \right)
\end{aligned}
$$
注意到上式不稳定，引入：
$$
m_i=\max_{j\le i}\{x_j\}.
$$
那么：
$$
\begin{aligned}
o_i
&= \log \left(\exp(o_{i-1}-m_{i})+  \exp(x_i-m_i) \right) + m_i \\
o_i-m_i &= \log \left(\exp(o_{i-1}-m_{i-1}+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right) \\
\bar o_i &= \log \left(\exp(\bar o_i+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right) \\
o_i & = \bar o_i + m_i
\end{aligned}
$$
整体递推：
$$
\begin{aligned}
m_0 &  =-\infty, \\
\bar o_0 &  =-\infty, \\
m_i & =\max(m_{i-1}, x_i), \\
\bar o_i & =  \log \left(\exp(\bar o_{i-1}+m_{i-1}-m_i) + \exp(x_i -m_i)  \right), \\
o_i &= \bar o_i + m_i ,  \\
i&=1,\ldots, n.
\end{aligned}
$$
反向（数值不稳定）：
$$
\begin{aligned}
\frac{dx_i}{\exp(x_{i})}
& =\sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(- o_j) \\
& =\frac{\partial l}{\partial o_i} \times \exp(- o_i)+ \sum_{j= i+1}^{n}\frac{\partial l}{\partial o_j} \times \exp(- o_j) \\
&= \frac{\partial l}{\partial o_i} \times \exp(- o_i)+\frac{dx_{i+1}}{\exp(x_{i+1})} \\
dx_n &= \frac{\partial l}{\partial o_n} \times \exp(x_n- o_n)    \\
dx_i&= \frac{\partial l}{\partial o_i} \times \exp(x_i- o_i) + \exp(x_i-x_{i+1})dx_{i+1}
\end{aligned}
$$

上式可能数值不稳定，需要测试。





# block递推

前向：

注意下式成立，
$$
\begin{aligned}
\bar o_i &= \log \left(\exp(\bar o_i+m_{i-1}-m_{i})+  \exp(x_i-m_i) \right).
\end{aligned}
$$
故我们维护每个block的最大值：
$$
\begin{aligned}
m&= n/B, \\
M_0&=  -\infty, \\
M_{i}  & =\max\{ M_{i-1}, x_{(i-1)B+k},k=1,\ldots, d\}, \\
\bar O_i &= [\bar o_{(i-1)B+1},\ldots, \bar o_{(i-1)B+B}], \\
X_i&=[x_{(i-1)B+1},\ldots,  x_{(i-1)B+B}],  \\
Y_{i, k}&= \log\left(\sum_{j=1}^k\exp(x_{(i-1)B+j}- M_{i}) \right),   \\
\bar O_i&= \log \left(\exp(\bar O_i+M_{i-1}-M_{i})+  \exp(Y_i) \right),\\
O_i&= \bar O_i + M_i.

\end{aligned}
$$
接下来考虑$Y_{i, k}= \log\left(\sum_{j=1}^k\exp(x_{(i-1)B+j}- M_{i}) \right)$的并行计算，考虑前缀和：
$$
\begin{aligned}
y_i &= \sum_{j\le i} x_j.  \\
\left[
\begin{matrix}
y_1 \\
\vdots  \\
y_n
\end{matrix}
\right]
&=\left[
\begin{matrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0  \\
\vdots &\vdots  &\vdots &0 \\
1 & 1 &1 & 1
\end{matrix}
\right]
\left[
\begin{matrix}
x_1 \\
\vdots  \\
x_n
\end{matrix}
\right]
\end{aligned}
$$
反向：

注意公式：
$$
\begin{aligned}
dx_i &=  \sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j)
\end{aligned}
$$

- 对$n$维度并行；
- 每个block level cumsum；

考虑下式：
$$
y_i=\sum_{j=1}^i x_i.
$$
block循环：
$$
\begin{aligned}
Y_0 &=[0, \ldots, 0], \\
Y_{i}&=Y_{i-1}[-1] + \mathrm{cumsum}(X_i),
\end{aligned}
$$
反向（数值不稳定）：
$$
\begin{aligned}
p_i&=\exp(x_i-o_i), \\
q_{i+1} &=\exp(x_i -x_{i+1}), \\
dx_n &= p_n\frac{\partial l}{\partial o_n} ,  \\
dx_i&= p_i\frac{\partial l}{\partial o_i} \times \exp(x_i- o_i) +  q_{i+1} dx_{i+1}

\end{aligned}
$$
block循环需要类乘，暂时不考虑。



# block并行

前向：
$$
\begin{aligned}
x &=[x_1,\ldots, x_n], \\
\bar o_{i,k} & = \log \sum_{j=iB+1}^{iB+k} \exp(x_j), i=1,\ldots , M, k=1,\ldots, B ,  \\
\Delta_0 &= -\inf,   \\
\Delta_i & =\log(\exp(\Delta_{i-1}) +  \exp(\bar o_{i, B}))   \\
o_{i, k}&= \log\left( \exp(\bar o_{i,k}) + \exp(\Delta_{i}) \right), i=2,\ldots M,k=1,\ldots B.
\end{aligned}
$$
流程：

- 并行计算$\bar o_{i, k}, i=1,\ldots, M$；
  - 数值稳定版本；
- 计算$\Delta_i$；
  - 数值稳定版本；
- 计算$o_{i,k}$；
  - 数值稳定版本；

反向（数值不稳定版本）：

- 计算

进行递推：
$$
\begin{aligned}
p_i&=\exp(x_i-o_i), \\
q_{i+1} &=\exp(x_i -x_{i+1}), \\
dx_n &= p_n\frac{\partial l}{\partial o_n} ,  \\
dx_i&= p_i\frac{\partial l}{\partial o_i} \times \exp(x_i- o_i) +  q_{i+1} dx_{i+1}

\end{aligned}
$$
反向（数值稳定版本）：

- 两两分组，(1, n), (2, n - 1), ...；
- 每一组计算：
  - $\sum_{j= i}^{n}\frac{\partial l}{\partial o_j} \times \exp(x_i- o_j)$
  - 这个分组计算；
