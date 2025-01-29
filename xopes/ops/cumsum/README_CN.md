# Cumsum

## Sequential算法

### Forward

给定输入$\mathbf x\in \mathbb R^{n}$，reverse flag，计算输出$\mathbf y \in \mathbb R^n$，我们记：
$$
\begin{aligned}
\bar x_i & \triangleq x_{n+1-i}, \\
\bar y_i  &  \triangleq y_{n+1-i}.
\end{aligned}
$$
如果reverse = False：
$$
y_i =\sum_{j=1}^i x_i.
$$
如果reverse = True：
$$
\begin{aligned}
y_{n+1-i} &= \sum_{j=1}^{i}x_{n+1-j}  \\
\bar y_i&= \sum_{j=1}^{i} \bar x_{j}.

\end{aligned}
$$


### Backward

给定输入$\mathbf {dy}\in \mathbb R^{n}$，reverse flag，计算输出$\mathbf {dx} \in \mathbb R^n$，我们记：
$$
d\bar y_i \triangleq dy_{n+1-i}.
$$
如果reverse = False：
$$
\begin{aligned}
dx_i
&=\sum_{j=i}^n d y_j \\
&= \sum_{j=i}^n d y_{n+1-j} \\
&= \sum_{j=1}^{n-i+1} d \bar y_{j}.
\end{aligned}
$$
如果reverse = True：
$$
dx_i=\sum_{j=1}^{n-i+1} d y_j.
$$
所以不论是Forward还是Backward，Reverse = True或者False，都可以转换为前缀和的形式。





## Chunk算法

### Forward

注意到：

如果reverse = False：
$$
\begin{aligned}
y_{s+t}
&=\sum_{j=1}^{s+t} x_j \\
&= \sum_{j=1}^{s} x_j + \sum_{j=s+1}^{s+t} x_{j} \\
&= \sum_{j=1}^{s} x_j + \sum_{j=1}^{t} x_{s+j} \\
&= y_s + \sum_{j=1}^{t} x_{s+j}.
\end{aligned}
$$
如果reverse = True：
$$
\begin{aligned}

y_{n+1-s-t}
&= \bar y_{s+t}   \\
&=\sum_{j=1}^{s+t} \bar  x_j \\
&= \sum_{j=1}^{s}  \bar x_j + \sum_{j=s+1}^{s+t}  \bar x_{j} \\
&= \sum_{j=1}^{s}  \bar x_i + \sum_{j=1}^{t} \bar  x_{s+j} \\
&=\bar y_{s} + \sum_{j=1}^{t} x_{n+1-s-j} \\
&= y_{n+1-s} + \sum_{j=1}^{t} x_{n+1-s-j} .
\end{aligned}
$$
假设按chunk size $c$进行切分：
$$
\begin{aligned}
\mathbf x &= [\mathbf x_1,\ldots, \mathbf x_{n/c}], \\
\mathbf y  &= [\mathbf y_1,\ldots, \mathbf y_{n/c}], \\
 \mathbf y_i & = f(\mathbf x_i).
\end{aligned}
$$
其中$f$表示`cumsum`。根据上述讨论，我们有：

如果reverse = False：
$$
\begin{aligned}
y_{(i-1)c+j}
&=y_{(i-1)c} + [ \mathbf y_i]_j \\
y_{ic} &=y_{(i-1)c+c}.
\end{aligned}
$$
如果reverse = True：
$$
\begin{aligned}
y_{(i-1)c+j}
&=y_{ic+1} + [ \mathbf y_i]_j \\
y_{(i-1)c+1} &=y_{(i-1)c+1}.
\end{aligned}
$$




### Backward

基本同前向的思路。



## 补充

注意上述算法可以chunk先并行，然后reduce计算，注意此时会增加一些io，但是并行度更高。另一方面，局部的`cumsum`可以使用tl.cumsum或者矩阵乘法完成。
