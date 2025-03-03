# Cumsum

我们用一个图解释前向反向。

Reverse = False:

Forward:

```
x1
x1 + x2
...
x1 + x2 + ... + xn-1
x1 + x2 + ... + xn-1 + xn
```

Backward:

```
don + don-1 + ... + do2 + do1
don + don-1 + ... + do2
...
don + don-1
don
```

Reverse = True:

Forward:

```
xn + xn-1 + ... + x2 + x1
xn + xn-1 + ... + x2
...
xn + xn-1
xn
```

Backward:

```
do1
do1 + do2
...
do1 + do2 + ... + don-1
do1 + do2 + ... + don-1 + don
```



## Forward

给定输入$\mathbf x\in \mathbb R^{n}$，reverse flag，计算输出$\mathbf o \in \mathbb R^n$。

如果reverse = False：
$$
o_i =\sum_{j=1}^i x_i.
$$
如果reverse = True：
$$
\begin{aligned}
o_{i} &= \sum_{j=1}^{n-i + 1}x_{n+1-j}

\end{aligned}
$$



## Backward

给定输入$\mathbf {dy}\in \mathbb R^{n}$，reverse flag，计算输出$\mathbf {dx} \in \mathbb R^n$。

如果reverse = False：
$$
\begin{aligned}
dx_i
&=\sum_{j=1}^{n+1-i} d o_j.
\end{aligned}
$$
如果reverse = True：
$$
dx_i=\sum_{j=1}^{i} d o_j.
$$
所以不论是Forward还是Backward，Reverse = True或者False，都可以转换为前缀和的形式。
