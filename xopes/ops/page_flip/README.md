# PageTurner: Page-Flipping with Linear Time Complexity

## 命名

- $\mathbf e_{j}=f(x_j)$：权重；
- $s_t^j$：累积权重；
- $s_{t}^{j-1}/s_t^{j-1}$：相对累积权重；



## $O(n^2)$的rnn

首先回忆online softmax attention:
$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&=s_t^{j-1}+\exp \left(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$
受此启发，定义如下形式的RNN (style 1)：
$$
\begin{aligned}
\mathbf s_{t}^0& =0, \mathbf o_{t}^0 =\mathbf 0,
\mathbf e_j\triangleq \exp(\mathbf y_j), \\
 \mathbf s_t^j&=\mathbf s_t^{j-1}+\mathbf e_j, \\
\mathbf{o}_t^j&=\left(\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{x}_j ,\\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$
注意上式可以化简为：
$$
\begin{aligned}
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
 \mathbf s_t&=\mathbf s_{t-1}+\mathbf e_t, \\
\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
这就是我们常见的linear rnn形式(additive)，注意到上式可以抽象为：
$$
\begin{aligned}
\mathbf o_0&= \mathbf 0, \\
\mathbf o_t &=f(\mathbf o_{t-1}, \mathbf x_t).
\end{aligned}
$$
因为每次只依赖于上一时刻，所以不具有翻书的性质。为了补充翻书的能力，我们考虑如下函数：
$$
\begin{aligned}
\mathbf o_1^0&= \mathbf 0, \\
\mathbf o_t^{j} &=f(\mathbf o_{t}^{j-1}, \mathbf x_j),j=1,\ldots, t,\\
\mathbf o_{t}^0&= \mathbf o_{t-1}^{t-1}.
\end{aligned}
$$
上式表示，**对于每个时刻$t$，都要重新过一遍$\mathbf x_1,\ldots, \mathbf x_t$**，受此启发，定义如下RNN (Style 2)：
$$
\begin{aligned}
\mathbf s_{0}^0& =0, \mathbf o_{0}^0 =\mathbf 0,
\mathbf e_j\triangleq \exp(\mathbf y_j)\\
 \mathbf s_t^j&=\mathbf s_t^{j-1}+\mathbf  e_j, \\
\mathbf{o}_t^j&=\left(\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{x}_j ,\\
\mathbf o_{t}^0&= \mathbf o_{t-1}^{t-1}, \mathbf s_{t}^0 = \mathbf s_{t-1}^{t-1},\\
\mathbf{o}_t&=\mathbf{o}_t^t, \mathbf s_t =\mathbf s_{t}^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$
备注，这里其实有两重递推：

- 第一重是$\mathbf s_t$的递推；
- 第二重是$\mathbf o_t$的递推；



下面来看两个例子。

### 例1(style 1)

当$t=1$时：
$$
\mathbf s_{1}=\mathbf e_1,\\
 \mathbf o_1 = \mathbf x_1.
$$
当$t=2$时：
$$
\mathbf s_{2}=\mathbf s_{1}+\mathbf e_2=\mathbf e_1 + \mathbf e_2,\\

\mathbf o_{2}=\frac{\mathbf e_1}{\mathbf e_1+\mathbf e_2}\mathbf x_{1}+\frac{\mathbf e_2}{\mathbf e_1+\mathbf e_2}\mathbf x_2.
$$
当$t=3$时：
$$
\mathbf s_{3}=\mathbf s_{2}+\mathbf e_2+\mathbf e_3=\mathbf e_1 + \mathbf e_2+\mathbf e_3,\\

\mathbf o_{3}=\frac{\mathbf e_1}{\mathbf e_1+\mathbf e_2+\mathbf e_3}\mathbf x_{1}+\frac{\mathbf e_2}{\mathbf e_1+\mathbf e_2+\mathbf e_3}\mathbf x_2+\frac{\mathbf e_3}{\mathbf e_1+\mathbf e_2+\mathbf e_3}\mathbf x_3.
$$


### 例2(style 2)

记$\mathbf e_j=\exp(\mathbf x_j)$。

当$t=1$时：
$$
\mathbf s_{11}=\mathbf e_1,\mathbf o_{11}=\mathbf x_1,\\
 \mathbf o_1 = \mathbf o_{11}.
$$
当$t=2$时：
$$
\mathbf s_{21}=\mathbf s_{11}+\mathbf e_1=2\mathbf e_1, \mathbf o_{21}=\frac{1}{2}\mathbf x_1+\frac 1 2 {\mathbf x_1}=\mathbf x_1,\\
\mathbf s_{22}=\mathbf s_{21}+\mathbf e_2 =2\mathbf e_1+\mathbf e_2, \mathbf o_{22}=\frac{2\mathbf e_1}{2\mathbf e_1+\mathbf e_2}\mathbf o_{21}+\frac{\mathbf e_2}{2\mathbf e_1+\mathbf e_2}\mathbf x_2=
\frac{2\mathbf e_1}{2\mathbf e_1+e_2}\mathbf x_1+\frac{\mathbf e_2}{2\mathbf e_1+e_2}\mathbf x_2,\\
\mathbf o_2 = \mathbf o_{22}.
$$
当$t=3$时：
$$
\mathbf s_{31}=\mathbf s_{22}+\mathbf e_1=3\mathbf e_1+\mathbf e_2,\mathbf o_{31}=\frac{2\mathbf e_1}{3\mathbf e_1+\mathbf e_2}x_1+\frac{\mathbf e_2}{3\mathbf e_1+\mathbf e_2}\mathbf x_2
,\\
\mathbf s_{32}=\mathbf s_{31}+\mathbf e_2=3\mathbf e_1+2\mathbf e_2,\mathbf o_{32}=\frac{3\mathbf e_1}{3\mathbf e_1+2\mathbf e_2}\mathbf x_1
+\frac{2\mathbf e_2}{3\mathbf e_1+2\mathbf e_2}\mathbf x_2,\\
\mathbf s_{33}=\mathbf s_{32}+\mathbf e_3=3\mathbf e_1+2\mathbf e_2+\mathbf e_3,\mathbf o_{33}=\frac{3\mathbf e_1}{3\mathbf e_1+2\mathbf e_2+\mathbf e_3}x_1+\frac{2\mathbf e_2}{3\mathbf e_1+2\mathbf e_2+\mathbf e_3}\mathbf x_2
+\frac{\mathbf e_3}{3\mathbf e_1+2\mathbf e_2+\mathbf e_3}\mathbf x_3, \\
\mathbf o_3 =\mathbf o_{33}.
$$


### 化简style 2

对公式进行递推可得：
$$
\begin{aligned}
\mathbf{o}_t^j
&=\left(\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{x}_j  \\
&=\left(\mathbf s_t^{j-1} / \mathbf s_t^j\right) \mathbf{o}_t^{j-1}+(\mathbf e_j /\mathbf s_{t}^j) \mathbf{x}_j  \\

&=\left(\mathbf s_t^{j-2} / \mathbf s_t^j\right) \mathbf{o}_t^{j-2}+(\mathbf e_{j-1} /\mathbf s_{t}^j) \mathbf{x}_{j-1}+(\mathbf e_j /\mathbf s_{t}^j) \mathbf{x}_j  \\
&=\ldots \\
&=\left(\mathbf s_t^{0} / \mathbf s_t^j\right) \mathbf{o}_t^{0}
+\sum_{k=1}^j (\mathbf e_{k} /\mathbf s_{t}^j) \mathbf{x}_{k}.  \\
&=\left(\mathbf s_{t-1}^{t-1} / \mathbf s_t^j\right) \mathbf{o}_{t-1}^{t-1}
+\sum_{k=1}^j (\mathbf e_{k} /\mathbf s_{t}^j) \mathbf{x}_{k}.  \\
&=\left(\mathbf s_{t-1}/ \mathbf s_t^j\right) \mathbf{o}_{t-1}
+\sum_{k=1}^j (\mathbf e_{k} /\mathbf s_{t}^j) \mathbf{x}_{k}.  \\
\end{aligned}
$$
取$j=t$可得：
$$
\begin{aligned}
\mathbf{o}_t
&=  \mathbf{o}_t^t \\
&=\left(\mathbf s_{t-1} / \mathbf s_t^t\right) \mathbf{o}_{t-1}
+\sum_{k=1}^j (\mathbf e_{k} /\mathbf s_{t}^t) \mathbf{x}_{k},  \\
&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}
+\sum_{k=1}^t (\mathbf e_{k} /\mathbf s_{t}) \mathbf{x}_{k}  \\

&\triangleq \left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}
+\mathbf p_t. \\
\end{aligned}
$$
另一方面：
$$
\begin{aligned}
\mathbf s_t
&= \mathbf s_{t}^t \\
&= \mathbf s_{t}^{t-1} + \mathbf e_t \\
&\ldots \\
&= \mathbf s_{t}^{0} +\sum_{k=1}^t \mathbf e_k\\
&= \mathbf s_{t-1} +\sum_{k=1}^t \mathbf e_k \\
&\triangleq \mathbf s_{t-1} + \mathbf q_t \\
\mathbf q_t &= \sum_{k=1}^t \mathbf e_k\\
\mathbf p_t &= \sum_{k=1}^t (\mathbf e_{k} /\mathbf s_{t}) \mathbf{x}_{k} \\
&=(\mathbf s_t/\mathbf s_{t-1})\sum_{k=1}^{t-1} (\mathbf e_{k} /\mathbf s_{t-1}) \mathbf{x}_{k} +  (\mathbf e_{t} /\mathbf s_{t}) \mathbf{x}_{t} \\
&= (\mathbf s_t/\mathbf s_{t-1}) \mathbf p_{t-1}+  (\mathbf e_{t} /\mathbf s_{t}) \mathbf{x}_{t}\\
&= (\mathbf s_t/\mathbf s_{t-1}) \mathbf p_{t-1}+  (1-\mathbf s_{t-1} /\mathbf s_{t}) \mathbf{x}_{t}.
\end{aligned}
$$
于是我们得到如下递推式：
$$
\begin{aligned}
\mathbf e_{t} &=\exp(\mathbf y_t), \\
\mathbf q_{t} &=\mathbf q_{t-1}+\mathbf e_{t}, \\
\mathbf s_{t} &=\mathbf s_{t-1} + \mathbf  q_{t}, \\
\mathbf p_{t}&=(\mathbf s_{t-1} / \mathbf s_t)\mathbf p_{t-1} +  (1-\mathbf s_{t-1} /\mathbf s_{t}) \mathbf{x}_{t},   \\
\mathbf o_{t}&=(\mathbf s_{t-1} / \mathbf s_t) \mathbf o_{t-1} + \mathbf p_t.
\end{aligned}
$$


### 扩展为multiply decay

另一种形式的递推为multiply decay，即：
$$
\log \mathbf s_t=  \log  \mathbf s_{t-1} +\mathbf e_t,
\mathbf s_{t-1}/\mathbf s_{t}= \exp(-\mathbf e_t).
$$
此时：
$$
\begin{aligned}
\log\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
\log \mathbf s_t&=  \log  \mathbf s_{t-1} +\mathbf e_t, \\
\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
代入可得：
$$
\begin{aligned}
\mathbf{o}_t&=\exp(-\mathbf e_t) \mathbf{o}_{t-1}+\left(1-\exp(-\mathbf e_t) \right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
翻书公式可以推广为：
$$
\begin{aligned}
\mathbf q_{t} &= \mathbf q_{t-1}+\mathbf e_t, \\
 \log \mathbf s_{t} &= \log \mathbf s_{t-1} +   \mathbf q_{t}, \\
\mathbf p_{t}&=(\mathbf s_{t-1} / \mathbf s_t)\mathbf p_{t-1} +  (1-\mathbf s_{t-1} /\mathbf s_{t}) \mathbf{x}_{t},   \\
\mathbf o_{t}&=(\mathbf s_{t-1} / \mathbf s_t) \mathbf o_{t-1} + \mathbf p_t.
\end{aligned}
$$


#### 验证

将公式$\mathbf s_t$的定义代入公式可得：
$$
\begin{aligned}
\log \mathbf s_t
&= \log \mathbf s_{t}^t \\
&= \log \mathbf s_{t}^{t-1} + \mathbf e_t \\
&\ldots \\
&= \log \mathbf s_{t}^{0} +\sum_{k=1}^t \mathbf e_k\\
&= \log \mathbf s_{t-1} +\sum_{k=1}^t \mathbf e_k \\
&\triangleq \log \mathbf s_{t-1} + \mathbf q_t \\
\mathbf q_t &= \sum_{k=1}^t \mathbf e_k\\
\mathbf p_t &= \sum_{k=1}^t (\mathbf e_{k} /\mathbf s_{t}) \mathbf{x}_{k} \\
&=(\mathbf s_t/\mathbf s_{t-1})\sum_{k=1}^{t-1} (\mathbf e_{k} /\mathbf s_{t-1}) \mathbf{x}_{k} +  (\mathbf e_{t} /\mathbf s_{t}) \mathbf{x}_{t} \\
&= (\mathbf s_t/\mathbf s_{t-1}) \mathbf p_{t-1}+  (\mathbf e_{t} /\mathbf s_{t}) \mathbf{x}_{t}\\
&= (\mathbf s_t/\mathbf s_{t-1}) \mathbf p_{t-1}+  (1-\mathbf s_{t-1} /\mathbf s_{t}) \mathbf{x}_{t}.
\end{aligned}
$$


### 拓展为更一般的形式(linear rnn)

注意到$ (1-\mathbf s_{t-1} /\mathbf s_{t})$这一项可以理解为input gate，所以将上述形式进一步扩展：

#### wo flip


$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\log \mathbf s_t&=  \log  \mathbf s_{t-1} +   \mathbf e_t,\\
\textbf{additive decay}:
 \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf s_t&=  \mathbf s_{t-1} +  \mathbf e_t,\\
\textbf{compute}:
\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\mathbf g_t \mathbf{x}_t .
\end{aligned}
$$


#### w flip

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0,\\
\mathbf q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
  \log \mathbf s_{t} &= \log \mathbf s_{t-1} +   \mathbf q_{t}, \\

\textbf{additive decay}:
\mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf  q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
\mathbf  s_{t} &=\mathbf s_{t-1} +\mathbf q_{t}, \\
\textbf{compute}:\mathbf p_{t}&=(\mathbf s_{t-1} / \mathbf s_t)\mathbf p_{t-1} +  \mathbf g_t \mathbf{x}_{t},   \\
\mathbf o_{t}&=(\mathbf s_{t-1} / \mathbf s_t) \mathbf o_{t-1} + \mathbf p_t.
\end{aligned}
$$



### 拓展为更一般的形式(linear attn)

我们考虑外积的形式：

#### wo flip

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\log \mathbf s_t&=  \log  \mathbf s_{t-1} +   \mathbf e_t,\\
\textbf{additive decay}:
 \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf s_t&=  \mathbf s_{t-1} +  \mathbf e_t,\\
\textbf{compute}:
\mathbf{o}_t&=\mathrm{diag}\{\left(\mathbf s_{t-1} / \mathbf s_t\right)\}  \mathbf{o}_{t-1}+\mathbf g_t \mathbf{x}_t^\top .
\end{aligned}
$$



#### w flip

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
  \log \mathbf s_{t} &= \log \mathbf s_{t-1} +  \mathbf q_{t}, \\

\textbf{additive decay}:
\mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf  q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
\mathbf  s_{t} &=\mathbf s_{t-1} +\mathbf q_{t}, \\
\textbf{compute}:\mathbf p_{t}&=\mathrm{diag}\{\left(\mathbf s_{t-1} / \mathbf s_t\right)\} \mathbf p_{t-1} +  \mathbf g_t \mathbf{x}_{t}^\top ,   \\
\mathbf o_{t}&=\mathrm{diag}\{\left(\mathbf s_{t-1} / \mathbf s_t\right)\} \mathbf o_{t-1} + \mathbf p_t.
\end{aligned}
$$



### 讨论：是否需要引入$\mathbf q_t$

#### 想法1：需要

引入后可以将递推式进行进一步扩展，例如：

multiply decay:
$$
\begin{aligned}
\mathbf  q_{t} &=\lambda\mathbf q_{t-1}+\mathbf e_t, \\
\log\mathbf  s_{t} &=\lambda \log\mathbf s_{t-1} +\mathbf q_{t}. \\
\end{aligned}
$$
additive decay:
$$
\begin{aligned}
\mathbf  q_{t} &=\lambda \mathbf q_{t-1}+\mathbf e_t, \\
\mathbf  s_{t} &=\lambda \mathbf s_{t-1} +\mathbf q_{t}. \\
\end{aligned}
$$


#### 想法2：不需要

直接将计算式合并可得：

multiply decay:
$$
\log \mathbf s_t =\sum_{k=1}^t (t+1-k)  \mathbf e_k.
$$
additive decay:
$$
\mathbf s_t =\sum_{k=1}^t (t+1-k) \mathbf e_k.
$$


### 讨论：$\mathbf s_t$的影响

#### wo flip

将公式展开不难得到：
$$
\begin{aligned}
\mathbf{o}_t
&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\mathbf g_t \mathbf{x}_t \\
&=\left(\mathbf s_{t-1} / \mathbf s_t\right) (\left(\mathbf s_{t-2} / \mathbf s_{t-1}\right) \mathbf{o}_{t-2}+\mathbf g_{t-1} \mathbf{x}_{t-1})+\mathbf g_t \mathbf{x}_t \\
&=\mathbf s_{t-2} /\mathbf s_t \mathbf{o}_{t-2}+\mathbf s_{t-1} / \mathbf s_t \mathbf g_{t-1}\mathbf x_{t-1}+\mathbf g_t \mathbf{x}_t  \\
&=\ldots \\
&=\sum_{k=1}^t(\mathbf s_k/ \mathbf s_t)\mathbf g_k \mathbf x_k.


\end{aligned}
$$


##### multiply

$$
\begin{aligned}
\mathbf s_t
&=\exp \left(\sum_{j=1}^t\mathbf e_j \right), \\
\mathbf s_k/\mathbf s_t
&=\exp \left(-\sum_{j=k+1}^t\mathbf e_j \right).
\end{aligned}
$$



##### additive

$$
\begin{aligned}
\mathbf s_t
&=\sum_{j=1}^t\mathbf e_j , \\
\mathbf s_k/\mathbf s_t
&=\left( \sum_{j=1}^k\mathbf e_j\right) /\left( \sum_{j=1}^t\mathbf e_j\right).
\end{aligned}
$$



#### w flip

将公式展开不难得到：
$$
\begin{aligned}
\mathbf p_{t}&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf p_{t-1} +  \mathbf g_t \mathbf{x}_{t},   \\
&=\sum_{k=1}^t(\mathbf s_k/ \mathbf s_t)\mathbf g_t \mathbf x_t,  \\
\mathbf o_{t}&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf o_{t-1} + \mathbf p_t\\
&=\sum_{k=1}^t(\mathbf s_k/ \mathbf s_t) \mathbf p_k \\
&=\sum_{k=1}^t(\mathbf s_k/ \mathbf s_t) \sum_{j=1}^k(\mathbf s_j/ \mathbf s_k)\mathbf g_j \mathbf x_j\\
&= \sum_{k=1}^t \sum_{j=1}^k(\mathbf s_j/ \mathbf s_t)\mathbf g_j \mathbf x_j \\
&= \sum_{j=1}^t \sum_{k=j}^t(\mathbf s_j/ \mathbf s_t)\mathbf g_j \mathbf x_j \\
&= \sum_{j=1}^t (t-j+1)(\mathbf s_j/ \mathbf s_t)\mathbf g_j \mathbf x_j.
\end{aligned}
$$



## 定义$k$阶RNN

在后续讨论之前，我们定义$k$阶RNN。

0阶，记：
$$
\mathbf h_t^0 =f^0(\mathbf x_t).
$$
1阶：
$$
\begin{aligned}
\mathbf h_t^0 &=f^0(\mathbf x_t), \\
\mathbf h_t^1 &= f^1(\mathbf h_{t-1}^1, \mathbf h_t^0).

\end{aligned}
$$
2阶：
$$
\begin{aligned}
\mathbf h_t^0 &=f^0(\mathbf x_t), \\
\mathbf h_t^1 &= f^1(\mathbf h_{t-1}^1, \mathbf h_t^0),\\
\mathbf h_t^2 &= f^2(\mathbf h_{t-1}^2,\mathbf h_t^1, \mathbf h_t^0).
\end{aligned}
$$
$k$阶：
$$
\begin{aligned}
\mathbf h_t^0 &=f^0(\mathbf x_t), \\
\mathbf h_t^1 &= f^1(\mathbf h_{t-1}^1, \mathbf h_t^0),\\
\ldots, \\
\mathbf h_t^k &= f^k(\mathbf h_{t-1}^k ,\mathbf h_{t}^{k-1} ,\ldots , \mathbf h_t^0).
\end{aligned}
$$
注意这里的阶数是最简形式的阶数，例如multiply decay是1阶RNN，因为$\mathbf s_{t-1}/\mathbf s_t=\exp(- e_t)$。

可以看出：

- Softmax attention是2阶RNN；
- Linear RNN wo flip；
  - additive是2阶RNN；
  - multiply是1阶RNN；
- Linear RNN w flip；
  - additive是4阶RNN；
  - multiply是3阶RNN；

在后续讨论中，我们将用第$j$次递推表示映射关系$\mathbf h_t^j =  f^{j-1}(\mathbf h_t^{j-1},\ldots, \mathbf h_t^{1},\mathbf x_t)$。

例如，对于：
$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&=s_t^{j-1}+\exp \left(\mathbf{q}_t \mathbf{k}_j^T / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$
我们有：

对于每个$s$：
$$
\begin{aligned}
\mathbf h_t^0 &=f^0(\mathbf x_t)=\exp \left(\mathbf{q}_s \mathbf{k}_t^T / \sqrt{d}\right), \\
\mathbf h_t^1 &= f^1(\mathbf h_{t-1}^1, \mathbf h_t^0)=\mathbf h_{t-1}^1+\mathbf h_t^0 ,\\
\mathbf h_t^2 &= f^2(\mathbf h_{t-1}^2,\mathbf h_t^1, \mathbf h_t^0)=
\left(\mathbf h_{t-1}^1 / \mathbf h_{t}^1\right) \mathbf h_{t-1}^2+\left(1-\mathbf h_{t-1}^1 / \mathbf h_{t}^1\right) \mathbf{v}_j.
\end{aligned}
$$
输出为：
$$
\mathbf o_s = \mathbf h_s^2.
$$
注意上述递推式应该进行化简，我们来看之前的例子。



### 例子

对于multply decay，因为
$$
\mathbf s_{t-1} / \mathbf s_t=\exp(-\mathbf e_t).
$$
所以递推式实际上为：
$$
\mathbf{o}_t=\exp(-\mathbf e_t) \mathbf{o}_{t-1}+\mathbf g_t \mathbf{x}_t
$$
这说明multiply decay实际上是1阶递推，而additive decay是2阶递推。对于flip的版本，multiply decay实际上是2阶递推，additive decay是3阶递推。



#### wo flip


$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\log \mathbf s_t&=  \log  \mathbf s_{t-1} +   \mathbf e_t,\\
\textbf{additive decay}:
 \mathbf s_{0}& =\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf s_t&=  \mathbf s_{t-1} +  \mathbf e_t,\\
\textbf{compute}:
\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\mathbf g_t \mathbf{x}_t .
\end{aligned}
$$



#### w flip

$$
\begin{aligned}
\textbf{multiply decay}:
\log \mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0,\\
\mathbf q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
  \log \mathbf s_{t} &= \log \mathbf s_{t-1} +   \mathbf q_{t}, \\

\textbf{additive decay}:
\mathbf s_{0}& =\mathbf 0,\mathbf q_{0}=\mathbf 0,\mathbf o_{0} =\mathbf 0, \\
\mathbf  q_{t} &=\mathbf q_{t-1}+\mathbf e_t, \\
\mathbf  s_{t} &=\mathbf s_{t-1} +\mathbf q_{t}, \\
\textbf{compute}:\mathbf p_{t}&=(\mathbf s_{t-1} / \mathbf s_t)\mathbf p_{t-1} +  \mathbf g_t \mathbf{x}_{t},   \\
\mathbf o_{t}&=(\mathbf s_{t-1} / \mathbf s_t) \mathbf o_{t-1} + \mathbf p_t.
\end{aligned}
$$



## 讨论新的attention

这一节我们讨论一些新的attention和我们之前讨论内容的关系。



### stick-breaking attention (使用multply decay)

注意下式为multiply版本的2次attention：
$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&=s_t^{j-1}+\exp \left(\mathbf{q}_t \mathbf{k}_j^T / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$
相应的也有additive版本的2次attention (stick-breaking attention)：
$$
\begin{aligned}
\log s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
\log  s_t^j&=\log s_t^{j-1}+\exp \left(\mathbf{q}_t \mathbf{k}_j^\top\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$

注意stick-breaking attention中没有使用$\exp(x)$，使用的是$\log(1+\exp(x))$。

总结：（需要修改）

$f^1$修改为

- $\mathbf h_t^1 = f^1(\mathbf h_{t-1}^1, \mathbf h_t^0)=\mathbf h_{t-1}^1\exp(\mathbf h_t^0)$；
  - 也等价于：$\log \mathbf h_t^1=\log \mathbf h_{t-1}^1+\exp(\mathbf q_s \mathbf k_t^\top)$；



### alibi

alibi将递推改写为：
$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&=s_t^{j-1}+\lambda^{t-j}\exp \left(\mathbf{q}_t \mathbf{k}_j^T / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$



### forgeting transformer

$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&= s_t^{j-1}+(f_j/f_t)\exp \left(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$



### Selective Attention Improves Transformer

$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&= s_t^{j-1}+\mathrm{sigmoid}(\mathbf a_t^\top \mathbf b_j)\exp \left(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$



### Gated ABC

$$
\begin{aligned}
\mathbf a_t &= \alpha_t \mathbf a_{t-1}+(1-\alpha_t)\mathbf k_t^\top, \\
\mathbf b_t&= \beta_t \mathbf b_{t-1}+(1-\beta_t)\mathbf v_t^\top,\\
\mathbf o_t &= \mathrm{Softmax}(\mathbf q_t \mathbf a_t^\top)\mathbf b_t.
\end{aligned}
$$

所以实际上gated abc是1阶rnn。



### RoPE

$$
\begin{aligned}
s_{t}^0& = 0, \mathbf o_{t}^0 =\mathbf 0,\\
 s_t^j&=s_t^{j-1}+\exp \left(\mathbf{q}_t \exp(i(t-j)\theta)\mathbf{k}_j^T / \sqrt{d}\right), \\
\mathbf{o}_t^j&=\left(s_t^{j-1} / s_t^j\right) \mathbf{o}_t^{j-1}+\left(1-s_t^{j-1} / s_t^j\right) \mathbf{v}_j, \\
\mathbf{o}_t&=\mathbf{o}_t^t,\\
t&=1,\ldots, n, \\
j&=1, \ldots, t.
\end{aligned}
$$



### 一些想法

结合之前的讨论，我们知道被修改的部分通常为第一次递推，即score function的递推（我们记$e_{tj}\triangleq f \left(\mathbf{q}_t, \mathbf{k}_j\right)\ge 0$）：

baseline:

2次：
$$
s_{t}^j =s_{t}^{j-1}+ e_{tj}.
$$
1次：
$$
s_{t}^j =s_{t}^{j-1}+ e_{j}.
$$
multiply decay:

2次+2次：
$$
s_{t}^j =s_{t}^{j-1}+f_{tj} e_{tj}.
$$
2次+1次：
$$
s_{t}^j =s_{t}^{j-1}+f_j/f_t  e_{tj}.
$$
1次+1次：
$$
s_{t}^j = s_{t}^{j-1}+ f_j e_{j}.
$$
additive decay:

2次：
$$
\log s_{t}^j =\log s_{t}^{j-1}+ e_{tj}.
$$
2次+2次：(additive + additive)
$$
\log s_{t}^j =\log s_{t}^{j-1}+ f_{t,j}e_{tj}.
$$
2次+1次：(additive + additive)
$$
\log s_{t}^j = \log s_{t}^{j-1}+f_j e_{tj}.
$$
1次：
$$
\log s_{j} =\log s_{j-1}+ e_{j}.
$$
1次+1次：(additive + multiply)
$$
\log s_{j} = \log s_{j-1}+ f_je_{j}.
$$


## 权重映射的选择

我们需要选择$f:\mathbb R\to \mathbb R^+$，并且是单调映射。

可以的选择：

- $f(x)=\exp(x)$；
- $f(x)=\log(1+\exp(x))$；
- $f(x)=\mathrm{Relu}(x)$；
- $f(x)=\log(\cos(\pi /2 \times \mathrm{sigmoid}(x)))$；



## 复数如何处理

首先回到最简单的RNN：
$$
\begin{aligned}
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
 \mathbf s_t&=\mathbf s_{t-1}+\mathbf e_t, \\
\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
在传统的RNN中引入复数的形式如下：
$$
\begin{aligned}
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
 \mathbf s_t&=\mathbf s_{t-1}+\mathbf e_t, \\
\mathbf{o}_t&=\exp(i\theta)\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
我们可以对此推广：

$\theta$的multiply decay：
$$
\begin{aligned}
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
\log \theta_t&= \log \theta_{t-1}+\Delta_t, \\
 \mathbf s_t&=\mathbf s_{t-1}+\mathbf e_t, \\
\mathbf{o}_t&=\exp(i(\theta_{t-1}/\theta_t))\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
$\theta$的additive decay：
$$
\begin{aligned}
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
\theta_t&= \theta_{t-1}+\Delta_t, \\
 \mathbf s_t&=\mathbf s_{t-1}+\mathbf e_t, \\
\mathbf{o}_t&=\exp(i(\theta_{t-1}/\theta_t))\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
