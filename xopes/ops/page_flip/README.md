# PageTurner: Page-Flipping with Linear Time Complexity

## $O(n^2)$的rnn

首先回忆online softmax attention:
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
\mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0,\mathbf e_j\triangleq \exp(\mathbf y_j) \\
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
翻书递推式可以推广为：
$$
\begin{aligned}
\log \mathbf s_{0}& =0,\mathbf o_{0} =\mathbf 0, \\
\log \mathbf s_t&=  \log  \mathbf s_{t-1} + \mathbf e_t,\\

\mathbf{o}_t&=\left(\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{o}_{t-1}+\left(1-\mathbf s_{t-1} / \mathbf s_t\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.
\end{aligned}
$$
注意到此时：
$$
\begin{aligned}
\mathbf{o}_t&=(\exp(-\mathbf e_t)) \mathbf{o}_{t-1}+\left(1-\exp(-\mathbf e_t)\right) \mathbf{x}_j ,\\
t&=1,\ldots, n.

\end{aligned}
$$
公式$\ref{ref7}$可以推广为：
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
