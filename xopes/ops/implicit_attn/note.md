# Inverse attention

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初始state $\mathbf s_0$，以及Decay $\Lambda \in \mathbb R^{n\times d}, \mathbf A\in \mathbb R^{n\times d},\mathbf B \in \mathbb R^{n\times e}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &=  (\mathrm{diag}(\lambda_i)+ \mathbf a_i \mathbf b_i^\top) \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
展开可得：
$$
\begin{aligned}
\mathbf s_t
&=\mathrm{diag}(\gamma_t) \mathbf s_0 + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
\left(
\mathbf k_i \mathbf v_i^\top + \mathbf a_i \mathbf p_i^\top
\right), \\

\mathbf p_{t+1}
& = \mathbf s_0^\top \mathrm{diag}(\gamma_t) \mathbf b_{t+1} +

\sum_{i=1}^t
\left(
 \mathbf v_i \mathbf k_i^\top +  \mathbf p_i\mathbf a_i^\top
\right)\mathrm{diag}(\gamma_t/\gamma_i)\mathbf b_{t+1} \\

&= \mathbf s_0^\top \mathrm{diag}(\gamma_{t+1}) \mathbf c_{t+1} +

\sum_{i=1}^t
\left(
 \mathbf v_i \mathbf k_i^\top +  \mathbf p_i\mathbf a_i^\top
\right)\mathrm{diag}(\gamma_{t+1}/\gamma_i)\mathbf c_{t+1} \\

\mathbf p_1 &=  \mathbf s_0^\top \mathbf b_1, \\

\mathbf b_t &= \mathbf c_t \odot \lambda_{t}

\end{aligned}
$$
另一方面，注意到：
$$
\mathbf p_{t+1}= \mathbf s_t^\top \mathbf b_{t+1}.
$$
那么：
$$
\begin{aligned}
\mathbf {da}_t
&= \frac{\partial l}{\mathbf {da_t}} \\

&= \sum_{j\ge t+1}\frac{\partial l}{\partial \mathbf {p}_j}
\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t} \\

&=  \sum_{j\ge t+1}  \left[\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}\right]^\top \mathbf {dp}_j .


\end{aligned}
$$
注意到：
$$
\begin{aligned}
\left[ \frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t} \right]_{n,m}
&= \frac{\partial [\mathbf p_t]_n \sum  [\mathbf a_t ]_k [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]_k }{\partial [\mathbf a_t]_m} \\

&= [\mathbf p_t]_n  [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]_m, \\

\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}&= \mathbf p_t [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]^\top .

\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {da}_t

&=  \sum_{j\ge t+1}  \left[\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}\right]^\top \mathbf {dp}_j \\

&= \sum_{j\ge t+1} \mathbf c_j  \mathrm{diag}(\gamma_j / \gamma_t) \mathbf p_t^\top \mathbf {dp}_j.


\end{aligned}
$$
补充：

展开第一个公式可得：
$$
\mathbf p_{t+1} = \mathbf s_0^\top \mathrm{diag}(\gamma_{t+1}) \mathbf c_{t+1} +

\sum_{i=1}^t
\left(
 \mathbf v_i \mathbf k_i^\top +  \mathbf p_i\mathbf a_i^\top
\right)\mathrm{diag}(\gamma_{t+1}/\gamma_i)\mathbf c_{t+1}.
$$
写成矩阵求逆可得：
$$
\begin{aligned}
\mathbf P
& = \mathrm{diag}(\Gamma)\mathbf C \mathbf S_0 +
\mathrm{tril}( \bar {\mathbf C} \bar {\mathbf K},-1)\mathbf V +\mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A},-1)\mathbf P,  \\

(\mathbf I - \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A},-1))
\mathbf P &= \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf K},-1)\mathbf V + \bar {\mathbf C} \mathbf S_0.

\end{aligned}
$$


# local -> global

考虑线性方程：
$$
(\mathbf I - \mathrm{tril}(\mathbf A\mathbf B^\top, -1)) \mathbf X = \mathbf Y.
$$
我们首先求解local方程：
$$
(\mathbf I - \mathrm{tril}(\mathbf A_i\mathbf B_i^\top, -1)) \mathbf Z_i = \mathbf Y_i.
$$
下面讨论，给定$\mathbf Z_i$，如何计算$\mathbf X_i$，为了方便讨论，我们定义：
$$
\begin{aligned}
\mathbf C &=  (\mathbf I - \mathrm{tril}(\mathbf A\mathbf B^\top, -1)), \\
\mathbf C_{i,j}&= \mathbf A_i \mathbf B_j, \\
\mathbf C_{i,i} &= (\mathbf I - \mathrm{tril}(\mathbf A_i\mathbf B_i^\top, -1)).

\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf X_i &= \mathbf C_{i,i}^{-1}
\left(
\mathbf Y_i - \sum_{j=1}^{i-1} \mathbf C_{i,j} X_j
\right) \\

&= \mathbf C_{i,i}^{-1}\mathbf Y_i - \mathbf C_{i,i}^{-1} \sum_{j=1}^{i-1} \mathbf C_{i,j} X_j \\

&= \mathbf Z_i  -  \mathbf C_{i,i}^{-1} \sum_{j=1}^{i-1} \mathbf A_i \mathbf B_j^\top \mathbf  X_j \\
&= \mathbf Z_i  -  \mathbf C_{i,i}^{-1}  \mathbf A_i \sum_{j=1}^{i-1}  \mathbf B_j^\top  \mathbf  X_j \\
&= \mathbf Z_i  -   \mathbf D_i\sum_{j=1}^{i-1}  \mathbf B_j^\top  \mathbf  X_j .

\end{aligned}
$$
因此我们得到算法：

- 并行求解：
  - $[\mathbf Z_i, \mathbf D_i] = (\mathbf I - \mathrm{tril}(\mathbf A_i\mathbf B_i^\top, -1))^{-1} [\mathbf Y_i, \mathbf A_i ]$；
- Use chunk loop:
  - $\mathbf S_0 = 0$；
  - $\mathbf X_i = \mathbf Z_i - \mathbf D_i \mathbf S_{i-1}$；
  - $\mathbf S_i = \mathbf S_{i-1} + \mathbf B_i^\top \mathbf X_i$；



## 应用

回顾之前的算法：
$$
\begin{aligned}
\mathbf s_t
&=\mathrm{diag}(\gamma_t) \mathbf s_0 + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
\left(
\mathbf k_i \mathbf v_i^\top + \mathbf a_i \mathbf p_i^\top
\right), \\

\mathbf p_{t+1}
& = \mathbf s_0^\top \mathrm{diag}(\gamma_t) \mathbf b_{t+1} +

\sum_{i=1}^t
\left(
 \mathbf v_i \mathbf k_i^\top +  \mathbf p_i\mathbf a_i^\top
\right)\mathrm{diag}(\gamma_t/\gamma_i)\mathbf b_{t+1} \\

&= \mathbf s_0^\top \mathrm{diag}(\gamma_{t+1}) \mathbf c_{t+1} +

\sum_{i=1}^t
\left(
 \mathbf v_i \mathbf k_i^\top +  \mathbf p_i\mathbf a_i^\top
\right)\mathrm{diag}(\gamma_{t+1}/\gamma_i)\mathbf c_{t+1} \\

\mathbf p_1 &=  \mathbf s_0^\top \mathbf b_1.


\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf P
& = \mathrm{diag}(\Gamma)\mathbf C \mathbf S_0 +
\mathrm{tril}( \bar {\mathbf C} \bar {\mathbf K}^\top,-1)\mathbf V +\mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1)\mathbf P,  \\

(\mathbf I - \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1))
\mathbf P &= \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf K}^\top,-1)\mathbf V + \bar {\mathbf C} \mathbf S_0.

\end{aligned}
$$
特殊情况，如果$\mathbf A=\mathbf K$，那么：
$$
\begin{aligned}
(\mathbf I - \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1))
\mathbf P
& = \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1)\mathbf V + \bar {\mathbf C} \mathbf S_0  \\
(\mathbf I - \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1))
\mathbf P & = \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1)\mathbf V - \mathbf V + \mathbf V+ \bar {\mathbf C} \mathbf S_0 \\


\mathbf P &= (\mathbf I - \mathrm{tril}( \bar {\mathbf C} \bar {\mathbf A}^\top,-1))^{-1}(\mathbf V+ \bar {\mathbf C} \mathbf S_0) - \mathbf V.
\end{aligned}
$$




### Fwd

- Step1：计算$\mathbf Y$
  - 如果不share：$\mathrm{tril}( \bar {\mathbf C} \bar {\mathbf K}^\top,-1)\mathbf V + \bar {\mathbf C} \mathbf S_0$；
    - 调用lightning；
  - 如果share：计算$\mathbf V+\bar {\mathbf C} \mathbf S_0$；
- Step2：
  - 并行求解：
    - $[\mathbf Z_i, \mathbf D_i] = (\mathbf I - \mathrm{tril}(\bar{\mathbf  C}_i \bar{ \mathbf A}_i^\top, -1))^{-1} [\mathbf Y_i, \bar{\mathbf C}_i ]$；
    - Use chunk loop:
      - $\mathbf H_0 = 0$；
      - $\mathbf P_i = \mathbf Z_i - \mathbf D_i \mathbf H_{i-1}$；
      - $\mathbf H_i = \mathbf H_{i-1} + \bar{\mathbf A}_i^\top \mathbf P_i$；
  - 注意share 版本还没有减$\mathbf V_i$；
- Step3：
  - 实现double gla：
    - $\mathbf s_t=\mathrm{diag}(\gamma_t) \mathbf s_0 + \sum_{i=1}^t  \mathrm{diag}(\gamma_t/\gamma_i)
      \left(
      \mathbf k_i \mathbf v_i^\top + \mathbf a_i \mathbf p_i^\top
      \right)$
      - 如果share，$\mathbf k_i =\mathbf a_i$，那么：$\mathbf v_i + \mathbf p_i= \mathbf p_i $，右边的$\mathbf p_i$即为前一步求解的$\mathbf p_i$；



### Bwd

因为前向是double gla，所以反向的梯度计算完全类似，设置reverse=False即可，另一方面，decay的计算为：
$$
\mathbf {d}\gamma_t =2 \mathbf {dq}_t - \mathbf {dk}_t -\mathbf {dv_t}.
$$
那么：
$$
\begin{aligned}
\mathbf {da}_t
&= \frac{\partial l}{\mathbf {da_t}} \\

&= \sum_{j\ge t+1}\frac{\partial l}{\partial \mathbf {p}_j}
\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t} \\

&=  \sum_{j\ge t+1}  \left[\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}\right]^\top \mathbf {dp}_j .


\end{aligned}
$$
注意到：
$$
\begin{aligned}
\left[ \frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t} \right]_{n,m}
&= \frac{\partial [\mathbf p_t]_n \sum  [\mathbf a_t ]_k [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]_k }{\partial [\mathbf a_t]_m} \\

&= [\mathbf p_t]_n  [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]_m, \\

\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}&= \mathbf p_t [\mathrm{diag}(\gamma_j / \gamma_t) \mathbf c_j]^\top .

\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {da}_t

&=  \sum_{j\ge t+1}  \left[\frac{\partial \mathbf {p}_j}{\partial \mathbf {a}_t}\right]^\top \mathbf {dp}_j \\

&= \sum_{j\ge t+1} \mathbf c_j  \mathrm{diag}(\gamma_j / \gamma_t) \mathbf p_t^\top \mathbf {dp}_j.


\end{aligned}
$$
所以这里再调一次gla。



## Matrix inverse forword substitution

给定矩阵$\mathbf A\in \mathbb R^{n\times n}$，其中：
$$
\mathbf A_{ij}=0, i > j.
$$
我们的目标是计算$\mathbf A^{-1}$。

根据定义可得：
$$
\sum_{j=1}^i a_{ij} x_{jk}= 1_{i=k}.
$$
向量化该方程即为：
$$
a_{ii} x_{ik} =1_{i=k}-\sum_{j=1}^{i-1} a_{ij} x_{jk}, \\
\mathbf x_i^\top  = \mathbf e_{i}^\top / a_{ii} - 1/a_{ii}\sum_{j=1}^{i-1} a_{ij} \mathbf x_{j}
=\mathbf e_{i}^\top  / a_{ii}-  1/a_{ii} [\mathbf a_i[:i-1]^\top \mathbf  X[:i-1, :]].
$$
注意到如果我们不需要后续使用$\mathbf A$，则可以使用inplace操作，这是因为，为了计算$\mathbf x_i$，我们需要$\mathbf A$的第$i$行，而$\mathbf X[:i-1, :]$可以存储在$\mathbf A $的前$i-1$行。



$i=1$时，
$$
a_{11} x_{1 k} = 1_{1=k}, x_{1k}=\frac{1_{1=k}}{a_{11}}.
$$
$i=2$时，
$$
a_{21} x_{1k} + a_{22} x_{2k} = 1_{2=k},
x_{2k} =(1_{2=k}- a_{21} x_{1k})/a22.
$$
更一般的，假设已知：
$$
\mathbf x_1 , \ldots, \mathbf x_k.
$$




## Matrix inverse Jacobian

给定矩阵$\mathbf A\in \mathbb R^{n\times n}$，其中：
$$
\mathbf A_{ij}=0, i > j.
$$
我们的目标是计算$\mathbf A^{-1}$。

我们记：
$$
\mathbf L = \mathrm{tril}(\mathbf A), \Lambda = \mathrm{diag}(\mathbf A).
$$
注意到：
$$
\begin{aligned}
&\mathbf A \mathbf x = \mathbf b, \\
\Leftrightarrow & (\Lambda +\mathbf L) \mathbf x = \mathbf b, \\
\Leftrightarrow & (\mathbf I - (-\Lambda^{-1} \mathbf L) ) \mathbf x = \Lambda ^{-1}\mathbf b .
\end{aligned}
$$
最后一行记为：
$$
\begin{aligned}
(\mathbf I - \mathbf L_1) \mathbf x &= \mathbf b_1 , \\
\mathbf x &= (\mathbf I - \mathbf L_1)^{-1} \mathbf b_1 \\

&= \sum_{k=0}^\infty  \mathbf L_1^k \mathbf b_1,  \\

\mathbf x_k &= \mathbf L_1 \mathbf x_{k-1} + \mathbf b_1.

\end{aligned}
$$

### 分块求逆矩

假设已知$\mathbf A_{ij}^{-1}, j \le i$，那么：
$$
\begin{aligned}
\sum_{j=k}^{i+1} \mathbf A_{i+1,j} \mathbf A^{-1}_{jk} &= \mathbf 0, k < i+1,  \\

\mathbf A_{i+1, k}^{-1} & =-\mathbf A_{i+1, i+1}^{-1} \left( \sum_{j=k}^{i} \mathbf A_{i+1, j} \mathbf A^{-1}_{jk}\right).

\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf A_{21}^{-1} &= -\mathbf A_{22}^{-1} \mathbf A_{21} \mathbf A_{11}^{-1}, \\
\mathbf A_{32}^{-1} &=- \mathbf A_{33}^{-1} \mathbf A_{32} \mathbf A_{22}^{-1}, \\
\mathbf A_{31}^{-1} &=- \mathbf A_{33}^{-1} \left(
\mathbf A_{31} \mathbf A_{11}^{-1}  +
\mathbf A_{32} \mathbf A_{21}^{-1} \right), \\
\mathbf A_{43}^{-1} &= - \mathbf A_{44}^{-1} \mathbf A_{43} \mathbf A_{33}^{-1},  \\

\mathbf A_{42}^{-1} &= - \mathbf A_{44}^{-1}
\left(
\mathbf A_{42} \mathbf A_{22}^{-1} + \mathbf A_{43} \mathbf A_{32}^{-1}
\right), \\

\mathbf A_{41}^{-1} &= - \mathbf A_{44}^{-1}
\left(
\mathbf A_{41} \mathbf A_{11}^{-1} + \mathbf A_{42} \mathbf A_{21}^{-1}+ \mathbf A_{43} \mathbf A_{31}^{-1}
\right).

\end{aligned}
$$






## 投影空间linear attn[fail]

$$
\mathbf s_t =\mathrm{diag}(1- \lambda_t) \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top.
$$

记：
$$
\gamma_t = \prod_{j=1}^t (1-\lambda_j).
$$
我们猜测，上式可以表达为：
$$
\mathbf s_t = \mathrm{diag}(\gamma_t) \mathbf s_0 + \sum_{j=1}^t \mathbf p_j \mathbf v_j^\top
+ \sum_{j=1}^t \mathbf k_j \mathbf v_j^\top.
$$
$t=1$时，
$$
\mathbf s_1 = \mathrm{diag}(1- \lambda_0) \mathbf s_0  + \mathbf k_1 \mathbf v_1^\top,
\mathbf p_1 = \mathbf 0.
$$
假设$t$时结论成立，那么当$t+1$时：
$$
\begin{aligned}
\mathbf s_{t+1}
& =\mathrm{diag}(1- \lambda_{t+1}) \mathbf s_{t} + \mathbf k_{t+1} \mathbf v_{t+1}^\top \\
&= \mathrm{diag}(1- \lambda_{t+1}) \left( \mathrm{diag}(\gamma_t) \mathbf s_0 + \sum_{j=1}^t \mathbf p_j \mathbf v_j^\top
+ \sum_{j=1}^t \mathbf k_j \mathbf v_j^\top \right) + \mathbf k_{t+1} \mathbf v_{t+1}^\top \\

&= \mathrm{diag}(\gamma_{t+1}) \mathbf s_0
+ \sum_{j=1}^t \mathbf p_j \mathbf v_j^\top
+ \sum_{j=1}^t \mathbf k_j \mathbf v_j^\top

+ \mathbf k_{t+1} \mathbf v_{t+1}^\top

- \mathrm{diag}( \lambda_{t+1}) \sum_{j=1}^t \mathbf p_j \mathbf v_j^\top
- \mathrm{diag}( \lambda_{t+1}) \sum_{j=1}^t \mathbf k_j \mathbf v_j^\top   \\

&= \mathrm{diag}(\gamma_{t+1}) \mathbf s_0 + \sum_{j=1}^t \mathbf k_j \mathbf p_j^\top
+


\end{aligned}
$$
