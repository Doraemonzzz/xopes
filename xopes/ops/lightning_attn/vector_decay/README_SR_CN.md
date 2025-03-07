# Lightning Attention with Vector Decay(Sequential Recurrence)

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e}$，我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= (\lambda_i\gamma_i^\top)\odot   \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times e}.
$$



## Forward

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e}$，注意如果Decay为空，我们使用$\Lambda=1-\mathbf K, \Gamma=1-\mathbf V$（此时我们默认$0\le \mathbf K \le 1, 0\le \mathbf V \le 1$）。

我们执行如下递归：
$$
\begin{aligned}
\mathbf s_0 &\in \mathbb R^{d\times e}, \\
\mathbf s_i &= [\lambda_i\gamma_i^\top]\odot   \mathbf s_{i-1} + \mathbf k_i \mathbf v_i^\top \\
&\triangleq  \mathbf a_i \odot \mathbf {s}_{i-1} + \mathbf k_i\mathbf v_i^\top, \\
\mathbf o_i^\top&= \mathbf q_i^\top\mathbf s_i \in \mathbb R^{e}.
\end{aligned}
$$
返回：
$$
\mathbf O= \left[\begin{matrix}
\mathbf o_1^\top  \\
\vdots \\
\mathbf o_n^\top  \\
\end{matrix} \right]\in \mathbb R^{n\times e}.
$$

定义：
$$
\begin{aligned}

\prod_{j=1}^t \lambda_j & = \alpha_t,
\log \alpha_ t= \sum_{j=1}^t \log \lambda_j,  \\
\prod_{j=1}^t \gamma_j & =\beta_t,
\log \beta_ t \sum_{j=1}^t \log \gamma_j,  \\
\mathbf a_t &= \lambda_t \gamma_t^\top,  \\
\mathbf A_t &= \odot_{j=1}^t \mathbf a_j
= \alpha_t \beta_t^\top.
\end{aligned}
$$
展开公式可得：
$$
\begin{aligned}
\mathbf s_t &= \mathbf a_t \odot  \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&=  \mathbf a_{t} \odot
\left( \mathbf a_{t-1} \odot  \mathbf s_{t-2} +\mathbf k_{t-1} \mathbf v_{t-1}^\top  \right)+\mathbf k_t \mathbf v_t^\top  \\
&= \mathbf a_{t} \odot  \mathbf a_{t-1} \odot \mathbf s_{t-2}  + \mathbf a_{t} \odot  \mathbf k_{t-1} \mathbf v_{t-1}^\top
+ \mathbf k_t \mathbf v_t^\top \\
&=  \ldots \\
&= \mathbf A_t\odot  \mathbf s_0  + \sum_{j=1}^t \left(\odot_{i=j+1}^t \mathbf a_i \right) \odot  \mathbf k_j \mathbf v_j^\top \\
&= \mathbf A_t\odot \mathbf s_0  + \sum_{j=1}^t \frac{\mathbf A_t}{\mathbf A_{j}}\odot  \mathbf k_j\mathbf v_j ^\top \\

&= \alpha_t \beta_t^\top \odot \mathbf s_0  + \sum_{j=1}^t \frac{ \alpha_t \beta_t^\top}{ \alpha_j \beta_j^\top}\odot  \mathbf k_j\mathbf v_j ^\top.
\end{aligned}
$$
注意到：
$$
\begin{aligned}
\mathbf q_t^\top
\left[\frac{ \alpha_t \beta_t^\top}{ \alpha_j \beta_j^\top}\odot  \mathbf k_j\mathbf v_j ^\top  \right]

&= \left[(\mathbf q_t\odot \alpha_t)^\top(\mathbf k_j / \alpha_j)(\mathbf v_j /\beta_j)^\top \right]\odot \beta_t.
\end{aligned}
$$
所以：
$$
\begin{aligned}
\mathbf o_t^\top
&= \mathbf q_t^\top [ \alpha_t \beta_t^\top \odot \mathbf s_0] +
\left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \odot \beta_t ,\\


\mathbf s_t
&=(\alpha_t\beta_t^\top)\odot \mathbf s_0  + \sum_{j=1}^t \frac{(\alpha_t\beta_t^\top)}{(\alpha_j\beta_j^\top)}\odot  \mathbf k_j\mathbf v_j ^\top.


\end{aligned}
$$



## Backward

### $\mathbf{dq}_n, \mathbf{dk}_n,\mathbf {dv}_n$

给定$\mathbf {do}_1,\ldots, \mathbf {do}_n, \mathbf {ds}_n$，计算：
$$
\begin{aligned}
\mathbf{dq}_t
&= \sum_{s=1}^n \frac{\partial \mathbf o_s}{\partial \mathbf q_t}\frac{\partial l}{\partial \mathbf o_s}  \\
&=\frac{\partial \mathbf o_t}{\partial \mathbf q_t}\frac{\partial l}{\partial \mathbf o_t}  \\
&=[ \alpha_t \beta_t^\top \odot \mathbf s_0] \mathbf{do}_t
+   \alpha_t\odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t), \\

\mathbf{dk}_t
&= \sum_{s=1}^n \frac{\partial \mathbf o_s}{\partial \mathbf k_t}\frac{\partial l}{\partial \mathbf o_s}
+ \frac{\partial  \mathbf s_n}{\partial \mathbf k_t} \frac{\partial l}{\partial \mathbf s_n}
\\
&= \sum_{s\ge t}^n \frac{\partial \mathbf o_s}{\partial \mathbf k_t}\frac{\partial l}{\partial \mathbf o_s}
+
\left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\\
&= (1/\alpha_t) \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)
+ \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t, \\


\mathbf{dv}_t
&= \sum_{s=1}^n \frac{\partial \mathbf o_s}{\partial \mathbf v_t}\frac{\partial l}{\partial \mathbf o_s}
+ \frac{\partial  \mathbf s_n}{\partial \mathbf v_t} \frac{\partial l}{\partial \mathbf s_n}\\
&= \sum_{s\ge t}^n \frac{\partial \mathbf o_s}{\partial \mathbf v_t}\frac{\partial l}{\partial \mathbf o_s}
+
\left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t \\
&= (1/\beta_t) \odot  \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right) +
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t.




\end{aligned}
$$



### $\mathbf {ds}_n$对于梯度的贡献

另一方面：
$$
\begin{aligned}

\mathbf s_n
&=(\alpha_n\beta_n^\top)\odot \mathbf s_0  + \sum_{j=1}^n \frac{(\alpha_n\beta_n^\top)}{(\alpha_j\beta_j^\top)}\odot  \mathbf k_j\mathbf v_j ^\top, \\


\end{aligned}
$$
所以：
$$
\begin{aligned}



 \frac{\partial [\mathbf{s}_n]_{ij}} {\partial [\mathbf{\log \alpha}_n]_r}
 &= \mathbf 1_{i=r}
 \left([\alpha_n]_i [\beta_n ]_j [\mathbf s_0]_{ij}  +   \sum_{s=1}^{n}[\alpha_n]_i [\beta_n ]_j/([\alpha_s]_i  [\beta_s ]_j)     [\mathbf k_s \mathbf v_s^\top]_{ij}
 -[\alpha_n]_i [\beta_n ]_j/([\alpha_n]_i  [\beta_n ]_j)     [\mathbf k_n \mathbf v_n^\top]_{ij} \right),
  \\

  \frac{\partial [\mathbf{s}_n]_{ij}} {\partial [\mathbf{\log \alpha}_t]_r}
 &=-\mathbf 1_{i=r}
 \left([\alpha_n]_i [\beta_n ]_j/([\alpha_t]_i  [\beta_t ]_j)     [\mathbf k_t \mathbf v_t^\top]_{ij} \right),
 t<n, \\

[\mathbf d \mathbf{\log \alpha}_n]_r
&=\sum_{i,j} [\mathbf {ds}_n]_{ij}  \frac{\partial [\mathbf{s}_n]_{ij}} {\partial [\mathbf{\log \alpha}_n]_r} \\
&=\sum_{i,j} [\mathbf {ds}_n]_{ij}  \mathbf 1_{i=r}\left(
[\alpha_n]_i [\beta_n ]_j [\mathbf s_0]_{ij}  +   \sum_{s=1}^{n-1}[\alpha_n]_i [\beta_n ]_j/([\alpha_s]_i  [\beta_s ]_j)     [\mathbf k_s \mathbf v_s^\top]_{ij} \right) \\
&=\sum_{j} [\mathbf {ds}_n]_{rj} \left(
[\alpha_n]_r [\beta_n ]_j [\mathbf s_0]_{rj}  +   \sum_{s=1}^{n-1}[\alpha_n]_r [\beta_n ]_j/([\alpha_s]_r  [\beta_s ]_j)     [\mathbf k_s \mathbf v_s^\top]_{rj} \right)  \\

\mathbf d \mathbf{\log \alpha}_n&= \alpha_n\odot  [(\mathbf {ds}_n \odot \mathbf s_0) \beta_n ]
+\sum_{s=1}^{n-1}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]\mathbf 1_{e},\\


[\mathbf d \mathbf{\log \alpha}_t]_r
&=\sum_{i,j} [\mathbf {ds}_n]_{ij}  \frac{\partial [\mathbf{s}_n]_{ij}} {\partial [\mathbf{\log \alpha}_t]_r} \\
&=-\sum_{i,j} [\mathbf {ds}_n]_{ij}  \mathbf 1_{i=r}\left([\alpha_n]_i [\beta_n ]_j/([\alpha_t]_i  [\beta_t ]_j)     [\mathbf k_t \mathbf v_t^\top]_{ij} \right) \\
&=-\sum_{j} [\mathbf {ds}_n]_{rj} \left(
[\alpha_n]_r [\beta_n ]_j/([\alpha_t]_r  [\beta_t ]_j)     [\mathbf k_t \mathbf v_t^\top]_{rj} \right)  \\

\mathbf d \mathbf{\log \alpha}_t&=   -[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}, t< n.


\end{aligned}
$$
注意到：
$$
\begin{aligned}
\left[\alpha_n\odot  [(\mathbf {ds}_n \odot \mathbf s_0) \beta_n ] \right]_i
&= \sum_j  [\alpha_n]_i [\mathbf {ds}_n]_{ij} [ \mathbf s_0]_{ij}[\beta_n ]_j, \\

\left[ [(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ] \mathbf 1_e \right]_i.
&= \sum_j [\mathbf {ds}_n]_{ij} [\mathbf s_0 ]_{ij} [\alpha_n]_i  [\beta_n]_j, \\

\alpha_n\odot  [(\mathbf {ds}_n \odot \mathbf s_0) \beta_n ]&= [(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ] \mathbf 1_e.

\end{aligned}
$$
因此：
$$
\mathbf{d}\log \bar \alpha_t =
\begin{cases}
 -[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e},  & t< n, \\

[(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ] \mathbf 1_e
+\sum_{s=1}^{n-1}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]\mathbf 1_{e}, & t=n.
\end{cases}
$$
类比可得：
$$
\mathbf {d}\log \bar \beta_t =

\begin{cases}
-[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d}, & t < n,  \\

[(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ]^\top \mathbf 1_d
+\sum_{s=1}^{n-1}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]^\top \mathbf 1_{d}, & t= n.


\end{cases}
$$
注意，上式是$\mathbf {ds}_n$部分对于梯度的贡献，为了进行区分，我们添加了上划线。

最后，我们补充$\mathbf {ds}_0$：
$$
\begin{aligned}
\frac{\partial [\mathbf s_n]_{ij}}{\partial [\mathbf s_0]_{st}}
&= [\alpha_n]_s [\beta_n]_t 1_{i=s}1_{j=t}, \\
[\mathbf {d\bar s}_0]_{st}

&= \sum_{ij} [\mathbf {ds}_n]_{ij} \frac{\partial [\mathbf s_n]_{ij}}{\partial [\mathbf s_0]_{st}} \\

&= \sum_{ij} [\mathbf {ds}_n]_{ij} [\alpha_n]_s [\beta_n]_t 1_{i=s}1_{j=t} \\
&= [\mathbf {ds}_n]_{st} [\alpha_n]_s [\beta_n]_t , \\

\mathbf {d\bar s}_0 &= [\mathbf {ds}_n] \odot [\alpha_n\beta_n^\top].

\end{aligned}
$$

### $\mathbf d{\log}\alpha_t$

$$
\begin{aligned}
\mathbf {d\log \alpha_t}
&= \sum_{s=1}^n \frac{\partial \mathbf o_s}{\partial \log \alpha_t}\frac{\partial l}{\partial \mathbf o_s}
+  \frac{\partial \mathbf s_n}{\partial  \log \alpha_t} \frac{\partial l}{\partial \mathbf s_n}
\\
&= \alpha_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]\beta_t \right]  +
\alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t)
 \\
&- (1/\alpha_t)\odot \mathbf k_t \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)
+  \mathbf{d}\log \bar \alpha_t .

\end{aligned}
$$

对于第一项，注意到：
$$
\begin{aligned}
{\left[\mathbf{q}_t \odot\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right] \mathbf{d o}_t\right]_i } & =\sum_j\left[\mathbf{q}_t\right]_i\left[\alpha_t\right]_i\left[\beta_t\right]_j\left[\mathbf{s}_0\right]_{i j}\left[\mathbf{d o}_t\right]_j, \\
{\left[\alpha_t \odot\left[\left[\mathbf{q}_t \mathbf{d o}_t^{\top}\right] \odot \mathbf{s}_0\right] \beta_t\right]_i } & =\sum_j\left[\alpha_t\right]_i\left[\mathbf{q}_t\right]_i\left[\mathbf{d} \mathbf{o}_t\right]_j\left[\mathbf{s}_0\right]_{i j}\left[\beta_t\right]_j, \\

\mathbf{q}_t \odot\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right] \mathbf{d} \mathbf{o}_t & =\alpha_t \odot\left[\left[\mathbf{q}_t \mathbf{d} \mathbf{o}_t^{\top}\right] \odot \mathbf{s}_0\right] \beta_t .
\end{aligned}
$$
回顾：
$$
\begin{aligned}
\mathbf{dq}_t
&=[ \alpha_t \beta_t^\top \odot \mathbf s_0] \mathbf{do}_t
+   \alpha_t\odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t) ,  \\


\mathbf{dk}_t
&= (1/\alpha_t) \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)
+ \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t.

\end{aligned}
$$
所以前两项可以化简为：
$$
\begin{aligned}
&\alpha_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]\beta_t \right]  +
\alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t)  \\

= &\alpha_t \odot\left[\left[\mathbf{q}_t \mathbf{d} \mathbf{o}_t^{\top}\right] \odot \mathbf{s}_0\right] \beta_t + \alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t)   \\

=& \mathbf{q}_t \odot\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right] \mathbf{d} \mathbf{o}_t
+ \alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t)    \\

=& \mathbf q_t \odot  \mathbf {dq}_t.

\end{aligned}
$$
第三项可以化简为：
$$
\begin{aligned}
&(1/\alpha_t)\odot \mathbf k_t \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)  \\

=& \mathbf k_t \odot
\left[

\mathbf {dk}_t -   \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t

\right] \\
= & \mathbf k_t \odot  \mathbf {dk}_t -  \mathbf k_t \odot \left[

  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t

\right].



\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {d\log \alpha_t}

&= \alpha_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]\beta_t \right]  +
\alpha_t\odot \mathbf q_t \odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t)
 \\
&- (1/\alpha_t)\odot \mathbf k_t \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)
+ \mathbf{d}\log \bar \alpha_t  \\

&=  \mathbf q_t \odot  \mathbf {dq}_t -  \mathbf k_t \odot  \mathbf {dk}_t +
\mathbf k_t \odot \left[

  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right]
+ \mathbf{d}\log \bar \alpha_t .

\end{aligned}
$$
现在考虑上式的后两项，首先我们有如下等式：
$$
\begin{aligned}
\left[ \mathbf k_t \odot \left[
  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right] \right]_i

&= \sum_j [\mathbf k_t ]_i [ \mathbf {ds}_n]_{ij} [\alpha_n]_i[\beta_n]_j / ([\alpha_t]_i [\beta_t]_j) [\mathbf v_t]_j, \\

\left[[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}\right]_i
&= \sum_j [\mathbf {ds}_n ]_i  [\mathbf k_t ]_i ( [\alpha_n]_i/ [\alpha_t]_i )
 [\mathbf v_t ]_j ( [\beta_n]_j/ [\beta_t]_j ),  \\

 \left[ \mathbf k_t \odot \left[
  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right] \right]_i &= \left[[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}\right]_i, \\

\mathbf k_t \odot \left[
  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right] &= [\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}.
\end{aligned}
$$
当$t=n$时：
$$
\begin{aligned}
&\mathbf k_t \odot \left[
  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right]
+ \mathbf{d}\log \bar \alpha_t  \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}
+ \mathbf{d}\log \bar \alpha_n \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e} +[(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ] \mathbf 1_e
+\sum_{s=1}^{n-1}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]\mathbf 1_{e} \\

=& [(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ] \mathbf 1_e +
\sum_{s=1}^{n}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \beta_n/\beta_s) (\mathbf v_s \odot \alpha_n/\alpha_s)^\top)]\mathbf 1_{e} \\
=& [\mathbf s_n \odot \mathbf {ds}_n]1_e.
\end{aligned}
$$
当$t<n$时：
$$
\begin{aligned}
&\mathbf k_t \odot \left[
  \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t
\right]
+ \mathbf{d}\log \bar \alpha_t  \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}
+ \mathbf{d}\log \bar \alpha_n \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}
-[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]\mathbf 1_{e}\\

=& \mathbf 0.
\end{aligned}
$$
综上：
$$
\mathbf {d}\log \alpha_t =
\begin{cases}
\mathbf q_t \odot  \mathbf {dq}_t -  \mathbf k_t \odot  \mathbf {dk}_t , & t < n \\
\mathbf q_t \odot  \mathbf {dq}_t -  \mathbf k_t \odot  \mathbf {dk}_t + [\mathbf s_n \odot \mathbf {ds}_n]1_e, & t=n
\end{cases}
$$
注意到：
$$
\begin{aligned}
\log \alpha_ t &= \sum_{j=1}^t \log \lambda_j, \\
\mathbf d \log \lambda_t
&= \sum_{j\ge t} \mathbf d \log \alpha_j.

\end{aligned}
$$
因为$\mathbf d \log \alpha_t$只有当$t=n$时会多出一项，因此我们得到如下公式：
$$
\begin{aligned}
\mathbf {d}\log \alpha_t
& =\mathbf q_t \odot  \mathbf {dq}_t -  \mathbf k_t \odot  \mathbf {dk}_t,  \\
\mathbf d \log \lambda_t
&= [\mathbf s_n \odot \mathbf {ds}_n]1_e + \sum_{j\ge t} \mathbf d \log \alpha_j.

\end{aligned}
$$


### $\mathbf d \log \beta_t$

根据之前的分析，我们不难猜出如下公式：
$$
\begin{aligned}
\mathbf {d}\log \beta_t
& =\mathbf o_t \odot  \mathbf {do}_t -  \mathbf v_t \odot  \mathbf {dv}_t,  \\
\mathbf d \log \gamma_t
&= [\mathbf s_n \odot \mathbf {ds}_n]^\top 1_d + \sum_{j\ge t} \mathbf d \log \beta_j.

\end{aligned}
$$
验证：
$$
\begin{aligned}
\mathbf {d\log \beta_t}
&= \sum_{s=1}^n \frac{\partial \mathbf{o}_s}{\partial \log \beta_t} \frac{\partial l}{\partial \mathbf{o}_s}+\frac{\partial \mathbf{s}_n}{\partial \log \beta_t} \frac{\partial l}{\partial \mathbf{s}_n}  \\

&= \beta_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]^\top \alpha_t \right]
+ \mathbf{do}_t \odot \left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \\

&- (1/\beta_t) \odot \mathbf v_t \odot   \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right) + \mathbf d\log \bar \beta_t.

\end{aligned}
$$
对于第一项，注意到：
$$
\begin{aligned}
{\left[\mathbf{q}_t^{\top}\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right] \odot \mathbf{d o}_t\right]_i } & =\sum_j\left[\mathbf{q}_t\right]_j\left[\alpha_t\right]_j\left[\beta_t\right]_i\left[\mathbf{s}_0\right]_{j i}\left[\mathbf{d o}_t\right]_i, \\
{\left[\beta_t \odot\left[\left[\left[\mathbf{q}_t \mathbf{d} \mathbf{o}_t^{\top}\right] \odot \mathbf{s}_0\right]^{\top} \alpha_t\right]\right]_i } & =\sum_j\left[\beta_t\right]_i\left[\mathbf{q}_t\right]_j\left[\mathbf{d o}_t\right]_i\left[\mathbf{s}_0\right]_{j i}\left[\alpha_t\right]_j, \\
\mathbf{q}_t^{\top}\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right] \odot \mathbf{d o}_t & =\left[\left[\left[\mathbf{q}_t \mathbf{d} \mathbf{o}_t^{\top}\right] \odot \mathbf{s}_0\right]^{\top} \alpha_t\right] \odot \beta_t .
\end{aligned}
$$
回顾：
$$
\begin{aligned}
\mathbf o_t^\top
&= \mathbf q_t^\top [ \alpha_t \beta_t^\top \odot \mathbf s_0] +
\left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \odot \beta_t, \\
\mathbf{dv}_t
&= (1/\beta_t) \odot  \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right) +
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t.

\end{aligned}
$$
所以前两项可以化简为：
$$
\begin{aligned}
&\beta_t\odot \left[\left[ [\mathbf q_t \mathbf{do}_t^\top ] \odot \mathbf s_0 \right]^\top \alpha_t \right]
+ \mathbf{do}_t \odot \left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right]  \\

=&\mathbf{d o}_t  \odot \mathbf{q}_t^{\top}\left[\alpha_t \beta_t^{\top} \odot \mathbf{s}_0\right]
+ \mathbf{do}_t \odot \left[ (\mathbf q_t \odot \alpha_t)^\top \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top \right] \\

=& \mathbf {do}_t \odot \mathbf o_t.
\end{aligned}
$$
第三项可以化简为：
$$
\begin{aligned}
 & (1/\beta_t) \odot \mathbf v_t \odot   \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right)  \\

=  &
 \mathbf v_t \odot \left(
 \mathbf {dv}_t - \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right) \\

 = & \mathbf v_t \odot \mathbf {dv}_t - \mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right].

\end{aligned}
$$
因此：
$$
\begin{aligned}
\mathbf {d\log \beta_t}
&= \sum_{s=1}^n \frac{\partial \mathbf{o}_s}{\partial \log \beta_t} \frac{\partial l}{\partial \mathbf{o}_s}+\frac{\partial \mathbf{s}_n}{\partial \log \beta_t} \frac{\partial l}{\partial \mathbf{s}_n}  \\

&= \mathbf o_t \odot \mathbf {do}_t - \mathbf v_t \odot \mathbf {dv}_t +\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right] + \mathbf d\log \bar \beta_t.

\end{aligned}
$$
现在考虑上式的后两项，首先我们有如下等式：
$$
\begin{aligned}
\left[\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right]
\right]_i

&= \sum_j [\mathbf v_t]_i [\mathbf {ds}_n]_{ji} [\alpha_n]_j [\beta_n]_i /([\alpha_t]_j [\beta_t]_i) [\mathbf k_t]_j, \\

\left[[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d} \right]_i

&= \sum_j [\mathbf {ds}_n]_{ji} [\mathbf k_t]_j [\alpha_n]_j /[\alpha_t]_j [\mathbf v_t]_i [\beta_n]_i[\beta_t]_j, \\

\left[\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right]
\right]_i &= \left[[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d} \right]_i, \\

\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right]
 &= [\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d} .
\end{aligned}
$$
当$t=n$时：
$$
\begin{aligned}
&\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right] + \mathbf d\log \bar \beta_t \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_n) (\mathbf v_t \odot \beta_n/\beta_n)^\top)]^\top \mathbf 1_{d}
+ \mathbf{d}\log \bar \beta_n \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_n) (\mathbf v_t \odot \beta_n/\beta_n)^\top)]^\top \mathbf 1_{d} +[(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ]^\top \mathbf 1_d
+\sum_{s=1}^{n-1}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]^\top \mathbf 1_{d} \\

=& [(\mathbf {ds}_n \odot \mathbf s_0 \odot (\alpha_n \beta_n^\top) ]^\top \mathbf 1_d +
\sum_{s=1}^{n}
[\mathbf {ds}_n \odot ((\mathbf k_s \odot \alpha_n/\alpha_s) (\mathbf v_s \odot \beta_n/\beta_s)^\top)]^\top \mathbf 1_{d}\\
=& [\mathbf s_n \odot \mathbf {ds}_n]^\top 1_d.
\end{aligned}
$$
当$t<n$时：
$$
\begin{aligned}
&\mathbf v_t \odot \left[
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t
 \right] + \mathbf d\log \bar \beta_t \\

= &[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d}
-[\mathbf {ds}_n \odot ((\mathbf k_t \odot \alpha_n/\alpha_t) (\mathbf v_t \odot \beta_n/\beta_t)^\top)]^\top \mathbf 1_{d} \\

=& \mathbf 0.

\end{aligned}
$$
综上：
$$
\mathbf {d}\log \beta_t =
\begin{cases}
\mathbf o_t \odot  \mathbf {do}_t -  \mathbf v_t \odot  \mathbf {dv}_t , & t < n \\
\mathbf o_t \odot  \mathbf {do}_t -  \mathbf v_t \odot  \mathbf {dv}_t + [\mathbf s_n \odot \mathbf {ds}_n]^\top 1_d, & t=n
\end{cases}
$$
注意到：
$$
\begin{aligned}
\log \beta_ t &= \sum_{j=1}^t \log \gamma_j, \\
\mathbf d \log \gamma_t
&= \sum_{j\ge t} \mathbf d \log \beta_j.

\end{aligned}
$$
因为$\mathbf d \log \beta_t$只有当$t=n$时会多出一项，因此我们得到如下公式：
$$
\begin{aligned}
\mathbf {d}\log \beta_t
& =\mathbf o_t \odot  \mathbf {do}_t -  \mathbf v_t \odot  \mathbf {dv}_t,  \\
\mathbf d \log \gamma_t
&= [\mathbf s_n \odot \mathbf {ds}_n]^\top 1_d + \sum_{j\ge t} \mathbf d \log \beta_j.

\end{aligned}
$$

###  $\mathbf {ds}_0$

$$
\begin{aligned}
\mathbf{ds}_0

&= \sum_{s=1}^n \frac{\partial \mathbf o_s}{\partial \mathbf s_0} \frac{\partial l}{\partial \mathbf o_s}
+\mathbf{d\bar s}_0 \\
&=   \sum_{s=1}^n [\alpha_s\beta_s^\top] \odot[\mathbf {q}_s  \mathbf {do_s}^\top]
+   [\mathbf {ds}_n] \odot [\alpha_n\beta_n^\top]
\end{aligned}
$$



### 化简

最后，我们考虑对$\mathbf {dk}_t, \mathbf {dv}_t$进行化简，我们假设$\mathbf{ds}_{n+1}\triangleq \mathbf{ds}_{n},\lambda_{n+1}=\mathbf 1_d, \gamma_{n+1}=\mathbf 1_e,\mathbf q_0=\mathbf 0_d, \mathbf{do}_0 = \mathbf 0_e$，考虑如下递推：
$$
\begin{aligned}
\mathbf {ds}_t
& = [\lambda_{t+1}\gamma_{t+1}^\top] \odot \mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t \\
& = [\lambda_{t+1}\gamma_{t+1}^\top] \odot
\left(
 [\lambda_{t+2}\gamma_{t+2}^\top] \odot \mathbf{ds}_{t+2}
+  \mathbf{q}_{t+1} \mathbf {do}^\top_{t+1}
\right)
+ \mathbf{q}_t\mathbf {do}^\top_t \\

&=  [\alpha_{t+2}\beta_{t+2}^\top / \alpha_{t}\beta_{t}]
\odot \mathbf{ds}_{t+2}  +  [\alpha_{t+1}\beta_{t+1}^\top / \alpha_{t}\beta_{t}]\odot [\mathbf{q}_{t+1}\mathbf {do}^\top_{t+1}]
+  \mathbf{q}_t\mathbf {do}^\top_t \\

&= \ldots \\

&=   [\alpha_{n+1}\beta_{n+1}^\top / \alpha_{t}\beta_{t}] \odot \mathbf{ds}_{n+1}
 + \sum_{j=t}^{n}[\alpha_{j}\beta_{j}^\top / \alpha_{t}\beta_{t}]   \odot [\mathbf{q}_{j}\mathbf {do}^\top_{j}]  \\

&=   [\alpha_{n}\beta_{n}^\top / \alpha_{t}\beta_{t}] \odot \mathbf{ds}_{n}
 + \sum_{j=t}^{n}[\alpha_{j}\beta_{j}^\top / \alpha_{t}\beta_{t}]   \odot [\mathbf{q}_{j}\mathbf {do}^\top_{j}],  \\

 \mathbf {ds}_0
 &= [\alpha_{n}\beta_{n}^\top / \alpha_{0}\beta_{0}] \odot \mathbf{ds}_{n}
 + \sum_{j=0}^{n} [\alpha_{j}\beta_{j}^\top / \alpha_{0}\beta_{0}]   \odot [\mathbf{q}_{j}\mathbf {do}^\top_{j}] \\

 &=  [\alpha_{n}\beta_{n}^\top ] \odot \mathbf{ds}_{n}
 +  \sum_{j=1}^{n} [\alpha_{j}\beta_{j}^\top]   \odot [\mathbf{q}_{j}\mathbf {do}^\top_{j}].

\end{aligned}
$$
所以：
$$
\begin{aligned}
\mathbf{dq}_t

&=[ \alpha_t \beta_t^\top \odot \mathbf s_0] \mathbf{do}_t
+   \alpha_t\odot \sum_{j=1}^t (\mathbf k_j/\alpha_j) (\mathbf v_j /\beta_j)^\top (\mathbf{do}_t  \odot \beta_t) \\
&= \mathbf s_t \mathbf {do}_t,  \\

\mathbf{dk}_t

&= (1/\alpha_t) \odot \left(\left[ \sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right] (\mathbf v_t /\beta_t) \right)
+ \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right] \mathbf v_t \\

&=  \mathbf {ds}_t \mathbf v_t, \\

\mathbf{dv}_t

&= (1/\beta_t) \odot  \left( \left[\sum_{s=t}^n  (\mathbf q_s \odot \alpha_s)(\mathbf{do}_s  \odot \beta_s)^\top \right]^\top (\mathbf k_t /\alpha_t)\right) +
 \left[ \mathbf {ds}_n \odot \frac{(\alpha_n\beta_n^\top)}{(\alpha_t\beta_t^\top)} \right]^\top \mathbf k_t\\

 &= \mathbf {ds}_t^\top \mathbf k_t .

 \end{aligned}
$$



## 整体化简

注意到反向的递推为（注意，这里第一行的$\mathbf {ds}_n$表示state梯度的输入）：
$$
\begin{aligned}
\mathbf {ds}_{n+1} &= \mathbf {ds}_n ,  \\
\mathbf {ds}_n  &= \mathbf {ds}_{n+1} + \mathbf{q}_n\mathbf {do}^\top_n, \\

\mathbf {ds}_t &= [\lambda_{t+1}\gamma_{t+1}^\top] \odot \mathbf{ds}_{t+1} + \mathbf{q}_t\mathbf {do}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {ds}_0&= [\lambda_{1}\gamma_{1}^\top] \odot \mathbf {ds}_1,  \\

\mathbf{dq}_t^\top &= \mathbf {do}_t^\top \mathbf s_t ^\top  ,\\

\mathbf{dk}_t^\top &=\mathbf v_t^\top \mathbf {ds}_t^\top,  \\

\mathbf{dv}_t& = \mathbf k_t^\top \mathbf {ds}_t.
\end{aligned}
$$
假设我们定义fwd的函数为：
$$
\mathbf O, \bar {\mathbf s}= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \Gamma, \mathbf s, \mathrm{reverse}).
$$
其中$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e},\Lambda\in \mathbb R^{n\times d},\Gamma \in \mathbb R^{n\times e},\mathbf O\in \mathbb R^{n\times e}, \mathbf s\in \mathbb R^{d\times e}$：

如果reverse = false:
$$
\begin{aligned}
\mathbf s_0 &=\mathbf s, \\
\mathbf s_t &= [\lambda_t\gamma_t^\top]\odot   \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top \\
&\triangleq  \mathbf a_t \odot \mathbf {s}_{t-1} + \mathbf k_t\mathbf v_t^\top, \\
t&=1,\ldots, n, \\
\mathbf o_t^\top&= \mathbf q_t^\top\mathbf s_t \in \mathbb R^{e}, \\
\bar {\mathbf s} &= \mathbf s_n.
\end{aligned}
$$
如果reverse = true:
$$
\begin{aligned}
\mathbf {s}_{n+1} &= \mathbf {s} ,  \\
\mathbf {s}_n  &= \mathbf {s}_{n+1} + \mathbf{k}_n\mathbf {v}^\top_n, \\

\mathbf {s}_t &= [\lambda_{t+1}\gamma_{t+1}^\top] \odot \mathbf{s}_{t+1} + \mathbf{k}_t\mathbf {v}^\top_t, \\
t&=1,\ldots, n- 1, \\
\mathbf {s}_0&= [\lambda_{1}\gamma_{1}^\top] \odot\mathbf {s}_1, \\
\bar{\mathbf s}& = \mathbf s_0.

\end{aligned}
$$
那么：
$$
\begin{aligned}
\mathbf O,  {\mathbf s}_n &= f(\mathbf Q, \mathbf K, \mathbf V, \Lambda, \Gamma, \mathbf s, \mathrm{false}), \\

\mathbf {dQ}, {\mathbf s}_n &= f(\mathbf {dO}, \mathbf V, \mathbf K, \Gamma, \Lambda, \mathbf s, \mathrm{false}), \\

\mathbf {dK},  {\mathbf {ds}_0} &= f(\mathbf {V}, \mathbf {dO}, \mathbf Q, \Gamma, \Lambda, \mathbf {ds}, \mathrm{true}), \\

\mathbf {dV},  {\mathbf {ds}_0} &= f(\mathbf {K}, \mathbf Q,\mathbf {dO} , \Lambda, \Gamma, \mathbf {ds}, \mathrm{true}).
\end{aligned}
$$
所以我们可以用一个函数解决前向反向计算的问题。
