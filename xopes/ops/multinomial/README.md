https://stats.stackexchange.com/questions/543536/how-to-sample-n-observations-from-a-multinomial-distribution-using-binomial-o

https://discuss.pytorch.org/t/trying-to-understand-the-torch-multinomial/71643


$$
p_1,\ldots, p_n.
$$

$$
X=YW_v,\\
p=\mathrm{Softmax}(X), Y\in \mathbb R^{b\times d}, W_v\in \mathbb R^{d\times V}.
$$

$$
V\gg d.
$$


$$
Y, W_v,
$$






问题，如何采样多项分布：
$$
bin(k, p_k), \\

\sum_{i=1}^n p_i =1.
$$

## 方案1

- Sample $p\sim U(0, 1)$;
- 找到$k$使得$\sum_{i=1}^k p_i \le p < \sum_{i=1}^{k+1} p_i$;

- 优点：简单直接，第二步速度为$O(\log n)$；
- 缺点；需要知道完整的$p_i$才能采样；





## 方案2

算法：

- $P=1, k=1$
- while True:
  - $x\sim B(p_k/P, 1-p_k/P)$
    - 如果$x=0$，则返回$k$；
  - $P=P-p_k$
  - $k=k+1$

证明：
$$
P_k = 1 - \sum_{j=1}^{k-1} p_j, \\
p(x=k)=p(x_k=0)\prod_{j=1}^{k-1}p(x_j=1)=\frac{p_k}{P_k} \prod_{j=1}^{k-1}(1-p_j /P_j)=
\frac{p_k}{P_k} \prod_{j=1}^{k-1}\frac{P_{j+1}}{P_j}=p_k
$$
假设：
$$
p_k=\frac{\exp(x_k)}{\sum_k \exp(x_k)}
$$
那么：
$$
P_{k}=\frac{{\sum_{j=k}^n \exp(x_j)}}{{\sum_j \exp(x_j)}}
$$


- 缺点；需要知道完整的$p_i$才能采样（错误）；





## 方案3

block算法：

- 假设分为$m$个chunk，每个chunk $c$个元素，$V=mc$；
  - $p_i^j, i=1,\ldots, m, j=1,\ldots, c$；
  - $p_{ij}=\frac{e_{ij}}{\sum_i \sum_j e_{ij}}$；
- 记：
  - $s_i =\sum_{j} e_{ij}$
  - $\bar p_{ij}=\frac{e_{ij}}{s_i}$
  - $\sum_{j}\bar p_{ij}=1$
- for $i=1,\ldots, m$；
  - $y_i \sim B(\bar p_{ij}),j=1,\ldots,c$；
- 构造概率$q_i=\frac{s_i}{\sum s_i}$；
- 采样$z \sim B(q_i)$；
- 选择$y_{z}$；

证明：
$$
P(y_z = (i, j))=P(z=i)P(y_i = j)
=q_i \frac{e_{ij}}{s_i}=\frac{e_{ij}}{\sum_{ij}e_{ij}}
$$

- 优点：可以online算，每次只需要部分信息即可；
- 缺点：类别少的时候可能慢一些；




$$
\frac{e_{ij}}{\sum_{ij} e_{ij}}=\frac{e_{ij}}{\sum_{j} e_{ij}}\frac{{\sum_{j} e_{ij}}}{\sum_{ij} e_{ij}}
$$

### Triton

```
parallel over (b, num group)
batch compute
stable softmax
compute cumsum pi
x = tl.rand
use argmax((x >= p) and (x < p)) to get multinomial(p)

save sample, lse

parallel over (b)

```
