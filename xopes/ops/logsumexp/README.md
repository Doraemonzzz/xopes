
# Log Sum Exp

For the input $\mathbf{x} \in \mathbb{R}^d$, compute:
$$
\begin{aligned}
\mathbf{o}
 = \mathrm{lse}(\mathbf{x}) =\log\left(
\sum_{j=1}^d \exp(x_j)
\right) .
\end{aligned}
$$

Supplementary definition:
$$
\mathrm{se}(\mathbf{x})=
\sum_{j=1}^d \exp(x_j).
$$
Thus:
$$
\mathbf{o}= \log\mathrm{se}(\mathbf{x}).
$$

## Forward

Input: $\mathbf{x} \in \mathbb{R}^d$.

We define:
$$
\begin{aligned}
f(\mathbf{x}) &\geq \max_{i=1}^d \{x_1,\ldots, x_d \}, \\
\mathbf{o}
 & = \mathrm{lse}(\mathbf{x}) \\
 &= \log\left(
\sum_{j=1}^d \exp(x_j)
\right)  \\
&=\log\left(
\sum_{j=1}^d \exp(x_j -f(\mathbf{x}))
\right) + f(\mathbf{x}) \\
&\triangleq \mathrm{slse}(\mathbf{x}) +  f(\mathbf{x}), \\
\mathrm{se}(\mathbf{x}) & = \exp(\mathrm{slse}(\mathbf{x}) +  f(\mathbf{x})) \\
&= \exp(\mathrm{slse}(\mathbf{x})) \exp(f(\mathbf{x})) \\
&\triangleq \mathrm{sse}(\mathbf{x}) \exp(f(\mathbf{x})),\\
\mathrm{lse}(\mathbf{x})
&= \log(\mathrm{sse}(\mathbf{x})) + f(\mathbf{x}).

\end{aligned}
$$
Here, `slse` is short for stable log sum exp, and `sse` is short for stable sum exp.

Given $\mathbf{x}_1 \in \mathbb{R}^{d_1}, \mathbf{x}_2 \in \mathbb{R}^{d_2}, \mathbf{x}=[\mathbf{x}_1, \mathbf{x}_2]\in \mathbb{R}^{d_1+d_2}=\mathbb{R}^{d}$, note:
$$
\begin{aligned}

\mathrm{lse}(\mathbf{x}) &=\log\left(
\sum_{j=1}^d \exp(x_j)
\right) \\
&=  \log\left(
\sum_{j=1}^{d_1} \exp(x_j)
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j)
\right) \\
&= \log\left(
\exp(\mathrm{lse}(\mathbf{x}_1))
+ \exp(\mathrm{lse}(\mathbf{x}_2))
\right) \\

&= \log\left(
\exp(\mathrm{lse}(\mathbf{x}_1)-f(\mathbf{x}))
+ \exp(\mathrm{lse}(\mathbf{x}_2)-f(\mathbf{x}))
\right) +f(\mathbf{x}) \\

&= \log\left(
\exp(\mathrm{slse}(\mathbf{x}_1)+f(\mathbf{x}_1)-f(\mathbf{x}))
+ \exp(\mathrm{slse}(\mathbf{x}_2)+f(\mathbf{x}_2)-f(\mathbf{x}))
\right)+f(\mathbf{x}) \\

f(\mathbf{x})&=\max(f(\mathbf{x}_1),f(\mathbf{x}_2)).
\end{aligned}
$$
Thus, block-wise recursion/parallelism can be used to accelerate forward computation. However, note that merging blocks requires operations such as `exp`, `add`, and `log`, which introduce additional computational overhead. To address this, we use the $\mathrm{sse}$ function:
$$
\begin{aligned}

\mathrm{sse}(\mathbf{x}) &=
\sum_{j=1}^d \exp(x_j-f(\mathbf{x}))
\\
&=
\sum_{j=1}^{d_1} \exp(x_j-f(\mathbf{x}))
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j-f(\mathbf{x})) \\

&=
\sum_{j=1}^{d_1} \exp(x_j-f(\mathbf{x}_1))\exp(f(\mathbf{x}_1 )-f(\mathbf{x}))
+ \sum_{j=d_1+1}^{d_1+d_2} \exp(x_j-f(\mathbf{x}_2))\exp(f(\mathbf{x}_2 )-f(\mathbf{x})) \\
&= \exp(f(\mathbf{x}_1 )-f(\mathbf{x})) \mathrm{sse}(\mathbf{x}_1) + \exp(f(\mathbf{x}_2 )-f(\mathbf{x}))\mathrm{sse}(\mathbf{x}_2)  \\

f(\mathbf{x})&=\max(f(\mathbf{x}_1),f(\mathbf{x}_2)).
\end{aligned}
$$
The following algorithms are proposed:

Assume $\mathbf{x}= [\mathbf{x}_1, \ldots, \mathbf{x}_k]\in \mathbb{R}^{kd}$.

### Recursive Version

- Initialize $m=0, \mathrm{sse}=0$;
- For $i=1,\ldots ,k$:
  - $m_i =\max(\mathbf{x}_i)$;
  - $m'=\max(m_i, m)$;
  - $\mathrm{sse}_i= \sum_{j=1}^d \exp(x_{i,j}-m')$;
  - $\mathrm{sse}= \exp(m-m') \mathrm{sse} + \mathrm{sse}_i$;
  - $m=m'$;
- Return $m, \mathrm{sse}$.

### Parallel Version

- Initialize $m=0, \mathrm{sse}=0$;
- For $i=1,\ldots ,k$, compute in parallel:
  - $m_i =\max(\mathbf{x}_i)$;
  - $\mathrm{sse}_i= \sum_{j=1}^d \exp(x_{i,j}-m_i)$;
- For $i=1,\ldots, k$:
  - $m'=\max(m_i, m)$;
  - $\mathrm{sse}= \exp(m-m') \mathrm{sse} + \exp(m_i-m') \mathrm{sse}_i$;
  - $m=m'$;
- Return $m, \mathrm{sse}$.

### Stability Analysis

Since $\exp(m-m')\leq 1, \exp(m_i-m')\leq 1, \exp(x_{i,j}-m_i)\leq 1$, each operation is numerically stable.

## Backward

Input: $\mathbf{do}\in \mathbb R$.

Compute:
$$
\begin{aligned}
p_{i}&= \exp(x_i - \mathbf{o}), \\
\frac{\partial o}{\partial x_i}
&= p_i, \\
\mathbf{dx}&= \mathbf{do} \odot \mathbf{p}.
\end{aligned}
$$
