# Kernel Regression

给定输入$ \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}, \mathbf \Beta \in \mathbb R^n$，以及Decay $\Lambda\in \mathbb R^{n}$，kernel function $\phi$, 记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
 \alpha_j /\alpha_i, & i < j.
\end{cases}
$$
计算：
$$
\mathbf O = [\mathbf I + \mathrm{diag}(\Beta) [\phi(\mathbf K \mathbf K^\top) \odot \mathbf M] ]^{-1} \mathbf V.
$$

实际使用中，$\phi$常用$\exp$ kernel和dot kernel，另外我们还会增加causal的约束，即使用：
$$
\mathrm{tril}(\phi(\mathbf K \mathbf K^\top) \odot \mathbf M, -1).
$$
