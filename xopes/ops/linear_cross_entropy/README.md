# Linear Cross Entropy

Given the input $\mathbf{x} \in \mathbb{R}^d, \mathbf{W} \in \mathbb{R}^{v \times d}, \mathbf{b} \in \mathbb{R}^v$, a one-hot label $\mathbf{y} \in \mathbb{R}^v, y_k = 1$, and a smoothing parameter $\lambda \in [0, 1]$, define:
$$
\mathbf{\bar{y}} = (1-\lambda) \mathbf{y} + \lambda / e \mathbf{1}, \\
\mathbf{1}^\top \mathbf{\bar{y}} = (1-\lambda) + \lambda = 1.
$$

Output:
$$
\begin{aligned}
\mathbf{z} &= \mathbf{W} \mathbf{x} + \mathbf{b}, \\
r &= \log\left( \sum_{j=1}^v \exp(z_j) \right), \\
o &= - \sum_{i=1}^v \bar{y}_i \left(z_i - r \right), \\
&= - \sum_{i=1}^v \left( (1-\lambda) y_i + \lambda / v \right) \left(z_i - r \right), \\
&= - (1-\lambda) z_k + r - \lambda / v \left( \sum_{i=1}^v z_i \right).
\end{aligned}
$$

## Forward

1. Initialize $m = 0, \text{sse} = 0, s = 0, z = 0$.
2. Compute $\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b} \in \mathbb{R}^v$.
3. Compute $m = \max(\mathbf{z})$.
4. Compute $\text{sse} = \sum_{j=1}^v \exp(y_{i,j} - m)$.
5. Compute $s = \sum_{j=1}^v z_{ij}$.
6. Compute $z = z_k$.
7. Set $c = y_k \neq \text{ignore}$ for reduction.
8. Return $-(1-\lambda) z + m + \log(\text{sse}) - \lambda / v s$.

## Backward

Input: $\mathbf{do} \in \mathbb{R}$.

Compute:
$$
\begin{aligned}
p_k &= \exp(z_k - r), \\
\frac{\partial o}{\partial z_i} &= -(1-\lambda) 1_{i=k} + p_i - \lambda / v, \\
\frac{\partial \mathbf{o}}{\partial \mathbf{z}} &= -(1-\lambda) \mathbf{y} + \mathbf{p} - \lambda / v, \\
\mathbf{dz} &= \mathbf{do} \frac{\partial \mathbf{o}}{\partial \mathbf{z}} \in \mathbb{R}^v, \\
\mathbf{dx} &= \mathbf{dz} \mathbf{W}^\top, \\
\mathbf{dW} &= (\mathbf{dz}) \mathbf{x}^\top, \\
\mathbf{db} &= \mathbf{dz}.
\end{aligned}
$$

For $\mathbf{dz}$:
$$
\mathbf{dz} = \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\text{one-hot}} + \mathbf{p} - \lambda / v \right).
$$

Thus:
$$
\begin{aligned}
\mathbf{dx} &= \mathbf{dz} \mathbf{W}^\top, \\
\mathbf{dW} &= (\mathbf{dz}) \mathbf{x}^\top, \\
\mathbf{db} &= \mathbf{dz}.
\end{aligned}
$$

Algorithm:
1. Compute $\mathbf{p} = \exp(\mathbf{z} - \text{lse})$.
2. Compute $\mathbf{dz} = \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\text{one-hot}} + \mathbf{p} - \lambda / v \right)$.
3. Compute $\mathbf{dx} = \mathbf{dz} \mathbf{W}^\top$.
4. Compute $\mathbf{dW} = (\mathbf{dz}) \mathbf{x}^\top$.
5. Compute $\mathbf{db} = \mathbf{dz}$.
6. Return $\mathbf{dx}, \mathbf{dW}, \mathbf{db}$.

---

## Proposal: Vocabulary Partitioning (Exploratory Version, Deprecated)

### Forward

Input: $\mathbf{x} \in \mathbb{R}^d, \mathbf{W} \in \mathbb{R}^{v \times d}, \mathbf{b} \in \mathbb{R}^v$, one-hot label $\mathbf{y} \in \mathbb{R}^v, y_t = 1$, and smoothing parameter $\lambda \in [0, 1]$.

#### Recursive Version

Assume:
$$
\mathbf{W} =
\begin{bmatrix}
\mathbf{W}_1 \\
\vdots \\
\mathbf{W}_k
\end{bmatrix}, \;
\mathbf{b} =
\begin{bmatrix}
\mathbf{b}_1 \\
\vdots \\
\mathbf{b}_k
\end{bmatrix}, \;
\mathbf{y} =
\begin{bmatrix}
\mathbf{y}_1 \\
\vdots \\
\mathbf{y}_k
\end{bmatrix}.
$$

1. Initialize $m = 0, \text{sse} = 0, s = 0, z = 0$.
2. For $i = 1, \ldots, k$:
   - Compute $\mathbf{z}_i = \mathbf{W}_i \mathbf{x} \in \mathbb{R}^{v/k}$.
   - Compute $m_i = \max(\mathbf{z}_i)$.
   - Update $m' = \max(m_i, m)$.
   - Update $\text{sse}_i = \sum_{j=1}^{v/k} \exp(y_{i,j} - m')$.
   - Update $\text{sse} = \exp(m - m') \text{sse} + \text{sse}_i$.
   - Update $m = m'$.
   - Compute $s_i = \sum_{j=1}^{v/k} z_{ij}$.
   - Update $s = s + s_i$.
   - If $t \in [iv/k + 1, (i+1)v/k]$, update $z = z_{i, t - iv/k}$.
3. Return $-(1-\lambda) z + m + \log(\text{sse}) - \lambda / v s$.

#### Parallel Version

1. Initialize $m = 0, \text{sse} = 0, s = 0, z = 0$.
2. For $i = 1, \ldots, k$, compute in parallel:
   - $\mathbf{z}_i = \mathbf{W}_i \mathbf{x} \in \mathbb{R}^{v/k}$.
   - $m_i = \max(\mathbf{z}_i)$.
   - $\text{sse}_i = \sum_{j=1}^{v/k} \exp(y_{i,j} - m')$.
   - $s_i = \sum_{j=1}^{v/k} z_{ij}$.
   - If $t \in [iv/k + 1, (i+1)v/k]$, update $z = z_{i, t - iv/k}$.
3. For $i = 1, \ldots, k$:
   - Update $m' = \max(m_i, m)$.
   - Update $\text{sse} = \exp(m - m') \text{sse} + \exp(m_i - m') \text{sse}_i$.
   - Update $m = m'$.
   - Update $s = s + s_i$.
4. Return $-(1-\lambda) z + m + \log(\text{sse}) - \lambda / v s$.


### Backward

**Input**: $\mathbf{do} \in \mathbb{R}$.

**Computation**:
$$
\begin{aligned}
p_k &= \exp(z_k - r), \\
\frac{\partial o}{\partial z_i}
&= -(1-\lambda)\frac{\partial z_k}{\partial z_i} + \frac{\partial r}{\partial z_i} - \frac{\lambda}{v} \frac{\partial \left( \sum_{i=1}^v z_i \right)}{\partial z_i} \\
&= -(1-\lambda) \delta_{i=k} + p_i - \frac{\lambda}{v}, \\
\frac{\partial \mathbf{o}}{\partial \mathbf{z}}
&= -(1-\lambda) \mathbf{y} + \mathbf{p} - \frac{\lambda}{v}, \\
\mathbf{dz} &= \mathbf{do} \frac{\partial \mathbf{o}}{\partial \mathbf{z}} \in \mathbb{R}^v, \\
\mathbf{dx} &= \mathbf{dz} \mathbf{W}^\top, \\
\mathbf{dW} &= (\mathbf{dz}) \mathbf{x}^\top, \\
\mathbf{db} &= \mathbf{dz}.
\end{aligned}
$$

For $\mathbf{dz}$:
$$
\begin{aligned}
\mathbf{dz}
&= \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\mathrm{one-hot}} + \mathbf{p} - \frac{\lambda}{v} \right) \\
&\triangleq \mathbf{dz}_1 + \mathbf{dz}_2, \\
\mathbf{dz}_1 &= \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\mathrm{one-hot}} - \frac{\lambda}{v} \right), \\
\mathbf{dz}_2 &= \mathbf{do} \odot \mathbf{p}.
\end{aligned}
$$

Thus:
$$
\begin{aligned}
\mathbf{dx} &= \mathbf{dz} \mathbf{W}^\top \\
&= \mathbf{dz}_1 \mathbf{W}^\top + \mathbf{dz}_2 \mathbf{W}^\top, \\
&= \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\mathrm{one-hot}} \mathbf{W}^\top - \frac{\lambda}{v} \mathbf{W}^\top + \mathbf{p} \mathbf{W}^\top \right), \\
\mathbf{dW} &= (\mathbf{dz}) \mathbf{x}^\top \\
&= (\mathbf{dz}_1) \mathbf{x}^\top + (\mathbf{dz}_2) \mathbf{x}^\top, \\
&= \mathbf{do} \left( -(1-\lambda) \mathbf{y}_{\mathrm{one-hot}} \mathbf{x}^\top - \frac{\lambda}{v} \mathbf{x}^\top + \mathbf{p} \mathbf{x}^\top \right), \\
\mathbf{db} &= \mathbf{dz}.
\end{aligned}
$$

---

**Algorithm**:

For $\mathbf{dz}_1$, computation can be performed directly without recursion.

For $\mathbf{dz}_2$ (ignoring the scalar $\mathbf{do}$ for simplicity), note the following relationship (considering chunk recursion): for the $j$-th chunk, the local log-sum-exp (LSE) is denoted as $\mathrm{lse}_j$, and the cumulative LSE up to the $j$-th chunk is $\mathrm{lse}^j$. Then:
$$
\begin{aligned}
\exp(\mathrm{lse}^j) &= \exp(\mathrm{lse}^{j-1}) + \exp(\mathrm{lse}_j), \\
\lambda_j &= \frac{\exp(\mathrm{lse}^{j-1})}{\exp(\mathrm{lse}^j)} \\
&= \frac{\exp(\mathrm{lse}^{j-1})}{\exp(\mathrm{lse}^{j-1}) + \exp(\mathrm{lse}_j)}.
\end{aligned}
$$

Supplement:
$$
\begin{aligned}
\prod_{j=1}^s \lambda_j
&= \prod_{j=1}^s \frac{\exp(\mathrm{lse}^{j-1})}{\exp(\mathrm{lse}^j)} \\
&= \frac{\exp(\mathrm{lse}^0)}{\exp(\mathrm{lse}^s)} \\
&= \exp(-\mathrm{lse}^s).
\end{aligned}
$$

For the probabilities $\mathbf{p}_j \in \mathbb{R}^{v/k}$ within the $j$-th chunk, accumulated globally as $\mathbf{p}_i^j \in \mathbb{R}^{v/k}, i=1, \ldots, k-1$, we have:
$$
\begin{aligned}
\mathbf{p}_i^j &= \frac{\exp(\mathbf{z}_i)}{\exp(\mathrm{lse}^j)} \\
&= \frac{\exp(\mathbf{z}_i)}{\exp(\mathrm{lse}^{j-1})} \times \frac{\exp(\mathrm{lse}^{j-1})}{\exp(\mathrm{lse}^j)} \\
&= \lambda_j \mathbf{p}_i^{j-1}, \, j > i, \\
\mathbf{p}_j &= \frac{\exp(\mathbf{z}_j)}{\exp(\mathrm{lse}_j)}, \\
\mathbf{p}_j^j &= \frac{\exp(\mathbf{z}_j)}{\exp(\mathrm{lse}^j)} \\
&= \frac{\exp(\mathbf{z}_j)}{\exp(\mathrm{lse}_j)} \times \frac{\exp(\mathrm{lse}_j)}{\exp(\mathrm{lse}^j)} \\
&= \mathbf{p}_j (1 - \lambda_j).
\end{aligned}
$$

For $\mathbf{dz}_2 \mathbf{W}^\top$ at the $j$-th chunk, denoted as $\mathbf{du}^j$:
$$
\begin{aligned}
\mathbf{du}^j
&= \lambda_j \mathbf{du}^{j-1} + (1-\lambda_j) \mathbf{p}_j \mathbf{W}_j^\top \\
&\triangleq \lambda_j \mathbf{du}^{j-1} + (1-\lambda_j) \mathbf{du}_j.
\end{aligned}
$$

Proof:
$$
\begin{aligned}
\mathbf{du}^j
&= [\mathbf{p}_1^j, \ldots, \mathbf{p}_j^j][\mathbf{W}_1, \ldots, \mathbf{W}_j]^\top \\
&= \sum_{i=1}^j \mathbf{p}_i^j \mathbf{W}_i^\top \\
&= \sum_{i=1}^{j-1} \mathbf{p}_i^j \mathbf{W}_i^\top + \mathbf{p}_j^j \mathbf{W}_j^\top \\
&= \lambda_j \mathbf{du}^{j-1} + (1-\lambda_j) \mathbf{p}_j \mathbf{W}_j^\top.
\end{aligned}
$$

For $\mathbf{x}^\top \mathbf{dz}_2$ at the $j$-th chunk, denoted as $\mathbf{dv}^j \in \mathbb{R}^{d \times je}$:
$$
\begin{aligned}
\mathbf{dv}^j
&= [\lambda_j \mathbf{dv}^{j-1}, (1-\lambda_j) \mathbf{x}^\top \mathbf{p}_j] \\
&= [\lambda_j \mathbf{dv}^{j-1}, (1-\lambda_j) \mathbf{dv}_j].
\end{aligned}
$$

Proof:
$$
\begin{aligned}
\mathbf{dv}_i^j
&= \mathbf{x}^\top \mathbf{p}_i^j \\
&= \mathbf{x}^\top (\lambda_j \mathbf{p}_i^{j-1}) \\
&= \lambda_j \mathbf{dv}_i^{j-1}, \, i \leq j-1, \\
\mathbf{dv}_j^j
&= \mathbf{x}^\top \mathbf{p}_j^j \\
&= (1-\lambda_j) \mathbf{x}^\top \mathbf{p}_j.
\end{aligned}
$$

Expanding:
$$
\begin{aligned}
\mathbf{dv}_i^k
&= \mathbf{dv}_i (1-\lambda_i) \prod_{j=i+1}^k \lambda_j \\
&= \mathbf{dv}_i (1-\lambda_i) \frac{\exp(-\mathrm{lse}^k)}{\exp(-\mathrm{lse}^i)} \\
&= \mathbf{dv}_i (1-\lambda_i) \exp(\mathrm{lse}^i - \mathrm{lse}^k).
\end{aligned}
$$

Note: The recursion for $\mathbf{dv}$ cannot be directly used in batch mode due to different decays across batches.
