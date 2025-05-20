

# Linear Implicit Attention

给定输入$\mathbf Q\in \mathbb R^{n\times d}, \mathbf K\in \mathbb R^{n\times d}, \mathbf V\in \mathbb R^{n\times e}$，初起始state $\mathbf s_0$，以及Decay $\Lambda\in \mathbb R^{n}$，记：
$$
\mathbf M_{ij}=
\begin{cases}
\prod_{t=j+1}^i \Lambda_t \triangleq  \alpha_i /\alpha_j, & i \ge j, \\
0, & i < j.
\end{cases}
$$
考虑Linear Attention：
$$
\begin{aligned}
&\mathbf O =\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right] \mathbf V + \Alpha \mathbf Q \mathbf S_0, \\
&\mathbf O-\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right] \mathbf V + \Alpha \mathbf Q \mathbf S_0 =0.
\end{aligned}
$$
假设$\mathbf A, \mathbf S_0$给定，那么该等式涉及$\mathbf Q, \mathbf K, \mathbf V, \mathbf O$，将其看成隐函数，那么我们可以给出四个形式：

- 给定$\mathbf Q, \mathbf K, \mathbf V$，输出$\mathbf O$；
  - 这种形式即为Linear Attention；
- 给定$\mathbf Q, \mathbf K, \mathbf O$，输出$\mathbf V$；
  - 称为Type1；
- 给定$\mathbf Q, \mathbf V, \mathbf O$，输出$\mathbf K$；
  - 称为Type2；
- 给定$\mathbf K, \mathbf V, \mathbf O$，输出$\mathbf Q$；
  - 称为Type3；



## Type1

此时形式为：
$$
\begin{aligned}
\mathbf V &=\left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right]^{-1} \left( \mathbf O - \Alpha \mathbf Q \mathbf S_0 \right)  \\
&\triangleq \left[
\left( \mathbf Q \mathbf K^\top\right) \odot \mathbf M
\right]^{-1}  \mathbf V.
\end{aligned}
$$
为了方便叙述，我们将最后一项仍然记为$\mathbf O $：



## Type2

此时形式为：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top, \\
\mathbf o_t^\top  &=  \mathbf q_t^\top \mathbf s_t.

\end{aligned}
$$
给定$\mathbf Q, \mathbf V, \mathbf O $，无法求解$\mathbf K$，这是因为State $\mathbf s_t$未知，假设$\mathbf s_t$已知，那么：
$$
\mathbf k_t =(\mathbf s_t - \lambda_t \mathbf s_{t-1})\mathbf v_t / (\mathbf v_t^\top \mathbf v_t)
=(\mathbf s_t - \lambda_t \mathbf s_{t-1})\mathbf v_t / \| \mathbf v_t \|^2.
$$
所以问题转化成求$\mathbf s_t$，我们考虑如下问题：
$$
\min_{\mathbf s_t} \frac 1 2 \| \mathbf o_t - \mathbf s_t^\top \mathbf q_t \|^2.
$$
求导后可得：
$$
\begin{aligned}
(  \mathbf s_t^\top \mathbf q_t -\mathbf o_t)\mathbf q_t^\top  &= 0  \\
 \mathbf q_t \mathbf q_t^\top \mathbf s_t &= \mathbf q_t \mathbf o_t^\top \\
  \mathbf s_t &= ( \mathbf q_t \mathbf q_t^\top )^{-1}  \mathbf q_t \mathbf o_t^\top
\end{aligned}
$$
这里的逆为伪逆，注意到（伪逆定义）：
$$
(\mathbf q_t \mathbf q_t^\top) (\mathbf q_t \mathbf q_t^\top / \| \mathbf q_t\|^4) (\mathbf q_t \mathbf q_t^\top)= \mathbf q_t \mathbf q_t^\top,
$$
那么：
$$
\mathbf s_t = ( \mathbf q_t \mathbf q_t^\top )^{-1}  \mathbf q_t \mathbf o_t^\top =
\mathbf q_t \mathbf q_t^\top / \| \mathbf q_t\|^4  \mathbf q_t \mathbf o_t^\top = \mathbf q_t \mathbf o_t^\top / \| \mathbf q_t\|^2.
$$
因此我们的递推公式为：
$$
\begin{aligned}
\mathbf k_t & =(\mathbf s_t - \lambda_t \mathbf s_{t-1})\mathbf v_t / (\mathbf v_t^\top \mathbf v_t) \\

&= \left(
\mathbf q_t \mathbf o_t^\top / \| \mathbf q_t\|^2 - \lambda_t \mathbf q_{t-1} \mathbf o_{t-1}^\top / \| \mathbf q_{t-1}\|^2
\right)\mathbf v_t /  \| \mathbf v_t \|^2.
\end{aligned}
$$
这种形式虽然复杂，暂时没有递推，所以求解起来反而最简单。





## Type3

此时形式为：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top, \\
\mathbf o_t^\top  &=  \mathbf q_t^\top \mathbf s_t.

\end{aligned}
$$
给定$\mathbf K, \mathbf V, \mathbf O $，求解$\mathbf Q$：
$$
\begin{aligned}
\mathbf s_t &= \lambda_t \mathbf s_{t-1} + \mathbf k_t \mathbf v_t^\top, \\
\mathbf q_t^\top  &= \mathbf o_t^\top  \mathbf s_t^{-1}.

\end{aligned}
$$
此时和Mesa的形式基本一样。
