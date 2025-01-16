
# Lrpe Rotate 1d

## Forward Propagation

**Forward Input:**
$$
\mathbf X \in \mathbb R^{n\times d }, \mathbf \theta \in \mathbb R^{d/2},
\mathrm{offset} \in \mathbb N, \\
\text{act: Name of the activation function}, \text{dim: Dimension on which the activation function is applied}.
$$

**Forward Output:**
$$
\mathbf O \in \mathbb R^{n\times d}.
$$

**Computation:**
$$
\begin{aligned}
{\mathbf {\bar X}} &= f_{\text{act}}(\mathbf X, \mathrm{dim}), \\
\mathbf o_t^\top &= \bar {\mathbf x}_t^\top \mathbf W_t , \\
\mathbf W_t(\Theta) &= \text{block-diag}\{\mathbf W_{t,1},\ldots, \mathbf W_{t,d/2}\} \in \mathbb R^{d\times d},  \\
\mathbf W_{t,k} &= \left[\begin{array}{cc}
\cos \left((t+\mathrm{offset}) \theta_k\right) & -\sin \left((t+\mathrm{offset})\theta_k\right) \\
\sin \left((t+\mathrm{offset})\theta_k\right) & \cos \left((t+\mathrm{offset}) \theta_k\right)
\end{array}\right] \in \mathbb R^{2\times 2}.
\end{aligned}
$$

Notice that:
$$
\begin{aligned}
\left[\begin{array}{cc}
\cos \left(\theta\right) & -\sin \left(\theta\right) \\
\sin \left(\theta\right) & \cos \left(\theta\right)
\end{array}\right]
\left[\begin{array}{c}
x_1 \\
x_2
\end{array}\right]
&=
\left[\begin{array}{c}
x_1 \cos \theta - x_2 \sin \theta \\
x_2 \cos \theta + x_1 \sin \theta
\end{array}\right]
\end{aligned}
$$

For simplicity and implementation convenience, we assume:
$$
\mathbf X = [\mathbf X_1, \mathbf X_2], \mathbf X_i \in \mathbb R^{n\times d/2}.
$$

The final output is:
$$
\begin{aligned}
\mathbf O_1 &= \mathbf X_1 \odot \cos \theta - \mathbf X_2 \odot \sin \theta, \\
\mathbf O_2 &= \mathbf X_1 \odot \sin \theta + \mathbf X_2 \odot \cos \theta, \\
\mathbf O &= [\mathbf O_1, \mathbf O_2].
\end{aligned}
$$



## Backward Propagation

**Backward Input:**
$$
\mathbf {dO} \in \mathbb R^{n\times d}.
$$

**Backward Output:**
$$
\mathbf {dX} \in \mathbb R^{n\times d}.
$$

Here, $n$ represents the sequence length, $d$ represents the head dimension, and $\mathrm{offset}$ is used only during the inference phase of language models.

**Computation:**
$$
\begin{aligned}
\mathbf {dx}^\top_t &= {\mathbf o}_t^\top \mathbf W_t^\top \\
  &= {\mathbf o}_t^\top \mathbf W_t^\top \\
 &= {\mathbf o}_t^\top \mathbf W_t(\Theta)^\top \\
  &= {\mathbf o}_t^\top \mathbf W_t(-\Theta), \\
\mathbf {d X} &= f'_{\text{act}}(\mathbf{d{\bar X}}, \mathrm{dim}).
\end{aligned}
$$

We assume:
$$
\mathbf {dO} = [\mathbf {dO}_1, \mathbf {dO}_2], \mathbf {dO}_i \in \mathbb R^{n\times d/2}.
$$

The final output is:
$$
\begin{aligned}
\mathbf {dX}_1 &= \mathbf {dO}_1 \odot \cos \theta + \mathbf {dO}_2 \odot \sin \theta, \\
\mathbf {dX}_2 &= -\mathbf {dO}_1 \odot \sin \theta + \mathbf {dO}_2 \odot \cos \theta, \\
\mathbf {dX} &= [\mathbf {dX}_1, \mathbf {dX}_2].
\end{aligned}
$$



## Additional Notes

During implementation, we also support the following features:

- Since the concept of "head" is present, we support $\theta$ shapes of $(h, d/2)$, $(d/2)$, and $(h)$.
- We allow the rope operation to be applied to only specific head dimensions.
