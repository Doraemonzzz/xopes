
# Cumsum

We explain the forward and backward processes using a diagram.

**Reverse = False:**

**Forward:**
```
x1
x1 + x2
...
x1 + x2 + ... + xn-1
x1 + x2 + ... + xn-1 + xn
```

**Backward:**
```
don + don-1 + ... + do2 + do1
don + don-1 + ... + do2
...
don + don-1
don
```

**Reverse = True:**

**Forward:**
```
xn + xn-1 + ... + x2 + x1
xn + xn-1 + ... + x2
...
xn + xn-1
xn
```

**Backward:**
```
do1
do1 + do2
...
do1 + do2 + ... + don-1
do1 + do2 + ... + don-1 + don
```

## Forward

Given an input $\mathbf{x} \in \mathbb{R}^{n}$ and a reverse flag, compute the output $\mathbf{o} \in \mathbb{R}^{n}$.

- If reverse = False:
  $$
  o_i = \sum_{j=1}^{i} x_j.
  $$

- If reverse = True:
  $$
  o_{i} = \sum_{j=1}^{n-i+1} x_{n+1-j}.
  $$

## Backward

Given an input $\mathbf{dy} \in \mathbb{R}^{n}$ and a reverse flag, compute the output $\mathbf{dx} \in \mathbb{R}^{n}$.

- If reverse = False:
  $$
  dx_i = \sum_{j=1}^{n+1-i} d o_j.
  $$

- If reverse = True:
  $$
  dx_i = \sum_{j=1}^{i} d o_j.
  $$

Therefore, whether it is forward or backward, regardless of whether reverse = True or False, both can be converted into a prefix sum form.
