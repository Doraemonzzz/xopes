# Xopes

Xopes: Toolbox for Accelerating Deep Learning Operators.

# Todo
- [ ] Additive Recurrence
  - [ ] Baseline
  - [ ] Recurrence Triton
    - [ ] fwd
    - [ ] bwd
  - [ ] Block Recurrence Triton
    - [ ] fwd
    - [ ] bwd
  - [ ] Block Parallel Triton
    - [ ] fwd
    - [ ] bwd
- [ ] LogCumsumExp
  - [ ] Recurrence Triton
    - [x] fwd
  - [ ] Block Recurrence Triton
    - [x] fwd
  - [ ] Block Parallel Triton
    - [x] fwd
- [ ] Lrpe
  - [x] Cosine Triton
    - [ ] Document
    - [x] fwd
    - [x] bwd
    - [ ] Add offset
    - [x] Auto config
- [ ] MdLrpe
  - [ ] Readme
  - [x] Cosine Triton
    - [x] fwd
    - [x] bwd
    - [ ] Add offset
    - [x] Auto config
  - [ ] Cosine Parallel Triton
    - [x] fwd
    - [x] bwd
    - [ ] Add offset
    - [x] Auto config
  - [ ] Cosine Cache Triton
    - [x] fwd
    - [x] bwd
    - [ ] Add offset
    - [x] Auto config
- [ ] Tpe
  - [ ] Triton
- [ ] Fuse Linear Attention Output Gate (flao)
  - [x] Non causal
    - [x] Document
    - [x] Lao Torch
    - [x] Flao Torch
    - [x] Flao Triton
      - [x] fwd
      - [x] bwd (in torch since no speed advantage)
      - [x] autotune
    - [ ] Flao Left Product Triton
      - [ ] fwd
      - [ ] bwd
    - [ ] Interface
- [ ] Custom benchmark function
  - [ ] https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
- [ ]

# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
```
