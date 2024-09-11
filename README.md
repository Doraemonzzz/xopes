# Xopes

Xopes: Toolbox for Accelerating Deep Learning Operators.

# Plan

- 240910
  - [x] add fwd_fn, bwd_fn for lrpe, md_lrpe;
  - [x] add test, benchmark for flao_fal;
  - [x] add lrpe document;
- 240911
  - [ ] add md_lrpe document;
  - [x] add act;
    - [x] add softmax
    - [x] relu
    - [x] sigmoid
    - [x] silu
    - [x] none
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] add flao_fal document;
  - [x] clear flao code, add interface;
  - [ ] fuse act + lrpe;
    - [x] relu
    - [x] sigmoid
    - [x] silu
    - [x] none
    - [ ] softmax
- 240911
  - [ ] add md_lrpe document;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [ ] fuse act + lrpe;
    - [ ] softmax
  - [ ] test fused act + lrpe + linear attention with output gate

# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
```

Symbol Explanation: When benchmarking, use `o` to represent the output of the function. For the function name, use `fn_version` where `fn` is the function name and `version` can be either `torch` or `triton`.
