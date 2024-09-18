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
- 240912
  - [ ] add md_lrpe document;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] fuse act + lrpe;
    - [x] softmax
  - [ ] test fused act + lrpe + linear attention with output gate
- 240913
  - [ ] add md_lrpe document;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [ ] add mask for lrpe sp 1d
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
- 240914
  - [ ] add md_lrpe document;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [ ] add mask for lrpe sp 1d
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
  - [ ] custom benchmark function
- 240918
  - [x] add md_lrpe cosine document;
  - [ ] add lrpe cosine 2d;
  - [ ] add lrpe cosine 3d;
  - [x] add feature mask for lrpe cosine 1d;
  - [x] add feature mask for lrpe cosine md;
    - [x] triton
    - [x] triton cache
  - [ ] add act for lrpe md;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] add mask for lrpe sp 1d
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
  - [ ] custom benchmark function

# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
```

Symbol Explanation: When benchmarking, use `o` to represent the output of the function. For the function name, use `fn_version` where `fn` is the function name and `version` can be either `torch` or `triton`.
