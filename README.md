# Xopes

Xopes: Toolbox for Accelerating Deep Learning Operators.

# Plan
- oplr
- lce
- lcse
- ce
  - [x] 250107
- ewbo;
  - [x] 250107
- logsumexp
  - [x] 250103
- householder
  - [x] 250101
  - [x] 250111 update
- normalize
  - [x] 241231
  - [x] 250111 fix bug
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
  - [x] add feature mask for lrpe cosine 1d;
  - [x] add feature mask for lrpe cosine md;
    - [x] triton
    - [x] triton cache
  - [x] add act for lrpe cosine md;
    - [x] triton
    - [x] triton cache
    - [ ] left softmax + dim = -2
  - [x] benchmark lrpe cosine md with act;
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] add mask for lrpe sp 1d
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
  - [ ] custom benchmark function
- 240919
  - [x] add act for lrpe cosine md;
    - [x] triton
    - [x] triton cache
    - [x] triton block parallel
      - [x] left softmax + dim = -2
        - [ ] left bwd
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
    - [ ] lrpe cosine md
  - [ ] custom benchmark function
- 240923
  - [x] add act for lrpe cosine md;
    - [x] triton block parallel
      - [ ] left bwd
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
    - [ ] lrpe cosine md
  - [ ] custom benchmark function
  - [ ] multinomial
    - [x] torch
    - [x] torch online
    - [ ] triton online
    - [ ] triton parallel
- 240924
  - [x] add act for lrpe cosine md;
    - [x] triton block parallel
      - [ ] left bwd
  - [ ] add act;
    - [ ] add jit act
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
      - [ ] https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/activations.py
  - [x] test fused act + lrpe + linear attention with output gate
    - [ ] left softmax + dim = -2
    - [ ] lrpe cosine md
  - [ ] custom benchmark function
  - [ ] multinomial
    - [x] triton online
    - [x] triton parallel
    - [x] triton parallel gumbel
    - [ ] document
- 240929
  - [ ] multinomial
    - [ ] triton parallel gumbel small vocab bug
    - [ ] unify input shape
    - [ ] reduce kernel bug
    - [ ] online_gumbel_multinomial_torch

# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
[Test Add]
```

Symbol Explanation: When benchmarking, use `o` to represent the output of the function. For the function name, use `fn_version` where `fn` is the function name and `version` can be either `torch` or `triton`.
