# Xopes

Xopes: Toolbox for Accelerating Deep Learning Operators.

# Act related kernel
- lrpe
- act
- gate_linear

# Todo
- flce large batch bug.
- lightning attn with element-wise decay.
- update readme.


# Plan
- inverse
  - [x] 250503 add forward substitution method
  - [x] 250503 add jacobian method
  - [x] 250505 add forward substitution/jacobian triton version
  - [ ] Readme
- lightning attn with element-wise recurrence
  - [x] 250430
- lightning attn vector decay
  - [x] 250419
  - [x] 250427 finish fast implementation
- tpa decode 250409
- recurrence state update
  - [ ] lightning attn
- lightning attn log decay
  - [x] 250318
- lightning attn scalar decay data dependent
  - [x] 250315
- logcumsumexp
  - [x] 250313
- lightning attn positional encoding
  - [x] 250310
- add contiguous decorator
- chunk cumsum
  - [x] 250305
- normalize with gate
  - [x] 250301
- lightning attn scalar decay
  - [x] 250227
  - [x] add dld 250309
  - [x] fix bug 250315
  - [ ] varlen
  - [ ] note
- gate_linear
  - [x] 250202
- cumsum
  - [x] 250130
- polar_recurrence vector decay
  - [x] 20250129 add document
- Update b h d -> B H D
  - [x] 250119
- act
  - [x] 250116
- lrpe-cosine
  - [x] 250208 add chunk verion
  - [x] 250115
  - [x] update softmax function
- lrpe-rope
  - [x] 250208 add chunk verion
  - [x] 250116
  - [x] update softmax function
- householder sum update
- oplr
  - [x] 250113 sequential doc/kernel
  - [ ] document
- lce
  - [x] 250120
  - [x] clear kernel
  - [x] 250501 fix index bug
- lcse
- ce
  - [x] 250107
  - [x] 250119 fix nan bug
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
[Enhancement]
[Refactor]
```

Symbol Explanation: When benchmarking, use `o` to represent the output of the function. For the function name, use `fn_version` where `fn` is the function name and `version` can be either `torch` or `triton`.
