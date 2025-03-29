# Chunk cumsum scalar decay benchmark

## A100

### Reverse = True

#### Speed
```
chunk_cumsum_scalar_decay-speed-batch4-h16-reverse_False-bf16:
        n    triton     torch  torch_compile       fla
0  1024.0  0.011578  0.113031       0.202076  0.009474

chunk_cumsum_scalar_decay-speed-batch16-h16-reverse_False-bf16:
        n    triton     torch  torch_compile       fla
0  1024.0  0.012485  0.118118       0.180217  0.013568
```

#### Memory
```
chunk_cumsum_scalar_decay-memory-batch4-h16-reverse_False-bf16:
        n  triton  torch  torch_compile    fla
0  1024.0    0.25  0.875            0.5  0.375

chunk_cumsum_scalar_decay-memory-batch16-h16-reverse_False-bf16:
        n  triton  torch  torch_compile  fla
0  1024.0     1.0    3.5            2.0  1.5
```

### Reverse = False

#### Speed
```
chunk_cumsum_scalar_decay-speed-batch4-h16-reverse_True-bf16:
        n   triton     torch  torch_compile       fla
0  1024.0  0.00889  0.251649       0.066474  0.009117

chunk_cumsum_scalar_decay-speed-batch16-h16-reverse_True-bf16:
        n    triton     torch  torch_compile       fla
0  1024.0  0.013078  0.265088        0.09563  0.013726
```

#### Memory
```
chunk_cumsum_scalar_decay-memory-batch4-h16-reverse_True-bf16:
        n  triton     torch  torch_compile    fla
0  1024.0    0.25  0.907227            0.5  0.375

chunk_cumsum_scalar_decay-memory-batch16-h16-reverse_True-bf16:
        n  triton     torch  torch_compile  fla
0  1024.0     1.0  3.626465            2.0  1.5
```
