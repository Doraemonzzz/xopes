# Lightning Attention Log Decay with Cumsum fuse

## Notation
- triton_sep: triton implementation with separate cumsum and dld compute
- triton: triton implementation with fused cumsum and dld compute

## A100

### Speed
```
dld-with-cumsum-benchmark-speed-fwd-batch4-head32-dim128-dim128-True--1-bf16:
         n  Triton Sep    Triton
0    256.0    0.321419  0.021331
1    512.0    0.335195  0.025151
2   1024.0    0.329888  0.030708
3   2048.0    0.362412  0.043169
4   4096.0    0.343157  0.065848
5   8192.0    0.336792  0.111494
6  16384.0    0.342721  0.196936
7  32768.0    0.654264  0.369651
```


### Memory
```
dld-with-cumsum-benchmark-memory-fwd-batch4-head32-dim128-dim128-True--1-bf16:
         n  Triton Sep     Triton
0    256.0    8.812988   8.562988
1    512.0    9.625488   9.125488
2   1024.0   11.250488  10.250488
3   2048.0   14.500488  12.500488
4   4096.0   21.000488  17.000488
5   8192.0   34.000488  26.000488
6  16384.0   60.000488  44.000488
7  32768.0  112.000488  80.000488
```
