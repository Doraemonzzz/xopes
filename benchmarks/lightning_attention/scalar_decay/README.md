# Lightning Attention with Scalar Decay

## Notation

- lasd_r: Triton recurrence
- flash: Flash attention

## A100

### Speed
```
lasd-speed-fwd-batch4-head32-dim128-bf16:
        n    LASD_R     Flash
0   256.0  0.655428  0.054305
1   512.0  1.281070  0.105579
2  1024.0  2.510729  0.256017
3  2048.0  4.970747  0.791219

lasd-speed-bwd-batch4-head32-dim128-bf16:
        n     LASD_R     Flash
0   256.0   3.077055  0.188792
1   512.0   6.117918  0.385829
2  1024.0  12.221627  0.937125
3  2048.0  24.393583  2.681542
```


### Memory
```
lasd-memory-fwd-batch4-head32-dim128-bf16:
        n      LASD_R       Flash
0   256.0   40.000488   32.125977
1   512.0   72.000488   64.250977
2  1024.0  136.000488  128.500977
3  2048.0  264.000488  257.000977

lasd-memory-bwd-batch4-head32-dim128-bf16:
        n      LASD_R       Flash
0   256.0  102.800488  103.050977
1   512.0  189.600488  206.100977
2  1024.0  363.200488  412.200977
3  2048.0  710.400488  824.400977
```
