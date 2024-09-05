# A800

## Head dim = 128

2d version, sequence length is $n^2$.
```
md_lrpe_cosine-speed-fwd-batch12-head12-dim128-bf16-2d:
       n    Triton      Torch
0   16.0  0.264255   0.507486
1   32.0  0.555315   1.024399
2   64.0  1.412280   3.187679
3  128.0  5.134485  12.095540

md_lrpe_cosine-memory-fwd-batch12-head12-dim128-bf16-2d:
       n       Triton        Torch
0   16.0    27.001953    87.007812
1   32.0   108.001953   348.022461
2   64.0   432.001953  1392.081055
3  128.0  1728.001953  5568.316406

md_lrpe_cosine-speed-bwd-batch12-head12-dim128-bf16-2d:
       n    Triton      Torch
0   16.0  0.258258   0.194619
1   32.0  0.500497   0.738121
2   64.0  1.639229   3.080421
3  128.0  6.154274  12.229036

md_lrpe_cosine-memory-bwd-batch12-head12-dim128-bf16-2d:
       n       Triton        Torch
0   16.0    62.551953   128.551465
1   32.0   250.201953   514.201465
2   64.0  1000.801953  2056.801465
3  128.0  4003.201953  8227.201465
```

3d version, sequence length is $n^3$.
```
md_lrpe_cosine-speed-fwd-batch12-head12-dim128-bf16-3d:
      n     Triton      Torch
0   4.0   0.205550   0.411757
1   8.0   0.486381   0.753872
2  16.0   2.124623   3.299589
3  32.0  15.727946  24.246519

md_lrpe_cosine-memory-fwd-batch12-head12-dim128-bf16-3d:
      n       Triton         Torch
0   4.0     6.751953     22.130371
1   8.0    54.001953    177.517090
2  16.0   432.001953   1416.112793
3  32.0  3456.001953  11328.878418

md_lrpe_cosine-speed-bwd-batch12-head12-dim128-bf16-3d:
      n     Triton      Torch
0   4.0   0.210388   0.062451
1   8.0   0.404273   0.384454
2  16.0   2.413144   3.080609
3  32.0  17.940943  24.483002

md_lrpe_cosine-memory-bwd-batch12-head12-dim128-bf16-3d:
      n       Triton         Torch
0   4.0    15.639453     32.138965
1   8.0   125.101953    257.101465
2  16.0  1000.801953   2056.801465
3  32.0  8006.401953  16454.401465
```
