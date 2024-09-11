# A800

```
act-speed-fwd-batch12-head12-dim128-act_relu-bf16:
         n    Triton     Torch
0    256.0  0.034165  0.019579
1    512.0  0.062193  0.032218
2   1024.0  0.116155  0.055339
3   2048.0  0.225513  0.099138
4   4096.0  0.444840  0.186722
5   8192.0  0.882199  0.362886
6  16384.0  1.756082  0.714608
7  32768.0  3.504138  1.418184

act-speed-fwd-batch12-head12-dim128-act_sigmoid-bf16:
         n    Triton     Torch
0    256.0  0.034442  0.020130
1    512.0  0.061550  0.032681
2   1024.0  0.116173  0.056768
3   2048.0  0.225564  0.103652
4   4096.0  0.444650  0.196482
5   8192.0  0.881356  0.383194
6  16384.0  1.756652  0.756446
7  32768.0  3.505296  1.501453

act-speed-fwd-batch12-head12-dim128-act_silu-bf16:
         n    Triton     Torch
0    256.0  0.034389  0.020922
1    512.0  0.062097  0.033212
2   1024.0  0.116206  0.058178
3   2048.0  0.225589  0.106230
4   4096.0  0.444191  0.202024
5   8192.0  0.881460  0.393944
6  16384.0  1.756321  0.776794
7  32768.0  3.506221  1.543298

act-speed-fwd-batch12-head12-dim128-act_none-bf16:
         n    Triton     Torch
0    256.0  0.003732  0.003730
1    512.0  0.003734  0.003728
2   1024.0  0.003730  0.003719
3   2048.0  0.003712  0.003711
4   4096.0  0.003713  0.003712
5   8192.0  0.003711  0.003717
6  16384.0  0.003713  0.003710
7  32768.0  0.003712  0.003726

act-memory-fwd-batch12-head12-dim128-act_relu-bf16:
         n  Triton   Torch
0    256.0    18.0    18.0
1    512.0    36.0    36.0
2   1024.0    72.0    72.0
3   2048.0   144.0   144.0
4   4096.0   288.0   288.0
5   8192.0   576.0   576.0
6  16384.0  1152.0  1152.0
7  32768.0  2304.0  2304.0

act-memory-fwd-batch12-head12-dim128-act_sigmoid-bf16:
         n  Triton   Torch
0    256.0    18.0    18.0
1    512.0    36.0    36.0
2   1024.0    72.0    72.0
3   2048.0   144.0   144.0
4   4096.0   288.0   288.0
5   8192.0   576.0   576.0
6  16384.0  1152.0  1152.0
7  32768.0  2304.0  2304.0

act-memory-fwd-batch12-head12-dim128-act_silu-bf16:
         n  Triton   Torch
0    256.0    18.0    18.0
1    512.0    36.0    36.0
2   1024.0    72.0    72.0
3   2048.0   144.0   144.0
4   4096.0   288.0   288.0
5   8192.0   576.0   576.0
6  16384.0  1152.0  1152.0
7  32768.0  2304.0  2304.0

act-memory-fwd-batch12-head12-dim128-act_none-bf16:
         n  Triton   Torch
0    256.0     9.0     9.0
1    512.0    18.0    18.0
2   1024.0    36.0    36.0
3   2048.0    72.0    72.0
4   4096.0   144.0   144.0
5   8192.0   288.0   288.0
6  16384.0   576.0   576.0
7  32768.0  1152.0  1152.0

act-speed-bwd-batch12-head12-dim128-act_relu-bf16:
         n    Triton     Torch
0    256.0  0.078223  0.041958
1    512.0  0.096592  0.080236
2   1024.0  0.182051  0.144383
3   2048.0  0.354167  0.271011
4   4096.0  0.698870  0.524329
5   8192.0  1.388389  1.030858
6  16384.0  2.767783  2.043795
7  32768.0  5.531606  4.070539

act-speed-bwd-batch12-head12-dim128-act_sigmoid-bf16:
         n    Triton     Torch
0    256.0  0.077237  0.042061
1    512.0  0.096812  0.080618
2   1024.0  0.182263  0.144993
3   2048.0  0.354213  0.272048
4   4096.0  0.699107  0.527146
5   8192.0  1.388890  1.037181
6  16384.0  2.770959  2.057968
7  32768.0  5.533720  4.096228

act-speed-bwd-batch12-head12-dim128-act_silu-bf16:
         n    Triton     Torch
0    256.0  0.098585  0.042438
1    512.0  0.137011  0.082365
2   1024.0  0.182419  0.148127
3   2048.0  0.354363  0.280473
4   4096.0  0.699270  0.540400
5   8192.0  1.389028  1.066066
6  16384.0  2.770426  2.117301
7  32768.0  5.532970  4.214682

act-speed-bwd-batch12-head12-dim128-act_none-bf16:
         n    Triton     Torch
0    256.0  0.027041  0.027085
1    512.0  0.043668  0.043643
2   1024.0  0.075340  0.075808
3   2048.0  0.138518  0.138493
4   4096.0  0.265803  0.265362
5   8192.0  0.518583  0.517996
6  16384.0  1.028103  1.028244
7  32768.0  2.037244  2.036668

act-memory-bwd-batch12-head12-dim128-act_relu-bf16:
         n   Triton    Torch
0    256.0    44.55    44.55
1    512.0    89.10    89.10
2   1024.0   178.20   178.20
3   2048.0   356.40   356.40
4   4096.0   712.80   712.80
5   8192.0  1425.60  1425.60
6  16384.0  2851.20  2851.20
7  32768.0  5702.40  5702.40

act-memory-bwd-batch12-head12-dim128-act_sigmoid-bf16:
         n   Triton    Torch
0    256.0    44.55    44.55
1    512.0    89.10    89.10
2   1024.0   178.20   178.20
3   2048.0   356.40   356.40
4   4096.0   712.80   712.80
5   8192.0  1425.60  1425.60
6  16384.0  2851.20  2851.20
7  32768.0  5702.40  5702.40

act-memory-bwd-batch12-head12-dim128-act_silu-bf16:
         n   Triton    Torch
0    256.0    44.55    44.55
1    512.0    89.10    89.10
2   1024.0   178.20   178.20
3   2048.0   356.40   356.40
4   4096.0   712.80   712.80
5   8192.0  1425.60  1425.60
6  16384.0  2851.20  2851.20
7  32768.0  5702.40  5702.40

act-memory-bwd-batch12-head12-dim128-act_none-bf16:
         n  Triton   Torch
0    256.0    27.0    27.0
1    512.0    54.0    54.0
2   1024.0   108.0   108.0
3   2048.0   216.0   216.0
4   4096.0   432.0   432.0
5   8192.0   864.0   864.0
6  16384.0  1728.0  1728.0
7  32768.0  3456.0  3456.0
```