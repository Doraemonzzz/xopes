# Lightning Attention with Positional Encoding

## Notation

- lape_r: Lightning attention with positional encoding recurrence
- lape_p: Lightning attention with positional encoding parallel

## A100

### Speed
```
lape-speed-fwd-batch4-head32-dim128-bf16:
         n  LAPE_Recurrence  LAPE_Parallel
0    256.0         0.601625       0.271241
1    512.0         1.290126       0.317768
2   1024.0         2.589064       0.507440
3   2048.0         5.137877       0.956144
4   4096.0        10.218731       1.836642
5   8192.0        20.389959       3.606418
6  16384.0        40.226723       7.136195
7  32768.0        81.342590      14.334118

lape-speed-bwd-batch4-head32-dim128-bf16:
         n  LAPE_Recurrence  LAPE_Parallel
0    256.0         3.207083       1.264351
1    512.0         6.299784       1.189274
2   1024.0        12.464073       1.803896
3   2048.0        24.786295       3.314132
4   4096.0        49.382210       6.316881
5   8192.0        98.786339      12.319504
6  16384.0       197.191162      24.367479
7  32768.0       393.686493      48.471664
```

### Memory
```
lape-memory-fwd-batch4-head32-dim128-bf16:
         n  LAPE_Recurrence  LAPE_Parallel
0    256.0        40.016113      56.016113
1    512.0        72.016113     104.016113
2   1024.0       136.016113     168.016113
3   2048.0       264.016113     328.016113
4   4096.0       520.016113     648.016113
5   8192.0      1032.016113    1288.016113
6  16384.0      2056.016113    2568.016113
7  32768.0      4104.016113    5128.016113

lape-memory-bwd-batch4-head32-dim128-bf16:
         n  LAPE_Recurrence  LAPE_Parallel
0    256.0       111.631860     127.632324
1    512.0       207.631860     239.632324
2   1024.0       398.631860     430.632324
3   2048.0       780.831421     844.831909
4   4096.0      1545.631421    1673.631909
5   8192.0      3075.231421    3331.231909
6  16384.0      6134.431421    6646.431909
7  32768.0     12252.831421   13276.831909
```
