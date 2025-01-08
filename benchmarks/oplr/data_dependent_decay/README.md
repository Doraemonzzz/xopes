# OPLR no decay benchmark

## Notation

- to: Torch
- toc: Torch Compile

## H100

### Speed

```
oplr_data_dependent_decay-speed-fwd-batch4-dim128-dim128-bf16:
        n          to        toc
0   256.0   21.662888   3.857692
1   512.0   45.554607   5.865893
2  1024.0   61.805504  11.943113
3  2048.0  126.783875  38.275146

oplr_data_dependent_decay-speed-bwd-batch4-dim128-dim128-bf16:
        n          to        toc
0   256.0   81.676926   7.312066
1   512.0  111.869858  13.181466
2  1024.0  204.030045  24.149136
3  2048.0  392.604828  59.966911
```

### Memory

```
oplr_data_dependent_decay-memory-fwd-batch4-dim128-dim128-bf16:
        n        to     toc
0   256.0   129.625   73.75
1   512.0   258.875  147.50
2  1024.0   517.375  295.50
3  2048.0  1034.375  591.00

oplr_data_dependent_decay-memory-bwd-batch4-dim128-dim128-bf16:
        n           to          toc
0   256.0   195.589453   174.107031
1   512.0   390.676953   338.825391
2  1024.0   781.101953   623.441797
3  2048.0  1561.951953  1180.948047
```
