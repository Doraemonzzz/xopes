# Inverse benchmark

## Notation

- tr: Triton
- to: Torch
- toc: Torch Compile

## A800

### Speed
```
inverse-speed-fwd-batch4-fp32:
      n  FS-Torch  FS-Triton-Naive  FS-Triton-Loop  Jac-Torch  Jac-Triton  FS-Torch-Compile  Jac-Torch-Compile
0  16.0  1.565568         0.019258        0.041893   0.555933    0.012153          0.422826           0.267086
1  32.0  3.240770         0.027968        0.160096   1.098109    0.015543          0.941969           0.579487
2  64.0  6.568690         0.097864        0.872180   2.201261    0.038826          1.923564           1.405010

inverse-speed-fwd-batch4-fp32:
       n  FS-Triton-Naive  Jac-Triton
0   16.0         0.019400    0.012304
1   32.0         0.028737    0.016407
2   64.0         0.093533    0.024167
3  128.0         0.413369    0.082856
```


### Memory
```
inverse-memory-fwd-batch4-fp32:
      n   FS-Torch  FS-Triton-Naive  FS-Triton-Loop  Jac-Torch  Jac-Triton  FS-Torch-Compile  Jac-Torch-Compile
0  16.0   8.266602            8.375           8.375   8.884277       8.375          8.639648              8.750
1  32.0   8.657227            9.125           9.125  11.145020       9.125         10.155273             10.625
2  64.0  10.188477           12.125          12.125  20.172363      12.125         16.186523             18.125

inverse-memory-fwd-batch4-fp32:
       n  FS-Triton-Naive  Jac-Triton
0   16.0            8.375       8.375
1   32.0            9.125       9.125
2   64.0           12.125      12.125
3  128.0           24.125      24.125
```
