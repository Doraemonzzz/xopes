# Lightning Attention with Vector Decay Sub-Intra

## Notation

- LAVD_S1_DK: Vector decay with sub-intra with dk
- LAVD_S2_DK: Vector decay with sub-intra-sep with dk
- LAVD_DK: Vector decay with intra with dk (only sub-intra)
- LAVD_S1_DK_DV: Vector decay with sub-intra with dk and dv
- LAVD_S2_DK_DV: Vector decay with sub-intra-sep with dk and dv
- LAVD_DK_DV: Vector decay with intra with dk and dv (only sub-intra)

## A100

### Speed
```
lavd-speed-fwd-batch4-head32-dim128-bf16:
        n  LAVD_S1_DK  LAVD_S2_DK    LAVD_DK  LAVD_S1_DK_DV  LAVD_S2_DK_DV  LAVD_DK_DV
0   256.0    0.138908    0.143488   1.070115       0.138867       0.143519    1.069770
1   512.0    0.260577    0.263504   2.121088       0.260084       0.263481    2.119510
2  1024.0    0.498029    0.506046   4.193765       0.498062       0.504908    4.193856
3  2048.0    0.971267    0.985630   8.329923       0.971582       0.985476    8.349047
4  4096.0    1.916438    1.965908  16.599659       1.914760       1.965090   16.612597
5  8192.0    3.795009    3.882444  33.060211       3.793851       3.881277   33.059231
```
