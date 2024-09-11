# Xopes

Xopes: Toolbox for Accelerating Deep Learning Operators.

# Plan

- 240910
  - [x] add fwd_fn, bwd_fn for lrpe, md_lrpe;
  - [x] add test, benchmark for flao_fal;
  - [x] add lrpe document;
- 240911
  - [ ] add md_lrpe document;
  - [ ] clear flao code, add interface;
  - [ ] fuse act + lrpe;


# Note
```
[Feature Add]
[Bug Fix]
[Benchmark Add]
[Document Add]
```

Symbol Explanation: When benchmarking, use `o` to represent the output of the function. For the function name, use `fn_version` where `fn` is the function name and `version` can be either `torch` or `triton`.
