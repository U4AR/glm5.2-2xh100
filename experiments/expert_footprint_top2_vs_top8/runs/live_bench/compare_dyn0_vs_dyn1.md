# Live Benchmark: Dynamic Expert Cache Off vs On

| task | prompt tok | baseline s | adaptive s | delta s | speedup | baseline out tok/s | adaptive out tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 1169 | 17.410 | 17.733 | +0.323 | 0.982x | 7.352 | 7.218 |
| largest-eigenval | 179 | 7.705 | 7.394 | -0.311 | 1.042x | 16.612 | 17.311 |
| fix-git | 46 | 7.014 | 6.966 | -0.048 | 1.007x | 18.250 | 18.374 |
| compile-compcert | 84 | 7.179 | 7.008 | -0.171 | 1.024x | 17.829 | 18.266 |
| git-multibranch | 253 | 7.836 | 7.564 | -0.272 | 1.036x | 16.335 | 16.923 |

Total elapsed: baseline 47.144s, adaptive 46.665s, speedup 1.010x.
Completion throughput: baseline 13.575 tok/s, adaptive 13.715 tok/s, speedup 1.010x.
