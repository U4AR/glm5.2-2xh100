# Live Benchmark: MTP On, Dynamic Expert Cache Off vs On

Both runs used the same benchmark prompts and launch profile:

```text
MTP=1
GPU_EXPERTS=96
MEM_FRACTION=0.92
MAX_TOTAL_TOKENS=32768
KT_GPU_PREFILL_THRESHOLD=512
max_tokens=128
```

Only `DYN_UPDATE` changed.

| task | prompt tok | baseline s | adaptive s | delta s | speedup | baseline out tok/s | adaptive out tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 1169 | 14.048 | 14.203 | +0.155 | 0.989x | 9.111 | 9.012 |
| largest-eigenval | 179 | 3.955 | 3.641 | -0.314 | 1.086x | 32.366 | 35.159 |
| fix-git | 46 | 3.763 | 3.746 | -0.017 | 1.005x | 34.019 | 34.173 |
| compile-compcert | 84 | 3.615 | 3.658 | +0.043 | 0.988x | 35.413 | 34.992 |
| git-multibranch | 253 | 3.930 | 4.081 | +0.151 | 0.963x | 32.572 | 31.366 |

Total elapsed: baseline 29.311s, adaptive 29.329s, speedup 0.999x.
Completion throughput: baseline 21.835 tok/s, adaptive 21.821 tok/s, speedup 0.999x.

Server-side decode logs, excluding the long-prompt log line that includes first-token/prelude effects:

| run | steady decode mean tok/s | steady decode median tok/s | mean accept len |
|---|---:|---:|---:|
| baseline MTP | 34.553 | 34.690 | 3.284 |
| adaptive MTP | 34.845 | 34.965 | 3.216 |

Adaptive cache log summary:

| metric | value |
|---|---:|
| adaptive layer updates | 75 |
| total expert swaps | 452 |
| mean top2 hit before -> after | 0.7689 -> 0.8628 |
| mean top8 hit before -> after | 0.8455 -> 0.8883 |
| summed update time | 0.532s |

Interpretation: with MTP enabled, the adaptive cache did not improve end-to-end
speed on this one-pass five-task benchmark. It changed resident experts and
improved measured residency hit rates, but decode throughput was already around
34-35 tok/s and stayed there.
