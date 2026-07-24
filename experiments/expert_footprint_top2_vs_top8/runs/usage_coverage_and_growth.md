# Usage Coverage and Sequential Unique Expert Growth

Unit: one expert weight block = one `(layer, expert_id)` tensor = 18.0 MiB.

## top2

Total selections: 260,400. Unique blocks touched: 15,191 = 273,438 MiB = 267.0 GiB.

| coverage target | blocks needed | weight MiB | weight GiB | full routed expert share | actual coverage |
|---:|---:|---:|---:|---:|---:|
| 50% | 1,735 | 31,230 | 30.5 | 9.0% | 50.01% |
| 75% | 4,372 | 78,696 | 76.9 | 22.8% | 75.00% |
| 90% | 7,631 | 137,358 | 134.1 | 39.7% | 90.00% |
| 95% | 9,697 | 174,546 | 170.5 | 50.5% | 95.00% |
| 99% | 12,971 | 233,478 | 228.0 | 67.6% | 99.00% |
| 100% | 15,191 | 273,438 | 267.0 | 79.1% | 100.00% |

| task order | task unique blocks | new blocks added | cumulative unique blocks | cumulative GiB |
|---|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 13,778 | 13,778 | 13,778 | 242.2 |
| largest-eigenval | 7,381 | 543 | 14,321 | 251.7 |
| fix-git | 3,190 | 236 | 14,557 | 255.9 |
| compile-compcert | 4,339 | 181 | 14,738 | 259.1 |
| git-multibranch | 7,497 | 453 | 15,191 | 267.0 |

## top8

Total selections: 1,041,600. Unique blocks touched: 19,042 = 342,756 MiB = 334.7 GiB.

| coverage target | blocks needed | weight MiB | weight GiB | full routed expert share | actual coverage |
|---:|---:|---:|---:|---:|---:|
| 50% | 4,013 | 72,234 | 70.5 | 20.9% | 50.00% |
| 75% | 8,409 | 151,362 | 147.8 | 43.8% | 75.00% |
| 90% | 12,527 | 225,486 | 220.2 | 65.2% | 90.00% |
| 95% | 14,582 | 262,476 | 256.3 | 75.9% | 95.00% |
| 99% | 17,213 | 309,834 | 302.6 | 89.7% | 99.00% |
| 100% | 19,042 | 342,756 | 334.7 | 99.2% | 100.00% |

| task order | task unique blocks | new blocks added | cumulative unique blocks | cumulative GiB |
|---|---:|---:|---:|---:|
| llm-inference-batching-scheduler | 18,794 | 18,794 | 18,794 | 330.4 |
| largest-eigenval | 15,689 | 123 | 18,917 | 332.5 |
| fix-git | 9,543 | 47 | 18,964 | 333.4 |
| compile-compcert | 12,035 | 17 | 18,981 | 333.7 |
| git-multibranch | 16,300 | 61 | 19,042 | 334.7 |
