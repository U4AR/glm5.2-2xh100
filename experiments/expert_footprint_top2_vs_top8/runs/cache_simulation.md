# Adaptive Expert Cache Simulation

Replay unit: one token-layer row from the captured prefill traces. Hits are
counted before the policy sees the row, then the policy updates state and may
refresh residents at its update interval.

| capacity/layer | interval | policy | top2 hit | top8 hit | top2 misses/row | swaps | copy GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 104 | 32 | frequency | 72.25% | 66.72% | 0.555 | 12817 | 225.30 |
| 104 | 32 | lru | 71.59% | 66.48% | 0.568 | 32400 | 569.53 |
| 104 | 32 | static_uniform | 41.15% | 41.07% | 1.177 | 0 | 0.00 |
| 104 | 32 | weighted_top2 | 81.14% | 58.98% | 0.377 | 8578 | 150.79 |
| 104 | 32 | weighted_with_tail | 81.98% | 61.81% | 0.360 | 9818 | 172.58 |
