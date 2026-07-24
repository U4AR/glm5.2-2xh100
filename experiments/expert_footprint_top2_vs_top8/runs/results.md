# Expert Footprint: Top-2 vs Top-8

Model: GLM-5.2-W4AFP8. Per routed expert: 18.0 MiB int4 weights.
Top-2 means the two highest-weight experts from each captured top-8 router row.
p90 uses `numpy.percentile(..., 90)` over the analyzed routed layers.

| trace | layers | top2 total MiB | top8 total MiB | top8/top2 | top2 p90 MiB | top8 p90 MiB | p90 ratio | top2 count | top8 count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1_single | 75 | 46,962.0 | 144,288.0 | 3.07 | 774.0 | 2,372.4 | 3.07 | 2,609 | 8,016 |
| exp2_llm-inference-batching-scheduler | 75 | 248,004.0 | 338,292.0 | 1.36 | 3,708.0 | 4,608.0 | 1.24 | 13,778 | 18,794 |
| exp3_compile-compcert | 75 | 78,102.0 | 216,630.0 | 2.77 | 1,386.0 | 3,405.6 | 2.46 | 4,339 | 12,035 |
| exp3_fix-git | 75 | 57,420.0 | 171,774.0 | 2.99 | 972.0 | 2,754.0 | 2.83 | 3,190 | 9,543 |
| exp3_git-multibranch | 75 | 134,946.0 | 293,400.0 | 2.17 | 2,178.0 | 4,305.6 | 1.98 | 7,497 | 16,300 |
| exp3_largest-eigenval | 75 | 132,858.0 | 282,402.0 | 2.13 | 2,116.8 | 4,291.2 | 2.03 | 7,381 | 15,689 |
| exp3_llm-inference-batching-scheduler | 75 | 248,004.0 | 338,292.0 | 1.36 | 3,708.0 | 4,608.0 | 1.24 | 13,778 | 18,794 |

Footnote: footprint ignores W4AFP8 scale metadata; that is below 1% of expert weight bytes for this headline comparison.
