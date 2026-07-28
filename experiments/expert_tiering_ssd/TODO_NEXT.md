# Pick-up list — written 2026-07-28, machine stopped clean

Supersedes the 2026-07-27 list (kept in git history). Those questions are all
answered now; the branch map and the "rules that keep the numbers honest" from
that version are folded into §6 below.

## State of the world in one paragraph

Counting is free, deciding is free, **only physical movement costs anything** —
and it costs 16% because `load_expert`'s NUMA repack runs on the same CPUInfer
thread pool as the MoE forward, which is the decode bottleneck. The GPU cycle
is now *profitable* (+1.57 tok/s) after the stable-slot swap replaced the full
restage. Movement never converges, because the working set is larger than the
RAM tier — a capacity limit, not a signal-quality or latency problem. Full
ladder, phase profiles and the corrections: `RESULTS.md` §0, §4, §5.

## 1. Get the NUMA repack off the shared CPUInfer pool ← the only real lever left

`promote` is now the single largest genuine cost: 16.4 ms/visit RAM-only,
21.3 ms with the GPU cycle. It is **not** disk — proved: 16.4 ms with the
prefetch hitting 94-98%, 15.7 ms with prefetch off entirely. It is the repack in
`ktransformers/kt-kernel/operators/avx2/rawint4_packed_avx512vnni-moe.hpp`
~L895 `load_expert`, which calls `pool->dispense_backend()->do_numa_job` and
`do_work_stealing_job` — the same pool the forward uses (CPU ~93% busy, GPU0
48% idle).

Do this in order:
1. **Measure inside `load_expert`** — split repack from store-insert before
   optimising either. This exact code path has now eaten two wrong guesses
   (async prefetch, and my first reading of the fixed toll). *Split a cost
   before optimising half of it.*
2. Then: dedicated repack thread(s) outside the CPUInfer pool. C++ change +
   rebuild — `install.sh build` with `CPUINFER_USE_CUDA=1` **only**; a manual
   cmake yields a CPU-only .so and silently drops `submit_with_cuda_stream`.
3. Fallback if that is too invasive: rate-limit movement to a slice of each
   step's CPU budget. Cheap and pure Python, but it only trades cost for slower
   convergence — and convergence is capacity-bound anyway, so expect little.

## 2. Remove the prefetch lookahead (commit 7cc7ea0)

Verified worthless by the A/B above. Revert the lookahead and the
`_KtExpertPrefetcher` wiring in `kt_ep_wrapper.py`. **Keep** the
`read_expert`/`stage_expert`/`promote_expert(weights=)` split in `amx.py` — it
is load-bearing and was ported back to the repo copy precisely so a rebuild
would not silently revert it.

## 3. Do NOT ship `KT_TIER_MAX_PROMOTE=0`

Explicitly cancelling an item that was on the old list. Freezing the GPU cycle
was a workaround for the 250 ms restage; that bug is fixed and the cycle now
*earns* 1.57 tok/s. Current defaults are correct: `KT_TIER_MAX_PROMOTE=2`,
`KT_TIER_INCREMENTAL=1` (0 declines across 336 swaps).

## 4. Capacity crossover sweep — cancelled mid-flight, script ready to run

`scratchpad/capacity_sweep.sh` (copy it into the repo if it survives a reboot).
RAM 16/32/48/72 x frozen/move, scored against a 66-item reference built at
RAM=152/SSD=0 where every expert is reachable.

~4 h for all 9 boots; **~1.5 h trimmed to RAM 32 and 48**, which is where the
crossover must lie — RAM=72 frozen already holds reference accuracy, RAM=32
frozen is broken (0.5625 acc, 44% loops). Needs `runs/ref66.json`, which does
not exist yet; the reference boot is the script's first step.

Question it answers: below what RAM size is movement worth its 16%?

## 5. Loose measurement ends

- Pass 2 covered F/E/D/A only. **B and C (counting/deciding are free) rest on
  pass 1 alone** — margin is large (40.59, 40.56 vs 40.53) but unreplicated.
- The RAM=72 accept-length drop (frozen 2.912 → dynamic 2.680) is still
  unexplained. It is *not* the loop-rate confound that explains RAM=32 — both
  sides loop at 6.25%. Rungs B/C/D can attribute it; per-rung accept means are
  already in `logs/dc_p1_*.log` and `summarize_decompose.py` parses them.
- `token_latency.py` was written but never run. Measures inter-token gaps (p99,
  count over 150 ms) so a stall reads as a hitch instead of hiding in a median.
  Use it on the jitter movement adds: frozen spreads 40.3-40.7, dynamic 29-48.
- `KT_TIER_PROFILE=1` gives per-phase visit timings; the no-change profile logs
  on both TP ranks while the full profile is rank-0 only, so **do not compare
  their phase means directly**.

## 6. Traps that have already cost time

- **`kt_ep_wrapper.py` runs from `.venv/`**, not the repo tree. Edit
  `.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/kt_ep_wrapper.py`.
  Same story for `kt_kernel/utils/amx.py`.
- **`blocked % of wall time` is not a cost.** It measures elapsed time inside a
  call, and time spent waiting on the GPU is free — rung D reports `blocked
  4.5%` against a true cost of 16.1%. Measure throughput, not blocked time.
- **The 16-item accuracy set is retired** (66 now; `LEGACY_QA` kept for
  continuity). One item was worth 0.0625, so every past quality comparison was
  a one- or two-item difference.
- **Boot-to-boot variation is large** (repeat boots of one config span
  31.16-38.67); within-boot sd is 0.57. Compare within a boot, or run both
  orders.
- **One request in flight, always** — `CUDA_GRAPH_MAX_BS=1`; a second
  concurrent request drops both to eager and halves the reading. Use
  `bench_lock.py`.
- **Never compare across harnesses** — `decbench.py` (raw `/generate`) is the
  headline harness; `tier_bench.py` is chat-streaming.
- **Check `ps --sort=-pcpu`** before trusting a slow reading; a stray runaway
  once cratered decode 14 → 0.77 and faked a "too slow" verdict.
- `sub2` must never become the default; only `safe2`. SSD substitution is a
  corruption dial, not a speed feature.

## 7. Free speedup still unclaimed

Weights are on `/data` (1.8 GB/s SATA) while `/cache/nvme0` (3.2 GB/s) sits
idle. 1.8x on every cold read — boot time, and the SSD→RAM promotion path. An
expert is 18.56 MiB, so a promotion is 10.8 ms off `/data` vs 6.1 ms off NVMe.
Lower value than it looked: the ladder proved the disk read is not on the
critical path for *decode*. Still worth it for boot.

## 8. Write-up debts

- `README.md` needs the ladder result and must document both experiments with
  achieved speeds.
- `RESULTS.md` §1-§3 still carry `fill=resident` accuracy tables; §0 flags them
  but they should be replaced once the capacity sweep produces `fill=gpu`
  numbers on the 66-item set.

## Best measured config

```
KT_RAM_EXPERTS=72 KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 \
KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 \
KT_TIER_INCREMENTAL=1 GPU_EXPERTS=104 KT_TOPK_MODE=safe2 \
  bash experiments/expert_tiering_ssd/boot_tiered.sh
```

35.58 tok/s with the GPU cycle earning its keep, vs 40.53 frozen and 29.02 on
the pre-fix path. **Frozen still wins outright at RAM=72** — movement is
insurance for an undersized RAM tier, not a throughput feature.
