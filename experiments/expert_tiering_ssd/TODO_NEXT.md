# Pick-up list — written 2026-07-27, machine shut down clean

Server stopped, both GPUs at 0 MiB, host RAM back to 7 GB used. Nothing is
mid-flight; every item below starts from a cold boot.

## Branch map after today's reorganisation

| ref | contains | tip |
|---|---|---|
| `experiment/adaptive-decode-cache` | **this morning's live state** — the tree that was running before any tiering work. None of today's 7 work commits. | `9e007c2` |
| `experiment/expert-tiering-ssd` | **experiment 1** — three-tier store (GPU/RAM/SSD), static split | `fcb0b9a` |
| `experiment/expert-energy-tiering` | **experiment 2** — experiment 1 + the energy model, swap gate, prefetcher, README | `56c258e` |
| tags `today-20260727/experiment1-tiering-ssd`, `.../experiment2-energy` | immovable markers on today's two tips | |

All pushed to `origin`. To reproduce the pre-today world: `git checkout
experiment/adaptive-decode-cache`.

## Question 1 — is the expert cache updated for every layer on every token?

**No, and it never was.** Two different things run at two different rates:

| what | rate | cost |
|---|---|---|
| energy accumulation (`deepseek_v2.py`, in-graph) | **every layer, every token** | ~6 elementwise kernels on a 256-vector per layer; capture-safe, no D2H. This is the signal-gathering and it has to be per-token. |
| count-based cache tick (`kt_adaptive_on_decode_step`) | `KT_ADAPTIVE_LAYERS_PER_TICK=2` layers every `KT_ADAPTIVE_PERIOD=32` steps | one full 75-layer sweep per ~1200 steps |
| energy tick (`kt_energy_on_decode_step`) | one decision pass over all layers every `KT_ENERGY_PERIOD=4` steps, then **at most `KT_ENERGY_MAX_MOVES=4` moves globally** | ≤1 expert move per token |

What *was* wrong, and is now fixed in `56c258e`:

1. **The decision pass cost 61 ms per tick** — a per-layer Python loop with a
   device→host sync each iteration, against a 4.9 ms move. Batched into ~6
   kernels + one D2H for the whole model; `test_batched_energy.py` shows max
   difference 0.0 against the old path.
2. **The move budget was permanently saturated** — `KT_ENERGY_MIN_GAIN`
   defaulted to `0.0`, so any positive energy difference scheduled a swap:
   1800 promotions in 450 ticks = exactly 4.00/tick, its cap, forever, 65 GiB
   of traffic that never quiesced. Now gated on a ratio margin **and** an
   absolute floor scaled by the layer's mean energy (`_kt_energy_worth_moving`).
   `test_energy_gate.py`: 0 of 127 adjacent tail pairs pass, a genuine surge
   still does.
3. **Disk reads sat on the forward thread** — now a background prefetcher does
   the ~18.6 MiB read while tokens generate; only the buffer write is left in
   line (`test_prefetcher.py`).

⚠️ **None of these three fixes has been measured on a live server.** The energy
numbers in `RESULTS.md` (35.58 tok/s, 65–85-step promotion delay) all predate
all three. That is TODO 4.

Still unfixed and still in `boot_tiered.sh`: `KT_ADAPTIVE_COUNTS_DUMP_PT` forces
a `torch.save` on rank 0 every tick, and a count-based promotion into GPU does a
**full-layer restage (~149 ms)**. Both are suspects for TODO 2.

## Question 2 — is the low-RAM speed claim backed by a coherence check?

**No.** Every accuracy number published in `RESULTS.md` §3 was measured under
`KT_TIER_FILL_POOL=resident` — the *buggy* default that was replaced in
`fcb0b9a`. Under the shipped `fill=gpu` default there is exactly **one**
accuracy data point (RAM=32 static, 56.2 % QA), and the ladder rows RAM=64/16/8
have no accuracy number and no speed number at all.

So the headline "**43.35 tok/s in 67 GB of host RAM**" is a speed measurement
with no matching quality measurement. It must not be quoted as a result until
TODO 3 runs.

There is a specific reason to expect the two to be coupled, and it should be
tested as a falsifiable prediction:

> `should_skip_expert` means "not CPU-resident". Under `safe2`, a genuine top-K
> expert that is not CPU-resident cannot be computed, so it gets substituted.
> Shrinking the RAM tier shrinks the CPU-resident set — at RAM=32 only 32 of 256
> experts per layer are resident — so **safe2 degenerates toward sub2 as RAM
> falls**. That predicts the speed gain at low RAM is bought with substitution,
> i.e. RAM=8 should be both the fastest and the least coherent, monotonically.

If that holds, the honest framing of the whole experiment changes from "less RAM
is free" to "less RAM trades quality for speed on a measured curve".

## Question 3 — why didn't RAM=160 / SSD=0 reproduce ~40 tok/s?

It measured **28.33** (decbench, median of 4×300) against the **40.46** README
headline. The two runs are not the same configuration. Confirmed from both boot
logs:

| knob | `run_fast.sh` → 40.46 | `boot_tiered.sh` → 28.33 |
|---|---|---|
| routing mode | **`sub2`** | **`safe2`** |
| `GPU_EXPERTS` | **104** | **96** |
| `MEM_FRACTION` | 0.95 | 0.85 |
| `MAX_TOTAL_TOKENS` | 81920 | 4096 |
| placement | `hotcore` | `oracle` + warm mask |
| adaptive tick | **off** | on, period 32, + per-tick `torch.save` |

Leading hypothesis, in order of expected size:

1. **`sub2` vs `safe2`.** `sub2` substitutes even non-resident top-K experts, so
   it barely touches the CPU path; `safe2` sends every non-resident genuine
   top-K expert to the CPU path, which is the decode bottleneck. At RAM=160
   almost everything is CPU-resident, so `safe2` maximises CPU round-trips —
   the worst case for it — while `sub2` minimises them. This alone could
   account for most of the gap, and if so **there is no regression at all**,
   just two different routing contracts being compared.
2. `GPU_EXPERTS` 96 vs 104 — 8 fewer GPU experts per layer, more CPU work.
3. The adaptive tick's per-tick `torch.save` + 149 ms full-layer restages.
4. `MEM_FRACTION`/`MAX_TOTAL_TOKENS` — expected to be small at batch 1.

Note this also explains why RAM=32 measured *faster* (43.35) than RAM=160
(28.33) under identical code: fewer CPU-resident experts ⇒ fewer CPU
round-trips. Same mechanism as the prediction in Question 2.

---

## TODO, in order

### 1. Settle the baseline (do this first — everything else compares to it)

Was launched today at 12:13 and killed at shutdown; `logs/pretoday_runfast.log`
holds a partial boot only.

```bash
git checkout experiment/adaptive-decode-cache          # the morning tree
setsid env KT_GPU_PREFILL_THRESHOLD=0 \
  TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  ./run_fast.sh > logs/pretoday_runfast.log 2>&1 < /dev/null &
# boot is ~25 min here: no RAM tier, so the loader reads the whole checkpoint
python3 bench/perf_probe/decbench.py 300 5
```

Expected ~40.5 (README records 40.46, min 40.34 / max 40.60). **Use `setsid`** —
plain `nohup` let the harness kill the process group at 12:13 (`Exit 137`).

### 2. Matched-knob A/B — the actual regression test

Same harness, today's branch, tiering active but degenerate (SSD=0), every knob
matched to `run_fast.sh`:

```bash
git checkout experiment/expert-energy-tiering
KT_RAM_EXPERTS=160 GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=81920 \
KT_TOPK_MODE=sub2 WARM_START=0 KT_ADAPTIVE_DECODE=0 \
KT_ADAPTIVE_COUNTS_DUMP_PT= \
  bash experiments/expert_tiering_ssd/boot_tiered.sh
python3 bench/perf_probe/decbench.py 300 5
```

- matches TODO 1 ⇒ **no regression**; the 28.33 was a config difference, and
  the write-up should say so plainly.
- does not match ⇒ real regression. Then walk the knobs back one at a time in
  the order of the table above, one boot each.

### 3. Coherence at low RAM under `fill=gpu` — the biggest hole

```bash
for R in 160 64 32 16 8; do
  KT_RAM_EXPERTS=$R bash experiments/expert_tiering_ssd/boot_tiered.sh &
  # wait for ready, then:
  .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py
  .venv/bin/python bench/perf_probe/decbench.py 300 4
done
```

Record QA accuracy, loop rate and mean reasoning length beside tok/s for every
row, and test the monotonic prediction from Question 2. Also run each row under
`KT_TOPK_MODE=sub2` so the substitution effect can be separated from the tier
effect. This closes TODO 3 and TODO 5 of the old list together.

### 4. Re-measure the energy path

```bash
GPU_EXPERTS=96 KT_RAM_EXPERTS=32 KT_ENERGY=1 \
  bash experiments/expert_tiering_ssd/boot_tiered.sh
.venv/bin/python experiments/expert_tiering_ssd/tier_bench.py
.venv/bin/python -c "import torch; print(torch.load('/tmp/kt_energy_report.pt'))"
```

Compare against the pre-fix record: 35.58 tok/s, promotion delay 65–85 steps,
4.00 moves/tick pinned at cap. What to look for: moves/tick well below 4,
`prefetch_miss` small, promotion delay collapsing toward the 4-step tick period.

### 5. Rules that keep the numbers honest

- **One request in flight, always.** `CUDA_GRAPH_MAX_BS=1`; a second concurrent
  request drops both to eager and silently halves the reading (a 43.25 pass was
  recorded as 16.86 this way). Run everything under `bench_lock.py`.
- **Never compare across harnesses.** `decbench.py` (raw `/generate`) is the
  headline harness; `tier_bench.py` is chat-streaming. At RAM=32 they happened
  to agree (43.35 vs 43.43/43.01/43.20) but that is not guaranteed.
- **Check for runaway processes** (`ps --sort=-pcpu | head`) before trusting any
  slow reading — a stray `ugrep` once cratered decode 14 → 0.77 and faked a
  "too slow" verdict.

### 6. Free speedup available, unclaimed

Weights live on `/data` (**1.8 GB/s**, SATA) while `/cache/nvme0` (**3.2 GB/s**)
sits idle. Moving them is a 1.8× on every cold read: boot time, and the SSD→RAM
promotion path that the energy model depends on. An expert is 18.56 MiB
(3 × 6.000 MiB int4 + 3 × 0.188 MiB bf16 scales), so a promotion is 10.8 ms of
`/data` versus 6.1 ms of NVMe.

### 7. Write-up debts

- `RESULTS.md` §2 still shows RAM=64/16/8 as "no valid speed number" and §3's
  accuracy table is `fill=resident` — both must be replaced, not appended to.
- `README.md` records the RAM=160 anomaly as unresolved; TODO 2 resolves it.
- Optional: rebuild `experiment/expert-tiering-ssd` as a clean cut of tiering +
  the fill-pool fix only. It currently also carries the energy-model commits,
  which belong to experiment 2.
