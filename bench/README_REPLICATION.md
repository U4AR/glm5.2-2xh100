# Replicating the expert-streaming / prefetch results

Everything below was measured on 2x H100 NVL (TP2, 373 GB host RAM) with the
W4AFP8 weights at `/data/models/glm52-w4afp8`. The harnesses no longer hardcode
that box: paths, scratch space and queue chaining are all resolved at run time.
What they still require is the model itself and a GPU large enough to hold the
resident expert pool.

## What you need

- The repo's normal serving setup (`./run_fast.sh` must boot and answer on
  `127.0.0.1:8000`). Port 8000 is fixed by the launchers.
- Enough VRAM for `GPU_EXPERTS` resident experts per layer plus slack. The
  numbers below use 100 residents + landing slots on 2x80 GB. On a smaller card
  pass a smaller value (see "Other machines").
- `zlib`-only Python for the coherence detector; no extra installs.

## Environment

| variable | default | meaning |
|---|---|---|
| `RUNGLM_SCRATCH` | `$TMPDIR/runglm-bench`, else `/tmp/runglm-bench` | logs, markers, coherence texts |
| `TRITON_CACHE_DIR` | `/cache/nvme0/triton-cache` if present, else `$SP/triton-cache` | must NOT be on a small root disk; a full root disk makes the server SIGQUIT mid-run |
| `RUNGLM_CHAIN` | `0` | `1` makes a harness wait for the previous one's done-marker. Only for queueing several ladders on one box; standalone runs start immediately |
| `GPU_EXPERTS` | `100` | resident experts per layer |
| `MEM_FRACTION` | `0.94` | leave slack for CUDA graphs, the MTP draft and prefill scratch |

Every harness boots its own server, measures, and restores production when it
finishes. They each take the whole box: do not run two at once.

## The two results

Both configurations run the MoE **entirely on the GPU** (`KT_GPU_ONLY=1`): keep a
genuine expert when it is GPU-resident (or the prefetcher landed it), substitute
the rest from the resident pool. The CPU expert path is then zero by
construction, not by coverage.

### 1. Resident-only — no movement at all

```bash
bash bench/resident_only_coherence.sh
```

Expected (2x H100, GPU_EXPERTS=100):

```
Ronly-top8  53.48 ms/step   coherence CLEAN
Ronly-top4  53.22           CLEAN
Ronly-top2  53.36           DEGENERATE
Ronly-top0  53.50           DEGENERATE   <- calibration case
```

### 2. Full pipeline — predictor + concurrent movement

```bash
bash bench/pipeline_gate.sh 8      # arg = KT_PREFETCH_BLOCKS
```

Expected:

```
Gate-top8 56.34  Gate-top4 56.34  Gate-top2 56.23  Gate-top0 56.33 ms/step
f = 0.31 experts fetched per layer-call, 69% of coverage free via slot reuse
coherence: tier8 CLEAN, tier2 CLEAN, tier0 DEGENERATE
```

The +2.9 ms over resident-only buys coverage that moves the coherence cliff
left: tier2 eligibility is clean here and degenerate without movement.

## Reading the coherence verdict

`bench/coherence.py` scores **characters**, not words -- 400 repetitions of
`</think>` contain whitespace and fooled a word-level detector twice. Signals:
zlib compression ratio, longest whitespace-free run, distinct 8-grams.

**A run only counts if the calibration line says
`top8 clean = True  top0 degenerate = True`.** top0 keeps no genuine experts and
must collapse; if it does not, the detector or the boot is wrong and the other
rows mean nothing. Accept length is *not* a quality proxy -- degenerate output
inflates it (top0 reaches 3.70 while producing garbage).

Single-prompt rows near the cliff order by noise, not by K: in the gate run
tier4 showed one local `\_\_\_` collapse while tier2 stayed clean. Treat
anything finer than "clean at top8 / degenerate at top0" as needing more
prompts.

## The supporting ladders

| script | question it answers | headline |
|---|---|---|
| `exposure_rootcause.sh` | where do the ms between floor and prefetch-on go? | predictor +1.81, gather +6.87, routing +0.04 |
| `exposure_slope.sh` | barrier or per-byte contention? | intercept ~0.4 ms => no barrier; contention superlinear (2.4 -> 8.6 ms/expert) |
| `blocks_fine.sh` | is the gather's grid tuned? | 2/4/6/8/12/16 -> 66.9/63.0/62.5/**61.95**/62.5/64.6; 8 is optimal |

## Other machines

- **Fewer/smaller GPUs.** Lower `GPU_EXPERTS` until the server boots
  (`scripts/hardware_profile.py` estimates a safe value from free VRAM). The
  *shape* of every result holds -- the floor, the predictor tax and the
  superlinear copy tax are all properties of the design -- but the absolute
  ms/step will differ, and fewer residents means fewer genuine experts survive,
  which moves the coherence cliff right. Re-run the gate; do not port the
  numbers.
- **A faster host-to-GPU link** (e.g. GH200-class C2C) raises the value of
  movement: the copy tax scales with bytes in flight, so the same pipeline with
  a wider fetch becomes affordable there and is not here.
- **Non-CUDA-graph mode** invalidates the comparison entirely; all rows are
  captured-graph numbers.
