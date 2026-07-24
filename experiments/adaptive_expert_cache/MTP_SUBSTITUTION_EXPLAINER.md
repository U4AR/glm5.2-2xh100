# Why MTP Only Pays Off When Experts Are on the GPU

*A mechanism write-up for the GLM-5.2 hybrid CPU/GPU decode server (2×H100 NVL, TP2, sglang + ktransformers).*

## The one-sentence answer

Speculative decoding (MTP) verifies a **fixed 5-token batch every step**. On the
GPU that batch is nearly free (a matmul over 5 rows costs about the same wall-time
as 1 row), so every accepted token turns almost 1:1 into throughput — **+180%**. On
the CPU the same 5 tokens are computed **serially** (~2× the work of 1), so the
heavier verify step almost exactly cancels the tokens it wins back — **+11%, a wash.**

## The setup you need in your head

- GLM-5.2 is a MoE model: 75 MoE layers, 256 routed experts each, **top-8** routing.
- Our engine is **hybrid**: ~104/256 experts are resident on the GPU (W4AFP8 int4);
  the other ~152 live in CPU RAM (AVX512 int4). Every MoE layer computes its
  GPU-resident experts locally and takes a **CPU round-trip** (submit → compute →
  sync → merge) for the non-resident ones.
- Two routing modes (chosen by the `/tmp/kt_topk_mode` sentinel file):
  - **safe2 / plain** — always keep the *genuine* top-2 experts. If one isn't
    GPU-resident, it takes the CPU round-trip. **Correct, but pays CPU.**
  - **sub2 / substitution** — if a genuine top-2 expert isn't resident, **replace**
    it with the best GPU-resident expert. Almost nothing hits the CPU. **Fast, but
    approximate.**
- **MTP (NEXTN, "depth-3")**: `num_draft_tokens = 4`, so every decode step verifies
  a fixed **N+1 = 5** tokens in **one** target forward. Of those, `accept_len`
  (~1.8–4.0) are accepted. Throughput = `accept_len / step_time`.

## The measured data

Step-time is derived as `accept_len / gen_throughput` from the server decode logs.
The striking fact: **in both modes the verify step-time is FLAT across accept_len.**
What differs is its *absolute value*.

**sub2 (routed experts on GPU), MTP on — step-time ≈ 50 ms, flat:**

| accept_len | tok/s | step-time |
|-----------:|------:|----------:|
| 1.80 | 35.5 | 50.7 ms |
| 2.48 | 48.9 | 50.7 ms |
| 3.62 | 73.4 | 49.4 ms |
| 3.95 | 77.7 | 50.9 ms |

**plain (routed experts on CPU), MTP on — step-time ≈ 148 ms, flat:**

| accept_len | tok/s | step-time |
|-----------:|------:|----------:|
| 2.35 | 15.1 | 156 ms |
| 2.50 | 17.5 | 143 ms |
| 3.00 | 20.9 | 144 ms |
| 3.55 | 22.2 | 160 ms |

Canonical benchmark (`decode_bench.sh`, completion_tokens/e2e_latency, 5-run median):

| config | tok/s |
|---|---:|
| plain + MTP | 16.3 |
| sub2 + MTP | 61.5 |
| plain, no MTP | 14.7 (single-step ≈ 68 ms) |
| sub2, no MTP | 22.0 (single-step ≈ 45 ms) |

**Net MTP gain:**

| mode | no-MTP → MTP | gain |
|---|---|---:|
| plain (CPU experts) | 14.7 → 16.3 | **+11%** (wash) |
| sub2 (GPU experts) | 22.0 → 61.5 | **+180%** (full) |

## The mechanism, step by step

### 1. MTP's payoff is one ratio: verify-step time ÷ single-token-step time

MTP replaces a stream of single-token steps with a stream of **fixed 5-token verify
steps**, each yielding `accept_len` tokens. So the question of "does MTP help?"
reduces entirely to:

> How much more expensive is a 5-token verify step than a plain 1-token step?

Call that ratio **R = t_verify / t_single**. You come out ahead whenever
`accept_len > R` — you're paying R single-steps' worth of time to harvest
`accept_len` tokens.

| mode | t_single | t_verify | **R** | typical accept_len | verdict |
|---|---:|---:|---:|---:|---|
| sub2 (GPU) | 45 ms | 50 ms | **≈ 1.1** | ~2.7 | accept_len ≫ R → big win |
| plain (CPU) | 68 ms | 148 ms | **≈ 2.1** | ~2.7 | accept_len ≈ R → wash |

On the GPU you pay ~1.1 steps to win ~2.7 tokens. On the CPU you pay ~2.1 steps to
win the same ~2.7 tokens — the numerator and denominator nearly cancel.

### 2. Why R ≈ 1.1 on the GPU: GEMM batch-invariance

At decode sizes, an expert's forward is a small matrix multiply. Doing it over **5
rows instead of 1** is nearly the same wall-clock time, because the cost is
dominated by **kernel launch and weight-memory movement**, not by the arithmetic on
those few rows. You load the expert's weight tile into the SMs once; whether you
stream 1 activation row or 5 through it barely moves the needle. The 5-token verify
batch runs as **one batched GEMM per expert** → verify ≈ single-token step → R ≈ 1.1.

This is why sub2's step-time is a flat ~50 ms no matter how many tokens land: the
step does the same batched GPU work every time; only the *number accepted* varies,
and that scales throughput directly. `accept_len` converts almost 1:1 into tok/s.

### 3. Why R ≈ 2.1 on the CPU: the expert kernel is batch-linear

The AVX512 CPU expert kernel has no free lunch on batch size — it processes the 5
verify tokens **serially**, so its cost scales with token count (~2× a single
token here, plus the fixed submit/sync round-trip overhead that's paid once per
layer either way). The verify step is genuinely ~2× heavier work. MTP then trades a
~2× heavier step for ~2.7 accepted tokens, and the two nearly cancel → +11%.

Note plain's step-time is *also* flat across accept_len (~148 ms) — the 5 tokens'
CPU compute is incurred whether or not they're accepted. Flatness isn't the
question; the **absolute** verify cost is, and it's set by *where the batched
experts run*: batch-invariant GPU vs batch-linear CPU.

### 4. Why this used to be "MTP is a wash"

Historically MTP looked like a wash **because the verify batch was always
CPU-bound** — there was no way to convert `accept_len` into speed. Two changes
flipped it:

- **`3e39132`** — top-2 substitution moves the routed experts onto the GPU, so the
  verify batch rides the batch-invariant GEMM path (R collapses ~2.1 → ~1.1).
- **`67db44e`** — the verify batch is captured correctly under CUDA graphs (the kt
  verify-batch buffer bug), so the fast path actually runs.

Together they turn the same speculative tokens from "no leverage" into +180%.

## Mental model

> **MTP is a lever, and R is the fulcrum.** MTP always hands you the *same* prize —
> `accept_len` tokens per step. Whether that prize is worth the fixed 5-token verify
> cost depends only on R.
>
> - **GPU experts → R ≈ 1.** The verify step is barely more than a normal step
>   (batched matmul, launch/memory-bound). MTP is "free acceptance" → `accept_len`
>   *is* your speedup.
> - **CPU experts → R ≈ 2.** The verify step is genuinely twice the work (serial
>   loop). MTP buys tokens at nearly the price it pays for the step → break-even.
>
> The verify batch is fixed; the only thing you control is which side of the machine
> pays for it.

You can now predict outcomes without re-benchmarking:

- **Move experts to CPU → your MTP gain evaporates** (R climbs toward accept_len).
- **Keep experts on GPU → MTP pays off, but routing is approximate.**

## The design tension (state it plainly)

The CPU path is **both** the thing that guarantees accuracy (it computes the genuine
top-2 experts) **and** the thing that makes the MTP verify step ~2–3× heavier.
GPU substitution collapses the verify-step cost — unlocking MTP — precisely by *not*
computing the true top-2 (it uses the best resident replacements instead).

So **"fast MTP" and "exact routing" are governed by the same knob and pull in
opposite directions:**

| you want | choose | you get |
|---|---|---|
| exact top-2 routing | plain / safe2 | correct experts, R≈2, MTP ≈ wash (~16 tok/s) |
| maximum speed | sub2 (+ MTP) | approximate experts, R≈1, MTP full win (~61 tok/s) |

There is no setting that is simultaneously exact **and** MTP-accelerated on this
hardware, because MTP's leverage comes *from* getting the experts off the batch-linear
CPU — which is exactly what makes routing approximate. Any future "exact + fast"
design has to attack R directly: make the CPU expert kernel batch-invariant (batch
the 5 verify tokens into one wider GEMM/GEMV per expert), or make the genuine top-2
resident on the GPU so no substitution is needed. Placement tricks alone won't do it
— they rebalance the fixed submit/sync overhead, they don't change R.
