# RunGLM — GLM‑5.2 (754B) on two GPUs

Portable deployment now auto-detects conservative TP2 profiles. The first
non-H100 target is **2× L40/L40S with sufficient host RAM**; the measured
performance reference remains 2× H100 NVL. For a fresh machine:

```bash
git clone --branch experiment/adaptive-decode-cache --single-branch \
  https://github.com/U4AR/glm5.2-2xh100.git RunGLM
cd RunGLM
export RUNGLM_ALLOW_AVX2=1   # required only on AVX2-only hosts such as this L40 pod
INSTALL_SYSTEM_DEPS=1 ./setup.sh
python int4_scripts/download_w4afp8.py
./run_adaptive.sh
```

Read [PORTABLE_DEPLOYMENT.md](PORTABLE_DEPLOYMENT.md) for the RunPod recipe,
preflight behavior, hardware profiles, and current support boundary. Automatic
profiles establish safe first-boot capacity; throughput claims below remain
specific to the hardware on which they were measured.

Adaptive boot uses the committed hot-core ranking and matching normalized
frequency prior. The prior has only a 64-event bootstrap mass: it makes startup
ordering deterministic without resisting live decode-time convergence.

Serve **GLM‑5.2**, a 754B‑parameter MoE model, on a single dual‑H100 box using
**SGLang + KTransformers (kt‑kernel)** heterogeneous CPU+GPU Mixture‑of‑Experts.
A custom packed‑INT4 AVX‑512 CPU kernel + INT4 GPU experts, **top‑2 expert
substitution**, and **NEXTN/MTP speculative decode** push single‑stream decode to a
**measured ~40.5 tok/s** (2.75× the 14.7 tok/s plain baseline) while fitting the
CPU‑side experts in **~250 GB of RAM**.

The server is **OpenAI‑compatible** (`/v1/chat/completions`, streaming, reasoning
separation) and ships with a zero‑dependency browser chat UI.

## TL;DR — run the fast server

```bash
./run_fast.sh          # DEFAULT: top‑2 substitution + MTP depth‑3  → ~40.5 tok/s decode
./start_ui.sh          # browser chat UI on :8080  → open the forwarded port
```

`run_fast.sh` is the recommended way to run this. It defaults to the fastest coherent
config — the **top‑2 + MTP** mode that this repo's currently‑deployed server runs.
Everything else below (the plain INT4 path, the FP8 path) is slower and is kept for
comparison / lower‑VRAM boxes.

> The full engineering story — what was tried, what worked, and why — is in
> **[BLOG.md](BLOG.md)**, with focused write‑ups in
> [BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md) (the top‑2 trick) and
> [BLOG_MTP_CUDAGRAPH.md](BLOG_MTP_CUDAGRAPH.md) (MTP under CUDA graphs).

---

## Results — speed vs. the top‑N knob

The single biggest speed lever is **`KEEP`** (a.k.a. top‑N): how many of each token's
genuinely most‑important experts to keep on the slow CPU path. The rest of the 8 routed
experts are **substituted with the best GPU‑resident experts**, so fewer experts hit
the CPU. Lower `KEEP` = faster, with a quality trade‑off at the extreme. **MTP**
(NEXTN speculative decode, depth‑3) stacks on top for a further ~1.5×.

Decode tok/s on 2×H100‑NVL / AMD EPYC Zen4, single stream (the default row is a fresh
5‑run measurement on the live `GPU_EXPERTS=96` / 128k server; the rest are 5‑run medians
at `GPU_EXPERTS=104`):

| Command | `KEEP` (top‑N) | MTP | Decode | vs. baseline | Quality |
|---|:--:|:--:|---:|:--:|---|
| `./run_fast.sh` **(default)** | **2** | depth‑3 | **~40.5 tok/s** | **2.75×** | mostly clean; rare loops |
| `KEEP=4 ./run_fast.sh` | 4 | depth‑3 | ~29 tok/s (est.) | 2.0× | clean (baseline‑equivalent) |
| `KEEP=0 ./run_fast.sh` | 0 | depth‑3 | ~40 tok/s | 2.7× | drifts — not recommended |
| `MTP=0 ./run_fast.sh` | 2 | off | ~22 tok/s | 1.5× | mostly clean |
| `KEEP=4 MTP=0 ./run_fast.sh` | 4 | off | ~18.5 tok/s | 1.25× | clean |
| `MODE=off ./run_fast.sh` | — | depth‑3 | ~17.5 tok/s | 1.2× | clean (no substitution) |
| `MODE=off MTP=0 ./run_fast.sh` | — | off | ~14.7 tok/s | 1.0× | clean (plain INT4 baseline) |

The default measured **40.46 tok/s** (5×300 tok, min 40.34 / max 40.60 — dead steady),
MTP accept‑length ~2.7–3.9. Bench your own box with the two harnesses:

```bash
python3 bench/perf_probe/decbench.py 300 5   # pure decode tok/s (what the 40.5 above is)
bash    bench/decode_bench.sh 6 256          # ~33 tok/s — folds one‑time prefill into a
                                             # short run, so it reads lower; not a regression
```

### 2×L40 (non‑Hopper) — and the fp8‑KV / MTP pitfall

Measured 2026‑07‑25 on 2×L40 46 GB + dual EPYC 7773X (AVX2, no AVX‑512), plain
`safe2` routing + MTP depth‑3, `GPU_EXPERTS=24`:

| workload | accept | tok/s |
|---|:--:|---:|
| general‑prompt suite (10 prompts, aggregate) | 3.02 | **18.9** |
| structured / JSON output | 3.58 | 24.3 |
| code generation | ~3.0–3.3 | 19–22 |
| technical prose (`decbench.py`) | 2.63 | 16.5 |

Two settings account for a **+46%** swing over the previous 2×L40 defaults (12.9 tok/s):

1. **`kv_cache_dtype` must be `auto` (bf16) on any non‑`flashmla` attention
   backend.** fp8 KV is only validated on flashmla; on the Triton path that
   non‑Hopper cards fall back to, it measurably degrades **MTP/NEXTN acceptance**
   (accept 2.2 → 3.0+, tok/s 12.9 → 16.5). This presents as "MTP accept length
   collapsed on this machine" and is easy to misattribute to the CPU‑expert path.
   MLA makes the fix nearly free — `kv_lora_rank=512`, so bf16 KV at 8192 tokens
   costs under 1 GB. `hardware_profile.py` now defaults this automatically.
2. **`CPUINFER` should saturate memory bandwidth, not core count.** The CPU‑expert
   path is DRAM‑bandwidth‑bound. Measured streaming‑read sweep on this host:
   8 thr 163 GB/s · 16 thr 248 · 28 thr 296 · **56 thr 356** · 112 thr 271 — it
   peaks near 56 and *regresses* past it. Going 28 → 56 cut decode step time
   170 ms → 152 ms. Re‑measure per host; more threads is not monotonically better.

Note `decbench.py`'s technical‑essay prompt is the **worst case** for MTP
acceptance (2.63 vs 3.58 on structured output), so it understates real
coding‑agent throughput by ~30%. Cross‑check with a code or JSON prompt.

### Expert placement: `hotcore` is the default, and `uniform` was the trap

`--kt-expert-placement-strategy uniform` — the old default — spreads the expert
budget evenly *across layers* but then fills each layer with experts **0..N‑1 by
index**. Nothing about placement was informed by routing: expert 0 was resident
in all 75 layers, expert 200 in none. Top‑2 coverage was therefore just `N/256`
= **12.7%** at `GPU_EXPERTS=30`.

Routing is in fact strongly concentrated — but **the hot set differs per layer**.
Measured over 441k genuine‑top‑2 events (`hot_core_prior.pt`):

| N/layer | index 0..N‑1 (`uniform`) | best single global set | per‑layer hottest‑N (`hotcore`) |
|---:|---:|---:|---:|
| 24 | 10.0% | 14.0% | 51.3% |
| **30** | **12.7%** | 17.2% | **56.8%** |
| 104 | 41.0% | 49.9% | 90.3% |

The middle column is the point: the best *globally* hot set of 30 reaches only
17.2%, barely above arbitrary. Per‑layer placement reaches 56.8%, cutting CPU
expert round trips from 1.75 to 0.86 per token.

`PLACEMENT=hotcore` (now the default in `run_server_int4.sh`) slices each layer's
hottest‑N at boot from the committed, N‑agnostic ranking
`experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt`.

**The measured win is small — about +2%, not the +8% first reported here.** In the
only controlled comparison (both arms booted back‑to‑back in one sweep, same
machine state) it is **19.64 → 20.11 tok/s, +2.4%**, accept unchanged at ~3.1.
An earlier figure of +7.6% came from comparing runs booted hours apart and was
mostly **cross‑boot drift**: repeated hotcore boots alone span 20.11–21.26 tok/s,
a spread larger than the effect being measured. Never A/B this box across
separate boots; interleave the arms.

**Output is identical** — under `safe2` the genuine top‑2 always compute, so
placement changes only *where*, never *which* (accept lengths matched
prompt‑for‑prompt across placements). It is kept on because it is free and
strictly better on paper; just don't expect it to show up as a big number.

It is safe to leave on everywhere: if the ranking file is missing or its shape
does not match the model, it logs a warning and falls back to `uniform`
(verified by booting with a bogus `KT_HOTCORE_RANKING_PT`). `PLACEMENT=uniform`
restores the old behaviour. Rebuild the ranking for another workload with
`experiments/adaptive_expert_cache/decode_cache/build_hot_core.py`.

### Why placement barely moves the needle: the CPU path costs a *fixed* toll, not a per‑expert one

A budget sweep under `hotcore` (all four points on the 9‑prompt suite) shows
throughput is **nearly flat in CPU expert traffic**:

| config | coverage | CPU trips/token | tok/s | ms/token |
|---|---:|---:|---:|---:|
| hotcore N=8 | 29.1% | 1.42 | 18.19 | 54.98 |
| hotcore N=16 | 42.1% | 1.16 | 20.33 | 49.19 |
| hotcore N=30 | 56.8% | 0.86 | 20.11 | 49.73 |
| uniform N=30 | 12.7% | 1.75 | 19.64 | 50.92 |

Over a **2× range in CPU trips/token (0.86 → 1.75), ms/token moves only
49.7 → 50.9**. N=16 and N=30 are indistinguishable. (N=8 is the outlier and is
confounded: with only 8 resident experts the six substituted slots come from a
much poorer pool, so it generates different text at a lower accept length —
don't read it as a traffic effect.)

The `[kt-time]` stage breakdown agrees: per MoE layer, `cpu_wait` is **0.04 ms of
a 0.95 ms layer** — the CPU expert work is almost entirely hidden behind GPU
compute. The layer cost is instead `submit 0.18 / mask 0.17 / gpu 0.50 /
sync 0.08 / merge 0.03`, i.e. ~0.45 ms/layer of **fixed host overhead that no
placement can remove** — roughly 34 ms/step across 75 MoE layers. (Caveat: those
samples come from graph‑capture/prefill contexts, because the hooks sit in the
Python `apply()` which CUDA‑graph replay bypasses entirely — there is currently
**no steady‑state intra‑step attribution** available in the graphs‑on regime.)

**But the CPU path is *not* cheap — it is just insensitive to how much work you
give it.** Measured `top0` (`KEEP=0` + `/tmp/kt_skip_cpu`, all 8 slots filled from
GPU‑resident experts, CPU path skipped entirely; output incoherent by
construction, this is a ceiling probe only):

| config | tok/s | accept | ms/token | ms/step |
|---|---:|---:|---:|---:|
| `safe2` hotcore N=30 | 20.11 | 3.10 | 49.73 | 154.2 |
| `safe2` uniform N=30 | 19.64 | 3.07 | 50.92 | 156.3 |
| **`top0` — no CPU path** | **42.77** | 3.47 | 23.38 | **81.1** |

Removing the CPU path is worth **2.13× on tok/s / 1.90× on step time**. So it
costs **73 ms of a 154 ms step — 47% — or 0.97 ms per MoE layer.** Decomposing
that against the traffic sweep above:

* the part that **scales with expert count** is ~2.4 ms per trip/token, so at
  0.86 trips/token only **~2 ms/step**;
* the remaining **~71 ms/step (97%) is fixed** — paid per layer whether the CPU
  handles two experts or none.

A bandwidth cross‑check confirms it is not streaming: 0.86 trips/token × 4 draft
tokens × 19.46 MB = 67 MB/layer/step, which at the measured 356 GB/s is 0.19 ms/layer
= **14 ms/step of real DRAM traffic** against 73 ms/step of measured cost. Cutting
uniform→hotcore should have saved ~15 ms/step on bandwidth alone and saved 2.
The expert streaming is already hidden; what is exposed is the per‑layer
`submit_forward` + cross‑stream `sync` round trip, ~1 ms × 75 layers.

So the H100‑era conclusion in
`experiments/expert_footprint_top2_vs_top8/ORACLE_CEILING.md` holds here and is
now quantified: **per‑layer cost is `max(cpu_time, gpu_time)` plus a fixed
dispatch toll, and the toll — not the expert count — is what you are paying.**
That is the whole reason a 4.5× coverage improvement bought 2%.

The next lever is therefore **the per‑layer host round trip**, not placement,
not coverage, not more VRAM. Options worth measuring: a per‑layer "all routed
experts are resident → skip the CPU call" gate (only ~1% of layers qualify at
57% coverage, since it needs all 2×4 slots resident — but ~43% would qualify at
the H100's 90%), coalescing submit/sync across layers, or cutting the
`cudaLaunchHostFunc` host‑node round trip that CUDA graphs impose per layer.

Two levers that do **not** pay off at 2×L40, both measured — don't re‑litigate:

* **Raising `GPU_EXPERTS` is nearly exhausted.** Under `safe2` only the genuine
  top‑2 reach the CPU, so CPU traffic scales as `2 × (1 − residency)`. Going
  24 → 30 moves that just 1.81 → 1.77 experts/token: +3% (18.9 → 19.6 tok/s).
  `32 @ mem_fraction 0.91` OOMs during CUDA‑graph capture. Mattering would need
  residency near H100's 40% (104/256), which 46 GB cards cannot hold. (Those
  numbers are under `uniform`; `hotcore` raises the coverage *at* N=30 instead
  of needing a bigger N, which is why it wins where raising N did not.)
* **`./run_adaptive.sh` (decode‑time expert cache) is net‑negative here.** On a
  diverse 9‑prompt suite it scored 18.2 / 18.2 / 19.0 tok/s across three passes
  vs **19.6 without it**: `top2_cov` plateaus at ~0.51 (the working set keeps
  moving) while 122 layer swaps × 395 ms burned 48 s, ~16% of wall time. Its
  earlier "+13%" was measured against a slower 170 ms step *and* on `decbench`'s
  single repeated prompt — the best case for a cache. It may still help a
  long single‑task session; it hurts mixed traffic.

**Quality benchmark (LiveBench reasoning, 200 Qs, one at a time):** with the
server running, one portable command scores the top‑2 tier end‑to‑end (dataset
auto‑pulled from HuggingFace, so it works on a fresh box):

```bash
./bench/run_benchmark.sh                     # full suite, top‑2 (GLM5.2-top2)
MODEL=GLM5.2-top8 ./bench/run_benchmark.sh   # baseline tier, for A/B
LIMIT=20 ./bench/run_benchmark.sh            # quick 20‑question smoke test
```

See [bench/livebench/README.md](bench/livebench/README.md) for options and output format.

Freeze GLM‑5.2's own LiveBench outcome into a pass/fail label set (analogous to
Terminal‑Bench's `task_labels.txt`) so later runs can be diffed against it:

```bash
python bench/livebench/make_labels.py livebench_results.json          # write labels
python bench/livebench/make_labels.py new_run.json --compare          # diff vs labels
```

**Agentic benchmark (Terminal‑Bench 2.0, executed pass/fail):** one command runs
the real Harbor harness (a Docker container + graders per task) against the live
server and scores it against GLM‑5.2's original labels. Strictly sequential (one
task at a time, like LiveBench) — one GPU serves the agent, and concurrency
starves decode into false timeouts:

```bash
./bench/run_terminalbench.sh                 # all 42 tasks, one at a time
TASK=fix-git ./bench/run_terminalbench.sh    # one task only
```

Portable: on a **fresh machine** the first run bootstraps the harness itself —
it creates a venv, installs the pinned `harbor==0.16.1` (the terminus‑2 agent
ships inside it), and clones the `terminal-bench-2` task set into `TB_DIR`
(default `.terminalbench/`). Nothing to copy over; only Docker (rootless) and a
running server are prerequisites. The reference labels + verdict aggregator live
in [`bench/terminalbench/`](bench/terminalbench/). To pre‑warm without running:

```bash
./bench/terminalbench/setup_terminalbench.sh
```

> The two harnesses disagree on purpose: `decbench.py` reports steady‑state decode;
> `decode_bench.sh` amortizes the prefill of a short 256‑token request into the rate, so
> it always reads a few tok/s lower. Use `decbench.py` for the decode headline.

> **Quality note:** `KEEP=2` (the default) is coherent on the vast majority of prompts
> but can occasionally fall into a repetition loop on open‑ended generation. If you need
> guaranteed baseline‑equivalent quality, use `KEEP=4`. `KEEP=0` is the speed ceiling
> but degenerates — don't use it for real output. Details in
> [BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md).

### Per‑request "intelligence" tier (pick speed at call time)

The top‑N knob is also selectable **per request, live, via the OpenAI `model`
field** — no restart, no global flag. Append `-topN` (N = 0…8) to the model name:

```bash
curl localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top8", ...}'  # max quality (baseline)
curl localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top2", ...}'  # fast (default)
```

`-top8` is bit‑identical to the baseline; `-top2` is the fast default; any N works.
Different tiers can be **mixed in one batch / one CUDA graph** (the tier is a
per‑token value, so there is **no graph recapture** when it changes). The bare
name (`GLM5.2`) uses the server's default tier. The chat UI exposes it as an
"intelligence" dropdown. Full write‑up: [BLOG_INTELLIGENCE_TIER.md](BLOG_INTELLIGENCE_TIER.md).

**Batched throughput** (top‑2, throughput‑tuned `GPU_EXPERTS=96 MAX_RUNNING=8
CUDA_GRAPH_MAX_BS=8`), aggregate decode tok/s by concurrent requests:

| concurrent | + MTP | no‑MTP |
|---|---:|---:|
| 1 | 32 | 21 |
| 2 | 44 | 34 |
| 4 | 56 | 55 |
| 8 | 73 | 78 |

MTP is a single‑user win (harvests a lone stream's idle headroom); past ~4
concurrent users its verify cost stops paying for itself — run `MTP=0` for a busy
multi‑user endpoint.

Underlying precision options (both serve the same model quality; INT4 is faster **and**
needs ~half the RAM since the CPU‑expert path is memory‑bandwidth bound):

| Precision | CPU experts | GPU experts/layer | RAM needed | Plain decode |
|---|---|---:|---:|---:|
| **INT4 (default)** | packed RAWINT4 (AVX‑512 VNNI) | 96–104 | **~250 GB** | ~14.7 tok/s (before top‑2/MTP) |
| FP8 ("8‑bit") | block‑FP8 | 48 | ~629 GB (or NVMe swap) | ~8.7 tok/s |

---

## Hardware requirements

- **GPUs:** 2× H100 **NVL (96 GB)**, TP2. At `GPU_EXPERTS=104` each card uses
  ~88 GB. (80 GB H100s work with fewer GPU experts — more on CPU, more RAM, a touch
  slower.) Validated on driver 575 / CUDA 12.8.
- **CPU:** x86‑64 with **AVX‑512 VNNI** (this was tuned on an AMD EPYC 9V84 / Zen4,
  80 cores, 2 NUMA nodes). The packed‑INT4 kernel uses the AVX‑512 EVEX VNNI form;
  it does **not** require Intel AMX or 256‑bit `avx_vnni`.
- **RAM:** **~250 GB** for the INT4 config at `GPU_EXPERTS=104` (this is the binding
  constraint — anything ≥250 GB works; more lets you push more experts to CPU). The FP8
  config needs ~629 GB or NVMe swap.
- **Disk:** ~380 GB of fast scratch (NVMe) for the INT4 expert weights.
- **Software:** Linux, NVIDIA driver ≥ 575 / CUDA 12.8+, Python 3.12. The Python
  environment (SGLang fork + kt‑kernel + the patches below) is built in [Setup](#setup).

---

## What's in this repo

| File | Purpose |
|---|---|
| `run_fast.sh` | **The recommended launcher** — top‑2 substitution + MTP depth‑3 (~40.5 tok/s) |
| `run_server_int4.sh` | Underlying **INT4** launcher (plain baseline ~14.7 tok/s; `run_fast.sh` wraps it) |
| `run_server.sh` | Launch the **FP8** server (the 8‑bit path) |
| `chat_ui.py` / `start_ui.sh` | Zero‑dep browser chat UI → OpenAI endpoint |
| `chat_template.jinja` | GLM‑5.2 chat template (wired into both launchers) |
| `int4_scripts/` | Weight download / GPTQ repack / offline kernel tests |
| `bench/decode_bench.sh` | Decode throughput benchmark |
| `BLOG.md` | The full write‑up of the optimization journey |
| `BLOG_TOP2_EXPERTS.md` | The top‑2 expert‑substitution study (~1.5× faster decode) |
| `*_HANDOFF.md`, `PERF_CUDA_GRAPHS.md` | Deep‑dive engineering notes |

The Python environment is **not** committed (it's ~tens of GB), but the source‑level
patches that make this build work **are** tracked in‑tree under
`.venv/lib/python3.12/site-packages/` so [Setup](#setup) can restore them onto a fresh
venv:

- `kt_kernel/kt_kernel_ext*.so` — the custom kt‑kernel build, symbol
  `AVX512RawInt4Packed_MOE` (rebuild on your own CPU; see Setup).
- `sglang/.../quantization/w4afp8.py` — the `topk_ids == -1` remap fix for INT4 GPU
  experts under `ep_size=1` (without it the GPU MoE NaNs → garbage).
- `sglang/.../models/deepseek_v2.py`, `.../attention/nsa_backend.py`,
  `.../moe/kt_ep_wrapper.py`, `.../model_executor/cuda_graph_runner.py`,
  `.../attention/triton_backend.py` — supporting / MTP‑under‑CUDA‑graph fixes.

---

## Setup

These steps build the Python environment once. All paths below are **relative to the
repo**, so clone it anywhere — every launcher resolves `.venv`, `chat_template.jinja`,
the weights, etc. from its own location.

> **One place for paths.** Every launcher sources [`config.sh`](config.sh) at
> startup — that's the single file to edit if your weights don't live under
> `./weights/`. Set `WEIGHTS_DIR` (the parent dir of the checkpoints) there once and
> it applies to `run_fast.sh`, `run_server.sh`, `run_server_int4.sh`, the bench/
> experiment wrappers, and the `int4_scripts/` harnesses (via `int4_scripts/_paths.py`).
> Every value is also overridable per-invocation via env, e.g.
> `WEIGHTS_DIR=/mnt/nvme ./run_fast.sh`. The default `./weights` can be a real
> directory or a symlink to fast scratch (`ln -s /mnt/nvme/models ./weights`).

**This does not use stock `sglang` from PyPI.** It uses **KTransformers**, which
bundles its *own* SGLang fork (the `kvcache-ai/sglang` submodule, installed as the
package **`sglang-kt`**) plus the `kt-kernel` heterogeneous CPU+GPU MoE engine. On top
of that, this repo carries a handful of source patches (the INT4/NSA/MTP fixes listed
above). So setup is: install KTransformers into a venv, then overlay this repo's patches.

For a **bit-exact** rebuild, the validated component versions are: Python **3.12.9**,
torch **2.9.1+cu128**, transformers **5.12.1**, CUDA **12.8**, `kt-kernel` **0.6.2.post3**,
KTransformers branch **`U4AR/ktransformers` `glm5.2-2xh100-stable` (commit `512802b`)**
with the SGLang submodule pinned to **kvcache-ai/sglang @ `51032b712`**. The full freeze
is in [`requirements-lock.txt`](requirements-lock.txt). (The "Phala" you may have seen is
the *weights* — `PhalaCloud/GLM-5.2-W4AFP8` — not the SGLang code.)

```bash
# 1. Clone this repo (anywhere). Its tracked patches come down with it.
git clone <this-repo-url> RunGLM
cd RunGLM
export REPO=$(pwd)            # used in the examples below

# 2. Get the KTransformers sources (SGLang-kt + kt-kernel). It is NOT committed
#    here (gitignored). The GLM-5.2 packed-INT4 kernel + W4AFP8 loader are fork-only
#    work (not on kvcache-ai/ktransformers main), so use the dedicated stable branch
#    below. `--recursive` pulls the SGLang submodule pinned to kvcache-ai/sglang
#    @ 51032b712.
git clone --recursive -b glm5.2-2xh100-stable \
  https://github.com/U4AR/ktransformers.git ktransformers
cd ktransformers && git submodule update --init --recursive && cd "$REPO"
#    glm5.2-2xh100-stable contains commit 512802b9025d149681401f1c63519afa8caa34ea,
#    adding the packed AVX2 RAWINT4 backend. The validated AVX-512/H100 path is
#    unchanged. Do NOT use failed-be/* branches — those are abandoned streaming
#    experiments.

# 3. Create the Python 3.12 venv the launchers expect at ./.venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 4. kt-kernel needs hwloc + libnuma (dev headers at BUILD time, .so at runtime),
#    so set them up BEFORE the build in step 5.
#    With sudo:  sudo apt install -y libhwloc-dev libnuma-dev pkg-config
#    No sudo? Build hwloc into the venv prefix (this is what was validated here):
#      curl -fsSL -o hwloc.tar.gz \
#        https://download.open-mpi.org/release/hwloc/v2.11/hwloc-2.11.2.tar.gz
#      tar xf hwloc.tar.gz && cd hwloc-2.11.2
#      ./configure --prefix="$VIRTUAL_ENV" && make -j"$(nproc)" install && cd "$REPO"
#      export PKG_CONFIG_PATH="$VIRTUAL_ENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
#    Ensure $VENV/lib is on LD_LIBRARY_PATH for any process importing kt_kernel. The
#    launchers `source .venv/bin/activate`, so bake it into activate:
#      echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"' >> .venv/bin/activate

# 5. One-click install: submodules -> SGLang fork (sglang-kt) -> kt-kernel,
#    compiled for YOUR CPU (auto-detects AVX-512 VNNI/BF16/VBMI). Needs CUDA 12.8+
#    toolkit + a working nvcc. Validated on torch 2.9.1+cu128 / transformers 5.12.1
#    / kt-kernel 0.6.2.post3.
CPUINFER_USE_CUDA=1 ./ktransformers/install.sh        # `all` is the default
#    (kt-kernel only:  ./ktransformers/install.sh kt-kernel ;
#     for a different/older target CPU build with --manual — see install.sh -h)

# 6. Pin the rest of the wheels to the exact versions validated here (torch
#    2.9.1+cu128, transformers 5.12.1, triton 3.5.1, flashinfer 0.6.3, ...). The two
#    local packages (sglang-kt, kt-kernel) are already installed by step 5.
pip install -r requirements-lock.txt

# 7. Overlay THIS repo's patches on top of the freshly installed sglang-kt /
#    kt-kernel (they live in-tree under .venv/... and were just overwritten).
git checkout -- .venv

# 8. Sanity check.
kt doctor
```

> Step 6 matters: `install.sh` writes a clean `sglang-kt` + `kt-kernel` into `.venv`;
> `git checkout -- .venv` then restores the INT4-GPU remap, NSA, and MTP-under-CUDA-graph
> fixes this repo tracks on top of them. The compiled `kt_kernel_ext*.so` tracked here
> was built for an AMD Zen4 (AVX-512 VNNI, no AMX) CPU — if yours differs, keep the one
> step 5 just built for your machine instead of restoring the tracked `.so`.

> The launchers hard-code `VENV=.../.venv` relative to the repo and `source` it on
> startup, so once `./.venv` exists you don't activate it by hand. **Always** go
> through the launchers (or `source .venv/bin/activate`) — kt‑kernel needs
> `LD_LIBRARY_PATH` to find hwloc/libnuma.

---

## Quickstart

### 1. Get the weights

INT4 path uses Phala's W4AFP8 export (4‑bit experts + FP8 non‑experts), ~373 GB.
By default it downloads to `./weights/GLM-5.2-W4AFP8` (the path `config.sh` expects).
To put it on fast scratch (NVMe, ~380 GB free) either symlink `./weights` there, or
override the destination — pick **one place**:

```bash
# Default: downloads to ./weights/GLM-5.2-W4AFP8 (resumable, NFS-safe).
python int4_scripts/download_w4afp8.py

# Or send it elsewhere (both env vars are honored by config.sh too):
WEIGHTS_DIR=/path/to/nvme python int4_scripts/download_w4afp8.py   # -> /path/to/nvme/GLM-5.2-W4AFP8
WEIGHTS=/exact/dir/GLM-5.2-W4AFP8 python int4_scripts/download_w4afp8.py
```

The downloader disables hf‑xet and uses one worker by default (safe on NFS mounts,
where xet's parallel writes stall). On a fast local disk, bump throughput with
`HF_MAX_WORKERS=16 python int4_scripts/download_w4afp8.py`.

The FP8 path instead uses the original GLM‑5.2 FP8 checkpoint; put it at
`./weights/GLM-5.2-FP8` (or set `FP8_MODEL` in `config.sh`). It's also a source of
`chat_template.jinja`, already vendored in this repo.

### 2. Run the fast server (recommended — top‑2 + MTP, ~40.5 tok/s)

```bash
./run_fast.sh                       # DEFAULT: KEEP=2 + MTP depth‑3
```

Boot takes ~2–4 min (loads INT4 experts from NVMe + CUDA‑graph capture + MTP draft
warmup). Wait for `The server is fired up and ready to roll!`, then the OpenAI API is
live on `:8000`. `run_fast.sh` wraps `run_server_int4.sh` with the winning INT4 recipe,
writes the top‑K reroute sentinel `/tmp/kt_topk_mode`, and enables NEXTN/MTP.

**Change the top‑N knob** (and stack MTP on/off) — see the **Results** speed table
above for the numbers:

```bash
./run_fast.sh                 # KEEP=2 + MTP   → ~40.5 tok/s [default, recommended]
KEEP=4 ./run_fast.sh          # safer quality  → ~29 tok/s   (baseline‑equivalent)
KEEP=0 ./run_fast.sh          # max speed      → ~40 tok/s   (quality drifts — avoid)
MTP=0 ./run_fast.sh           # top‑2, no MTP  → ~22 tok/s
MODE=off ./run_fast.sh        # plain INT4 + MTP → ~17.5 tok/s
MODE=off MTP=0 ./run_fast.sh  # plain INT4 baseline → ~14.7 tok/s
```

You can switch `KEEP` live without a restart by rewriting the sentinel, e.g.
`printf sub4 > /tmp/kt_topk_mode` (the worker re‑reads it per request).

### 2b. Production config — long context + chat UI (what the live server runs)

The default boots a small 4096‑token window for benchmarking. The deployed server (the
one behind the chat UI) runs a real 128k context for coding agents and long chats, at
`GPU_EXPERTS=96` so the bigger KV pool fits:

```bash
GPU_EXPERTS=96 \
CONTEXT_LENGTH=131072 \
MAX_TOTAL_TOKENS=131072 \
MEM_FRACTION=0.94 \
./run_fast.sh                       # top‑2 + MTP, 128k context  → ~40.5 tok/s decode

./start_ui.sh                       # chat UI on :8080 (proxies → :8000)
```

This is the exact configuration of the currently‑running server: top‑2 + MTP depth‑3,
`GPU_EXPERTS=96`, 128k context, dense MLA attention (`DISABLE_NSA=1`), GPU bulk prefill
(`KT_GPU_PREFILL_THRESHOLD=2048`). See **2c. Long‑context mode** below for what the two
long‑context fixes do.

### 2c. Long‑context mode (for coding agents / long chats)

The production config in **2b** already runs 128k context *with* top‑2 + MTP. If you
instead want the **plain baseline** at long context (no substitution, no MTP — e.g. to
A/B quality), drive `run_server_int4.sh` directly:

```bash
# MODEL / KT_WEIGHT_PATH default to ./weights/GLM-5.2-W4AFP8 via config.sh;
# set them (or WEIGHTS_DIR) only if your weights live elsewhere.
KT_METHOD=RAWINT4 \
KT_RAWINT4_BACKEND=avx512_packed \
GPU_EXPERTS=96 \
CONTEXT_LENGTH=131072 \
MAX_TOTAL_TOKENS=131072 \
MEM_FRACTION=0.93 \
CPUINFER=72 \
MAX_RUNNING=2 \
bash run_server_int4.sh
```

Verified: coherent to 12k+ tokens, decode steady at **~14.6 tok/s** (plain baseline)
even at 12k+ positions, ~88 GB/card, `available_gpu_mem≈8 GB` at 128k. Decode speed is
unchanged vs the short‑context config (we are CPU‑MoE bound, so attention length is
free); top‑2 + MTP from **2b** then lifts it to ~40.5 tok/s.

Two correctness fixes are baked into `run_server_int4.sh` and are **on by default**
for this path — both were latent bugs that only a real long‑context / agent workload
triggers:

- **`DISABLE_NSA=1` (default) — fixes gibberish past ~2048 tokens.** GLM‑5.2 uses
  DeepSeek Sparse Attention (a lightning indexer picks the top `index_topk=2048`
  tokens). This sglang build runs *every* layer through that sparse path (it lacks
  the newer per‑layer `index_topk_pattern` Full/Sparse support), and the sparse
  kernels produce garbage once the sequence exceeds 2048 (hard cliff: 2041 ok, 2061
  gibberish) — and the topk kernels hard‑assert `topk==2048`, so the window can't be
  widened. The fix overrides `index_topk=null` (→ `is_deepseek_nsa()` false →
  **full dense MLA attention**, backend `flashmla`). Dense is the most accurate path
  (sparse only approximates it) and costs ~nothing here since attention isn't the
  bottleneck. Set `DISABLE_NSA=0` to restore native (buggy >2048) NSA.
- **`KT_GPU_PREFILL_THRESHOLD=2048` (default) — fast GPU bulk prefill.** A prefill
  chunk with ≥ this many tokens streams **all 256 experts** to the GPUs and runs the
  chunk through the cutlass W4A8 kernel, instead of computing the CPU experts on the
  AVX‑512 path. Measured **~1.9× faster TTFT** on large prompts (2660 tokens: ~20 s
  vs ~37 s on the CPU path), with **decode unaffected** (~14 tok/s) and output
  bit‑coherent. It needs ~5 GB of free VRAM for a transient 256‑expert scratch layer
  (fits at `GPU_EXPERTS=96` / 128k, `available_gpu_mem ≈ 8 GB`). Set
  `KT_GPU_PREFILL_THRESHOLD=0` to fall back to pure CPU prefill (~74 tok/s; SGLang's
  radix cache still makes follow‑up turns instant, a 12k prefix re‑prefills in ~0.6 s).

  > Two prerequisites were fixed to enable this path (both checked in): (1) W4AFP8
  > `create_weights` used to `assert` on a `weight_loader` kt‑kernel doesn't pass →
  > `AssertionError`/SIGQUIT on any prompt over the threshold (this is the crash a
  > coding agent hit on its first request); (2) the packed RAWINT4 CPU backend's
  > `write_weights_to_buffer` (the routine that streams an expert's packed int4 +
  > bf16 scales into the GPU staging buffer) was an upstream `"not yet implemented"`
  > stub — implemented here as a direct copy of the on‑disk‑identical packed layout.
  > **Rebuild the kernel** (`./install.sh build` with `CPUINFER_USE_CUDA=1`) to pick
  > these up.

> Tip: point your OpenAI‑compatible coding agent (e.g. `pi`, Continue, aider) at
> `http://<host>:8000/v1` with model `GLM5.2`. Use **streaming** and a sane
> `max_tokens`; avoid non‑streaming requests with huge `max_tokens` (they hold the
> connection for the whole generation).

### 2d. Run the FP8 server (8‑bit alternative — ~8.7 tok/s, needs ~629 GB / swap)

```bash
# MODEL defaults to ./weights/GLM-5.2-FP8 via config.sh (set FP8_MODEL there, or
# MODEL=... here, if it lives elsewhere).
GPU_EXPERTS=48 bash run_server.sh
```

### 3. Chat UI (optional)

```bash
./start_ui.sh          # serves http://localhost:8080, proxies -> :8000
```

Open the forwarded port 8080. The UI streams answers, shows the model's reasoning in
a collapsible panel, and reports tok/s.

---

## Using the OpenAI API

The server speaks the OpenAI Chat Completions API. GLM‑5.2's reasoning is returned in
a separate `reasoning_content` field (the answer is in `content`).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM5.2",
    "messages": [{"role": "user", "content": "Explain MoE routing in two sentences."}],
    "max_tokens": 512,
    "temperature": 0.6,
    "stream": true
  }'
```

Python (official `openai` client):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="GLM5.2",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    max_tokens=512, temperature=0.6,
)
msg = r.choices[0].message
print("reasoning:", getattr(msg, "reasoning_content", None))
print("answer:", msg.content)
```

> GLM‑5.2 reasons heavily; give it a generous `max_tokens` (≥256) or the answer may
> be truncated while it is still thinking.

---

## Tuning

The one knob that matters is **`GPU_EXPERTS`** (experts/layer placed on GPU; the rest
run on CPU):

- **Higher** → fewer CPU experts → faster decode, **but** more VRAM. `104` uses
  ~88 GB/card on 96 GB H100 NVL. `112` OOMs here.
- **Lower** → less VRAM, but more RAM and slower (the CPU path is the long pole).

If you OOM on GPU, lower `GPU_EXPERTS` and/or `MAX_TOTAL_TOKENS`. If you OOM on CPU
RAM, raise `GPU_EXPERTS`.

Benchmark a config:

```bash
bash bench/decode_bench.sh 5 256     # 5 runs, 256 tokens each; reports median tok/s
```

---

## How it works (one paragraph)

GLM‑5.2 has 256 routed experts per layer; we keep `GPU_EXPERTS` of them on the GPUs
(INT4, via SGLang's W4AFP8 cutlass kernel) and the rest in CPU DRAM (INT4, via a
custom packed AVX‑512‑VNNI kernel in kt‑kernel). Each decode step runs the CPU and
GPU expert halves concurrently and merges them, so per‑layer latency is
`max(CPU, GPU+attention)`. CUDA graphs eliminate per‑step launch overhead. Keeping
the CPU weights **packed at 4 bits** (instead of pre‑expanding to int8) halves the
memory traffic in the bandwidth‑bound CPU window — that took the plain baseline from
~10.9 to ~14.7 tok/s. Two more layers then stack on top to reach the headline ~40.5
tok/s: **top‑2 expert substitution** keeps each token's 2 most‑important experts on the
CPU and reroutes the low‑weight tail to GPU‑resident experts (fewer CPU experts ⇒
shorter critical path, ~1.5×), and **NEXTN/MTP speculative decode** (depth‑3, running on
the otherwise‑idle GPU while the CPU experts are the bottleneck) verifies ~2.7–3.9
tokens per step for another ~1.5×. Both run under CUDA graphs. Full details and the
dead‑ends in **[BLOG.md](BLOG.md)**, [BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md), and
[BLOG_MTP_CUDAGRAPH.md](BLOG_MTP_CUDAGRAPH.md).
