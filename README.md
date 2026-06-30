# RunGLM — GLM‑5.2 (754B) on 2×H100 at 14.6 tok/s

Serve **GLM‑5.2**, a 754B‑parameter MoE model, on a single dual‑H100 box using
**SGLang + KTransformers (kt‑kernel)** heterogeneous CPU+GPU Mixture‑of‑Experts.
A custom packed‑INT4 AVX‑512 CPU kernel + INT4 GPU experts, with dense MLA attention,
reach **~14.6 tok/s** single‑stream decode (coherent out to 12k+ tokens) while
fitting the CPU‑side experts in **~250 GB of RAM**.

The server is **OpenAI‑compatible** (`/v1/chat/completions`, streaming, reasoning
separation) and ships with a zero‑dependency browser chat UI.

> The full engineering story — what was tried, what worked, and why — is in
> **[BLOG.md](BLOG.md)**.

---

## Results

Measured on a 2×H100‑NVL / AMD EPYC (Zen4) box; expect similar on comparable hardware
(see [Hardware requirements](#hardware-requirements)):

| Precision | CPU experts | GPU experts/layer | RAM needed | Decode |
|---|---|---:|---:|---:|
| **INT4 (recommended)** | packed RAWINT4 (AVX‑512 VNNI) | 96–104 | **~250 GB** | **~14.6 tok/s** (dense MLA) |
| FP8 ("8‑bit") | block‑FP8 | 48 | ~629 GB (or NVMe swap) | ~8.7 tok/s |

Both serve the same model quality; the INT4 path is faster **and** needs roughly
half the RAM, because the CPU‑expert path is memory‑bandwidth bound and INT4 moves
half the bytes (see BLOG.md).

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
| `run_fast.sh` | **High‑speed top‑K expert substitution** (~22 tok/s, top‑2 default) |
| `run_server_int4.sh` | Launch the **INT4** server (the ~14.6 tok/s path) |
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
etc. from its own location.

**This does not use stock `sglang` from PyPI.** It uses **KTransformers** — the
`kvcache-ai/ktransformers` project, which bundles its *own* SGLang fork (the
`kvcache-ai/sglang` submodule, installed as the package **`sglang-kt`**) plus the
`kt-kernel` heterogeneous CPU+GPU MoE engine. On top of that, this repo carries a
handful of source patches (the INT4/NSA/MTP fixes listed above). So setup is: install
KTransformers into a venv, then overlay this repo's patches.

```bash
# 1. Clone this repo (anywhere). Its tracked patches come down with it.
git clone <this-repo-url> RunGLM
cd RunGLM
export REPO=$(pwd)            # used in the examples below

# 2. Get the KTransformers fork (the SGLang-kt + kt-kernel sources). It is NOT
#    committed here (gitignored) — clone it into ./ktransformers WITH submodules
#    (the bundled SGLang fork is a git submodule).
git clone --recursive https://github.com/kvcache-ai/ktransformers.git ktransformers
#    (this repo was validated against the U4AR/ktransformers fork; use that remote
#     if you need the exact streaming/MTP work-in-progress branches.)

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

# 6. Overlay THIS repo's patches on top of the freshly installed sglang-kt /
#    kt-kernel (they live in-tree under .venv/... and were just overwritten).
git checkout -- .venv

# 7. Sanity check.
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
Point `WEIGHTS` at fast scratch (NVMe) with ~380 GB free:

```bash
export WEIGHTS=/path/to/nvme/GLM-5.2-W4AFP8     # default: ./weights/GLM-5.2-W4AFP8
HF_HUB_ENABLE_HF_TRANSFER=1 python int4_scripts/download_w4afp8.py
```

The FP8 path instead uses the original GLM‑5.2 FP8 checkpoint; download it to a dir of
your choice and set `FP8_WEIGHTS` to it (it's also a source of `chat_template.jinja`,
already vendored in this repo).

### 2a. Run the INT4 server (recommended — ~14.6 tok/s, ~250 GB RAM)

```bash
MODEL=$WEIGHTS \
KT_METHOD=RAWINT4 \
KT_WEIGHT_PATH=$WEIGHTS \
KT_RAWINT4_BACKEND=avx512_packed \
GPU_EXPERTS=104 \
MAX_TOTAL_TOKENS=4096 \
MEM_FRACTION=0.94 \
CPUINFER=72 \
bash run_server_int4.sh
```

Boot takes ~2–3 min (loads INT4 experts from NVMe + CUDA‑graph capture). Wait for
`The server is fired up and ready to roll!`, then the OpenAI API is live on `:8000`.

### 2a‑fast. High‑speed mode — top‑2 expert substitution (~22 tok/s, default)

For **~1.5× faster decode** (22 tok/s vs 14.7) with one command, use `run_fast.sh`.
It keeps each token's genuinely most‑important experts and **substitutes the
low‑weight tail with the best GPU‑resident experts**, so fewer experts hit the slow
CPU path. The default keeps the **top‑2**:

```bash
./run_fast.sh            # KEEP=2 (top‑2)  → ~22 tok/s, ~1.5×   [default]
KEEP=4 ./run_fast.sh     # safer quality   → ~18.5 tok/s, ~1.25× (no degeneration found)
KEEP=0 ./run_fast.sh     # max speed       → ~29 tok/s, ~2×   (quality drift — not recommended)
MODE=off ./run_fast.sh   # plain baseline  → 14.7 tok/s
```

It wraps `run_server_int4.sh` with the winning INT4 recipe and writes the reroute
sentinel `/tmp/kt_topk_mode`. **Quality note:** `KEEP=2` is coherent on most tasks
but can occasionally fall into a repetition loop on open‑ended generation; use
`KEEP=4` if you need baseline‑equivalent quality. The full study, including how the
degeneration was caught, is in **[BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md)**.

| `KEEP` | decode | speedup | quality |
|---:|---:|---:|---|
| 4 | ~18.5 tok/s | 1.25× | clean |
| **2** (default) | **~22 tok/s** | **1.49×** | mostly clean, rare loops |
| 0 | ~29 tok/s | 1.98× | degenerates — avoid |

### 2b. Run the FP8 server (8‑bit alternative — ~8.7 tok/s, needs ~629 GB / swap)

```bash
MODEL=$FP8_WEIGHTS KT_WEIGHT_PATH=$FP8_WEIGHTS GPU_EXPERTS=48 bash run_server.sh
```

### 2c. Long‑context mode (for coding agents / long chats)

The quickstart above uses a tiny 4096‑token window. For a coding‑agent backend or
long conversations, run with a real context window:

```bash
MODEL=$WEIGHTS \
KT_METHOD=RAWINT4 \
KT_WEIGHT_PATH=$WEIGHTS \
KT_RAWINT4_BACKEND=avx512_packed \
GPU_EXPERTS=96 \
CONTEXT_LENGTH=131072 \
MAX_TOTAL_TOKENS=131072 \
MEM_FRACTION=0.93 \
CPUINFER=72 \
MAX_RUNNING=2 \
bash run_server_int4.sh
```

Verified: coherent to 12k+ tokens, decode steady at **~14.6 tok/s** even at 12k+
positions, ~88 GB/card, `available_gpu_mem≈8 GB` at 128k. Decode speed is unchanged
vs the short‑context config (we are CPU‑MoE bound, so attention length is free).

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
memory traffic in the bandwidth‑bound CPU window — that is the change that took decode
from ~10.9 to ~14.6 tok/s. Full details and the dead‑ends in **[BLOG.md](BLOG.md)**.
