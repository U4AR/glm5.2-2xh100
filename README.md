# RunGLM — GLM‑5.2 (754B) on 2×H100 at 13.4 tok/s

Serve **GLM‑5.2**, a 754B‑parameter MoE model, on a single dual‑H100 box using
**SGLang + KTransformers (kt‑kernel)** heterogeneous CPU+GPU Mixture‑of‑Experts.
A custom packed‑INT4 AVX‑512 CPU kernel + INT4 GPU experts reach **13.4–13.9 tok/s**
single‑stream decode while fitting the CPU‑side experts in **~250 GB of RAM**.

The server is **OpenAI‑compatible** (`/v1/chat/completions`, streaming, reasoning
separation) and ships with a zero‑dependency browser chat UI.

> The full engineering story — what was tried, what worked, and why — is in
> **[BLOG.md](BLOG.md)**.

---

## Results (this box)

| Precision | CPU experts | GPU experts/layer | RAM needed | Decode |
|---|---|---:|---:|---:|
| **INT4 (recommended)** | packed RAWINT4 (AVX‑512 VNNI) | 104 | **~250 GB** | **13.4–13.9 tok/s** |
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
- **RAM:** **~250 GB** for the INT4 config at `GPU_EXPERTS=104`. The FP8 config needs
  ~629 GB or NVMe swap.
- **Disk:** ~380 GB of fast scratch for the INT4 expert weights.

---

## What's in this repo

| File | Purpose |
|---|---|
| `run_server_int4.sh` | Launch the **INT4** server (the 13.4 tok/s path) |
| `run_server.sh` | Launch the **FP8** server (the 8‑bit path) |
| `chat_ui.py` / `start_ui.sh` | Zero‑dep browser chat UI → OpenAI endpoint |
| `chat_template.jinja` | GLM‑5.2 chat template (wired into both launchers) |
| `int4_scripts/` | Weight download / GPTQ repack / offline kernel tests |
| `bench/decode_bench.sh` | Decode throughput benchmark |
| `BLOG.md` | The full write‑up of the optimization journey |
| `*_HANDOFF.md`, `PERF_CUDA_GRAPHS.md` | Deep‑dive engineering notes |

The custom kt‑kernel build (with the packed‑INT4 CPU kernel) and three SGLang
patches are checked in under `.venv/` so the validated environment is reproducible:

- `kt_kernel_ext*.so` — custom build, symbol `AVX512RawInt4Packed_MOE`.
- `sglang/.../quantization/w4afp8.py` — the `topk_ids == -1` remap fix for INT4 GPU
  experts under `ep_size=1` (without it the GPU MoE NaNs → garbage).
- `sglang/.../models/deepseek_v2.py`, `.../attention/nsa_backend.py` — supporting fixes.

---

## Quickstart

### 0. Activate the environment

```bash
cd /data/models/RunGLM
source .venv/bin/activate     # also exports LD_LIBRARY_PATH for kt-kernel (hwloc/numa)
```

> Always `source` the venv — kt‑kernel links hwloc/libnuma from the venv prefix.

### 1. Get the weights

INT4 path uses Phala's W4AFP8 export (4‑bit experts + FP8 non‑experts), ~373 GB:

```bash
python int4_scripts/download_w4afp8.py        # -> /cache/nvme0/models/GLM-5.2-W4AFP8
```

FP8 path uses the FP8 checkpoint at `/data/models/GLM-5.2-FP8` (also the source of
`chat_template.jinja`).

### 2a. Run the INT4 server (recommended — 13.4 tok/s, ~250 GB RAM)

```bash
MODEL=/cache/nvme0/models/GLM-5.2-W4AFP8 \
KT_METHOD=RAWINT4 \
KT_WEIGHT_PATH=/cache/nvme0/models/GLM-5.2-W4AFP8 \
KT_RAWINT4_BACKEND=avx512_packed \
GPU_EXPERTS=104 \
MAX_TOTAL_TOKENS=4096 \
MEM_FRACTION=0.94 \
CPUINFER=72 \
bash run_server_int4.sh
```

Boot takes ~2–3 min (loads INT4 experts from NVMe + CUDA‑graph capture). Wait for
`The server is fired up and ready to roll!`, then the OpenAI API is live on `:8000`.

### 2b. Run the FP8 server (8‑bit alternative — ~8.7 tok/s, needs ~629 GB / swap)

```bash
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
memory traffic in the bandwidth‑bound CPU window — that is the change that took decode
from ~10.9 to 13.4 tok/s. Full details and the dead‑ends in **[BLOG.md](BLOG.md)**.
