# Setup Runbook — GLM-5.2 fast (top-2) on 2× H100 NVL

Reproduce the working **top-2 expert-substitution** server from a clean/wiped box.
Tested 2026-06-30: restore → re-download → launch → **21.8 tok/s** decode, coherent.

This box is **shared** and `/cache` is **ephemeral** (wiped between sessions). The git
repo + tracked venv patches survive in `/data/models/RunGLM`; the 373 GB weights do not.

---

## TL;DR

```bash
cd /data/models/RunGLM
git fetch fast && git reset --hard fast/main          # 1. restore shipped state (41cb0a2)
rm -f /tmp/kt_topk_mode /tmp/kt_skip_cpu              #    clear stale sentinels
HF_HUB_ENABLE_HF_TRANSFER=1 python int4_scripts/download_w4afp8.py   # 2. weights (~373GB, ~3min)
./run_fast.sh                                         # 3. launch top-2 (KEEP=2)
# wait ~4-5 min for boot, then verify:
python3 bench/perf_probe/decbench.py 200 5            # ~21-22 tok/s
```

---

## 0. What this is

GLM-5.2 (754B MoE) served on 2× H100 NVL via **INT4 + ktransformers heterogeneous MoE**.
Each token routes to 8 of 256 experts; most live in CPU RAM and the per-layer CPU↔GPU
sync is the decode bottleneck. The **top-K substitution** trick keeps the genuine top-K
experts (routing "identity") and replaces the low-weight tail with the best GPU-resident
experts → fewer CPU experts per layer → faster decode. Pure logical reroute in
`sglang .../models/deepseek_v2.py`, gated by a sentinel file — **no rebuild needed**.

Measured (TP2, GPU_EXPERTS=104, decode tok/s, 5-run median):

| mode | tok/s | speedup | quality |
|---|---|---|---|
| baseline (`MODE=off`) | 14.7 | 1.00× | reference |
| `KEEP=4` | 18.5 | 1.25× | clean, no loss observed |
| `KEEP=2` (default) | **21.8** | **1.48×** | clean; rare repetition loops |
| `KEEP=0` + CPU-skip | 29.2 | 1.98× | degenerates — NOT recommended |

---

## 1. Restore the repo to the shipped state

The runnable top-2 state is **`fast/main` = commit `41cb0a2`** ("Top-2 expert
substitution + run_fast.sh"). The tracked venv patches (kept via `.gitignore`
negation) ride along with the reset:
`deepseek_v2.py`, `kt_ep_wrapper.py`, `w4afp8.py`, `nsa_backend.py`, and the
compiled `kt_kernel_ext...so`.

```bash
cd /data/models/RunGLM
git fetch fast
git reset --hard fast/main      # discards any WIP commits/working changes
rm -f /tmp/kt_topk_mode /tmp/kt_skip_cpu        # clear stale sentinels
```

> **Before resetting away local WIP commits**, confirm they're backed up. The last
> streaming experiment is saved on branch **`failed-be/top2-cudagraph-20260630`** on
> remotes `fast` (`U4AR/glm52-fast-2xh100`) and the `U4AR/ktransformers` fork.
> Push new WIP to a `failed-be/<topic>-<date>` branch before resetting.

Remotes: `fast` → `U4AR/glm52-fast-2xh100` (the fast/top-2 repo, **use this**);
`origin` → `U4AR/glm5.2-2xh100` (older general repo).

Sanity check after reset:
```bash
git log --oneline -1            # -> 41cb0a2 Top-2 expert substitution ...
grep -n kt_topk_mode .venv/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py
#  -> _KT_ROUTE_MASS_FLAG = os.environ.get("KT_TOPK_MODE_FILE", "/tmp/kt_topk_mode")
```

## 2. Re-download the weights (when `/cache` was wiped)

~373 GB / 46 files from `PhalaCloud/GLM-5.2-W4AFP8` to `/cache/nvme0` (resumable —
re-running skips complete files). nvme0 has ~3.2 TB free; takes ~2.5–3 min here.

```bash
cd /data/models/RunGLM
HF_HUB_ENABLE_HF_TRANSFER=1 python int4_scripts/download_w4afp8.py
# verify:
ls /cache/nvme0/models/GLM-5.2-W4AFP8/*.safetensors | wc -l   # -> 41
ls /cache/nvme0/models/GLM-5.2-W4AFP8 | grep -E 'config.json|tokenizer.json|index'
```
`HF_TOKEN` / `HF_HOME` are already set in the environment.

## 3. Launch the top-2 server

```bash
cd /data/models/RunGLM
./run_fast.sh                  # KEEP=2 (top-2) default -> writes /tmp/kt_topk_mode=sub2
```
Other operating points (no rebuild):
```bash
KEEP=4 ./run_fast.sh           # safer quality, ~18.5 tok/s
KEEP=0 ./run_fast.sh           # max speed + CPU-skip, ~29 tok/s, degrades — avoid
MODE=off ./run_fast.sh         # plain baseline, ~14.7 tok/s
```
The config: `MODEL`/`KT_WEIGHT_PATH=/cache/nvme0/models/GLM-5.2-W4AFP8`,
`KT_METHOD=RAWINT4`, `KT_RAWINT4_BACKEND=avx512_packed`, `GPU_EXPERTS=104`,
`MEM_FRACTION=0.94`, `MAX_TOTAL_TOKENS=4096`, TP=2, CUDA graph on, flashmla, dense MLA.

**Boot takes ~4–5 min** (loads 75 MoE layers' experts ~220ms each, then captures the
CUDA graph). Watch for readiness:
```bash
# server logs to wherever you redirected run_fast.sh; health endpoint:
until curl -s http://localhost:8000/health >/dev/null; do sleep 5; done; echo UP
```
Uses ~88 GB/card at GPU_EXPERTS=104 — **don't raise blindly (112 OOMs)**.

## 4. Verify

```bash
cat /tmp/kt_topk_mode                              # -> sub2
python3 bench/perf_probe/decbench.py 200 5         # -> ~21-22 tok/s median
```
Coherence spot-check (temp 0) — a haiku should be valid 5-7-5 with **no** repetition
loop; `lambda s: s[::-1]` for reverse-string:
```bash
curl -s http://localhost:8000/generate -H 'Content-Type: application/json' -d \
 '{"text":"[gMASK]<sop><|user|>\nWrite a haiku about the ocean.<|assistant|>\n","sampling_params":{"temperature":0,"max_new_tokens":120}}' \
 | python3 -c 'import sys,json;print(json.load(sys.stdin)["text"])'
```

## 5. Stop / free the GPUs (always, between runs)

```bash
LP=$(ss -ltnp | grep ':8000' | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$LP" ] && { kill -9 "$LP"; pkill -9 -P "$LP"; }
pkill -9 -f sglang.launch_server
fuser -k -9 8000/tcp
# wait for VRAM to drain before relaunching:
watch -n2 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

## Troubleshooting

- **`address already in use` on :8000** — a launcher is still holding the port. Kill by
  explicit PID as above; don't `pkill -f` your own shell command. Wait for VRAM < 2 GB.
- **Global perf collapse (decode 14→<1, prefill balloons)** — a runaway process is
  starving the CPU experts. Check `ps --sort=-pcpu -eo pid,pcpu,comm | head` for a
  runaway (a stray `ugrep -r /` once did this) before trusting any "it got slow" verdict.
- **Gibberish past ~2048 tokens** — already fixed (dense MLA, NSA off) in `run_server_int4.sh`.
- **Repetition loops at KEEP=2** — rare; bump to `KEEP=4` for a cleaner curve.

---
*Verified working 2026-06-30: restore + 373 GB re-download (~2.5 min) + top-2 launch →
21.78 tok/s median (min 21.5, max 22.0), coherent.*
