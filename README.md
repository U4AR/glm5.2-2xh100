# RunGLM — GLM-5.2 (754B) on your own GPUs, with a speed/intelligence dial

Run the full 754B-parameter GLM-5.2 on hardware that cannot hold it, by keeping
the hot experts on the GPU and computing the rest on the CPU. Then **choose, per
request, how much intelligence you want to trade for speed** — by changing one
word in the model name.

```text
GLM5.2-top8    full routing, reference quality       ~15 tok/s
GLM5.2-top4    substitute 4 of 8 experts             ~27 tok/s
GLM5.2-top2    substitute 6 of 8   (default)         ~34 tok/s
```

Same server, same weights, no reload, mixed freely within one batch.

---

## How it works, in three sentences

GLM-5.2 routes every token to 8 of its 256 experts. Most of those experts live in
CPU RAM, and the per-layer CPU↔GPU round trip is what makes decoding slow. We
**keep the genuine top-K experts** the router chose — they carry the routing
identity — and **substitute the low-weight tail** with the best experts already
resident on the GPU, so fewer experts need the CPU round trip.

Lower K means fewer CPU round trips, so faster decoding and a bit less fidelity.
That is the dial.

---

## Hardware

You do **not** need two H100s. The launcher profiles the machine and sizes itself.

| | minimum | notes |
|---|---|---|
| GPU | ~24 GB VRAM, 1 card | more VRAM → more resident experts → faster |
| Host RAM | ~96 GB | the expert store lives here; 250 GB+ is comfortable |
| Disk | ~380 GB | int4 weights |
| CPU | AVX-512 VNNI preferred | AVX2 works via `RUNGLM_ALLOW_AVX2=1`, slower |

Any GPU count works — the profiler reads `CUDA_VISIBLE_DEVICES`, clamps tensor
parallelism to a valid power of two (an odd card count would otherwise abort at
startup), and sizes the resident expert set from the VRAM it actually finds.

Measured references: **2×H100 NVL ~34 tok/s**, **1×H100 ~24 tok/s** (91% of two
cards), **2×L40 on RunPod** validated (see [below](#running-on-a-rented-pod-runpod-and-similar)).

Two launchers, and picking the wrong one is the most common failure:

* `./run_fast.sh` — H100-tuned constants. Use on H100-class hardware.
* `./run_adaptive.sh` — profiles the machine first. **Use on anything else**,
  including L40/L40S/A100 and rented pods.

---

## Quick start

**1. Get the code and build it.** The build compiles a CPU kernel for *your*
machine, so it cannot be skipped.

```bash
git clone https://github.com/U4AR/glm5.2-2xh100.git RunGLM
cd RunGLM
INSTALL_SYSTEM_DEPS=1 ./setup.sh          # add RUNGLM_ALLOW_AVX2=1 on AVX2-only CPUs
```

**2. Download the weights** (~380 GB; runs in parallel with the build above if
you start it in another shell).

```bash
python int4_scripts/download_w4afp8.py
```

**3. Start the server.**

```bash
./run_fast.sh
```

It prints the profile it chose, then serves an OpenAI-compatible API on
**port 8000**. First boot takes 5–35 minutes depending on disk cache — most of it
is reading expert weights.

**4. Use it.** Any OpenAI client works; the tier is the model name.

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM5.2-top2",
  "messages": [{"role": "user", "content": "Write a haiku about mixture of experts."}]
}'
```

`chat_ui.py` gives you a browser UI with a tier dropdown:

```bash
python chat_ui.py     # http://localhost:7860
```

---

## Running on a rented pod (RunPod, and similar)

Validated on **RunPod 2× L40 with 500 GB RAM**. Two things differ from a bare-metal
box, and both will hard-fail if you skip them.

**Use persistent storage and point the weights at it.** A 500 GB volume is the
practical minimum for the ~380 GB checkpoint; 600 GB leaves room for the build.

```bash
export WEIGHTS_DIR=/workspace/weights      # survives pod restarts
```

**Opt into the AVX2 kernel if the CPU has no AVX-512.** RunPod's EPYC 7773X hosts
are AVX2+FMA only. Without this the build inherits the `avx512_packed` default and
**SIGILLs on the first expert** — an illegal-instruction crash, not a clean error.

```bash
export RUNGLM_ALLOW_AVX2=1
INSTALL_SYSTEM_DEPS=1 ./setup.sh
python int4_scripts/download_w4afp8.py
./run_adaptive.sh                          # NOT run_fast.sh — see below
```

**Use `run_adaptive.sh` on non-H100 cards.** `run_fast.sh` carries H100-tuned
constants; `run_adaptive.sh` profiles the machine and picks safe first-boot
settings — on 2×L40 that means `GPU_EXPERTS=24`, `MEM_FRACTION=0.85`,
`KT_RAWINT4_BACKEND=avx2_packed`, `KT_W4AFP8_GPU_BACKEND=marlin_sm80`, and Triton
rather than CUTLASS/FlashMLA for Ada SM89.

Check a pod before paying for it:

```bash
python scripts/hardware_profile.py --check --adaptive --weights-dir "$WEIGHTS_DIR/GLM-5.2-W4AFP8"
```

A failed preflight is deliberate — swap and disk are not counted as decode-time
RAM, because an expert served from swap does not decode, it stalls. Two hard
floors, independent of each other: roughly **53 GiB VRAM on a single card** (the
35 GiB dense trunk is not shardable when TP=1, so a single 48 GB card is *below*
the floor) or **~27 GiB per card on two**, and host RAM at **1.38 GiB per
non-resident expert-slot**.

Full detail, including the L40 pitfalls: [`PORTABLE_DEPLOYMENT.md`](PORTABLE_DEPLOYMENT.md).

---

## Choosing a tier

| model name | keeps | speed | use it for |
|---|---|---|---|
| `GLM5.2-top8` | all 8 | 1.0× | the reference answer; anything you will act on |
| `GLM5.2-top4` | 4 | ~1.8× | most work — no quality loss observed |
| `GLM5.2-top2` | 2 | ~2.3× | **default**; drafting, chat, long generations |
| `GLM5.2-top0` | 0 | ~3× | benchmarking only — **degenerates**, do not ship |

The honest caveat: `top4` and `top2` are backed by held-out benchmarking, `top0`
is not usable, and the quality gap between `top8` and `top2` has **not** been
measured on a hard task — the attempt is written up in
[`bench/deepswe/STATUS.md`](bench/deepswe/STATUS.md) and did not reach a verdict.
Treat the tiers as a latency dial with a real but unquantified fidelity cost.

---

## Common knobs

Everything is environment variables on `run_fast.sh`; the defaults are the tuned
configuration.

```bash
KEEP=4 ./run_fast.sh          # safer default tier
MTP=0 ./run_fast.sh           # disable speculative decode (~35% slower, less VRAM)
MODE=off ./run_fast.sh        # plain routing, no substitution — the reference
GPU_EXPERTS=64 ./run_fast.sh  # fewer resident experts if VRAM is tight
CUDA_VISIBLE_DEVICES=0 ./run_fast.sh   # force single-GPU
```

**If you are running agents or long chats, set `SLEEP_ON_IDLE=0`.** The default
releases KV memory between requests, which flushes the prefix cache every turn —
an ~11× slowdown that no health or memory metric will show you.

---

## Checking that it actually works

Three checks, in order. Each of these failed *silently* at some point during
development, so none of them is paranoid.

```bash
# 1. no boot-time scheduler exception — a half-crashed server still answers /health
grep -cE 'Scheduler hit an exception|OutOfMemoryError' server.log     # want 0

# 2. the tiers differ in speed — if they don't, the substitution patch didn't land
time curl -s localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top8", ...}'
time curl -s localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top2", ...}'

# 3. for agents/chat: the prefix cache is alive
grep -oE '#cached-token: [0-9]+' server.log | tail    # want non-zero and rising
```

⚠️ The tier is a no-op unless `/tmp/kt_topk_mode` exists. `run_fast.sh` writes it;
if you launch sglang by hand, you must too.

---

## Where things are

| path | what |
|---|---|
| [`REBUILD.md`](REBUILD.md) | how a rebuild works, what is pinned, what must be redone by hand |
| [`SETUP_RUNBOOK.md`](SETUP_RUNBOOK.md) | rebuilding from a wiped machine |
| [`CHANGES.md`](CHANGES.md) | every upstream file we modified, and why |
| [`STATUS_OF_FINDINGS.md`](STATUS_OF_FINDINGS.md) | which claims are solid, provisional, or **refuted** — wins any disagreement |
| [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) | the full lab notebook: every experiment, including the dead ends |
| [`backup/`](backup/) | patched files that live outside git (venv overlays, fork patches) |
| `bench/` | benchmark harnesses (LiveBench, DeepSWE, HLE, SWE-bench Pro) |

Most of the speed is **patches to sglang and ktransformers**, not new code here.
`setup.sh` reapplies them after the build; `CHANGES.md` lists them.

## Branches

`main` is the working implementation described above. Every other branch is an
experiment — three-tier SSD/RAM expert stores, energy-driven placement, expert
streaming over PCIe, prefetch prediction. Several were measured and **refuted**;
read `STATUS_OF_FINDINGS.md` before reviving one.

## License and provenance

Built on [ktransformers](https://github.com/kvcache-ai/ktransformers) and
[sglang](https://github.com/sgl-project/sglang); GLM-5.2 weights are Zhipu AI's.
This repo is the integration, the substitution mechanism, and the measurements.
