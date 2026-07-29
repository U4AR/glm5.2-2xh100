# Portable deployment

The supported portable target is **one or two NVIDIA GPUs on one host**,
sufficient system RAM, Linux, Python 3.12, and a CUDA 12.8+ development image.
The first automatic profile is `2xl40`; the measured reference remains `2xh100`;
single-GPU is measured below.

## What the machine actually has to have

Two hard floors, and they are independent — clearing one does not help with the
other. `scripts/hardware_profile.py --check` enforces both before launch.

**Per-card VRAM.** The dense trunk is about 35 GiB and is not optional; TP
shards it across cards, so on one GPU all 35 GiB lands on that card. Adding KV,
the CUDA-graph pool, the MTP draft model, prefill scratch and the smallest
resident expert set that has ever booted (4/layer) gives a floor of roughly
**53 GiB on one GPU** or **27 GiB per card on two**. A single 48 GB card is
therefore *below* the single-GPU floor — it clears the trunk but has nothing
left for the runtime. Two 48 GB cards are fine, because the trunk halves.

**Host RAM.** Every expert that is not GPU-resident is served from host RAM, at
**1.38 GiB per expert-slot** across the model's 75 MoE layers. Static placement
needs `(256 − GPU_EXPERTS) × 1.38 GiB` plus roughly 40 GiB of non-expert
overhead — measured 352 GB at `GPU_EXPERTS=24`. The three-tier build
(`experiments/expert_tiering_ssd`) caps the RAM tier and pushes the remainder to
SSD, which is what lowers this: measured **107 GB** at `KT_RAM_EXPERTS=48` and
**86 GB** at 32, both with movement enabled. Below that the fixed ~40-45 GiB
overhead dominates and there is no configuration left to trade away.

So the smallest host that runs this model at all is around **96 GB of RAM**, and
that is with the tiered build and its accuracy cost. Storage and swap do not
substitute: the preflight deliberately refuses to count them, because an expert
fetched from disk on the decode critical path is not a slower expert, it is a
stall long enough to dominate the step.

## Clone to serving

Identical on every supported host. `setup.sh` starts the 373 GB checkpoint
download in the background and waits for it after the build, so the two overlap
instead of running end to end — the download is network/disk bound and the
kt-kernel build is CPU bound, so serialising them idles whichever resource is
not in use.

```bash
git clone <this repo> RunGLM && cd RunGLM
INSTALL_SYSTEM_DEPS=1 ./setup.sh     # build + weights, concurrently
./run_adaptive.sh                    # safe2 + MTP depth 3 + adaptive cache
```

Progress for the background fetch goes to `logs/download_weights.log`; the build
prints to the terminal as usual. Useful knobs:

```bash
DOWNLOAD_WEIGHTS=0 ./setup.sh        # checkpoint already on a mounted volume
HF_MAX_WORKERS=16 ./setup.sh         # fast local disk (default 1 is NFS-safe)
./scripts/download_weights.sh        # fetch on its own; resumes; attaches to
                                     # an in-flight download instead of racing it
```

`scripts/download_weights.sh` uses its own `.dl-venv`, not the main `$VENV`,
because setup pip-installs into `$VENV` while the download runs and a downloader
importing `huggingface_hub` out of a directory being rewritten underneath it is
a race. Completion is checked against the checkpoint's own shard index rather
than a marker file, so a half-finished download from a killed run is correctly
seen as incomplete and resumed.

## Single GPU

Supported and measured. TP drops to 1, which puts the whole 35 GiB dense trunk
on the one card, so this needs an **80 GB-class card** — see the VRAM floor
above. `run_fast.sh` and `run_adaptive.sh` size themselves from what they find;
there is nothing to configure.

```bash
CUDA_VISIBLE_DEVICES=0 ./run_fast.sh    # auto: TP1, safe2, MTP depth 3
```

### Measured single-GPU result

1× H100 NVL (95 GiB, SM 9.0), `GPU_EXPERTS=24`, TP1, `safe2` routing, MTP
depth 3, greedy, 8 runs of 300 tokens:

- decode **23.74 tok/s** median (min 23.57, max 23.77 — spread 0.8%);
- MTP accept length 2.435;
- 63.3 GiB VRAM on the single card; 352 GB host RSS;
- boot to serving 2 min 45 s;
- 66-item QA eval: **66/66 correct (±0.028), 0% repetition loops**, verdict
  agreement 1.00 on all 66 items against the two-GPU full-coverage reference.

Quality is not a tradeoff here, and it should not be: `safe2` keeps the genuine
top-2 and sends a non-resident member down the CPU path rather than substituting
it, so a smaller resident set costs time, never correctness. Token-level
divergence from the reference is nonetheless large (prefix agreement 0.08),
which is expected and not a defect — TP1 reduces in a different order than TP2,
so greedy decoding parts company after a few tokens. Judge these configurations
on task accuracy, not on matching token streams.

The interesting part is the ratio: the same box at TP2 with `GPU_EXPERTS=104`
and the same routing contract runs ~30.5 tok/s, so **one card retains about 78%
of the two-card throughput while holding a quarter of the resident experts**.
That is not a small-model effect. Decode here is bound by the CPU expert path,
and TP2's second rank spends most of its time in NCCL all-reduce spin-wait
rather than on work; removing it costs less than the residency loss suggests.
Halving the cards does not halve the speed, but it does roughly double host RAM
demand, because every expert that is not resident moves to the CPU store.

## RunPod: 2× L40, 500 GB RAM

Use a RunPod PyTorch/CUDA development template with persistent storage mounted at
`/workspace`. A 500 GB volume is the practical minimum for the approximately
373 GB W4AFP8 checkpoint; 600 GB leaves healthier build/cache headroom.

The EPYC 7773X host is AVX2+FMA with no AVX-512, so the AVX2 CPU kernel must be
opted into explicitly — without it the build inherits `run_fast.sh`'s
`avx512_packed` default and SIGILLs at the first expert:

```bash
export RUNGLM_ALLOW_AVX2=1
INSTALL_SYSTEM_DEPS=1 ./setup.sh
./run_adaptive.sh
```

## How setup behaves

`setup.sh` builds kt-kernel on the final pod. No compiled kt-kernel extension
is distributed by the current branch: setup removes any stale local copy,
disables pip wheel-cache
reuse, and rebuilds against the final machine's Python, PyTorch, CUDA and CPU.
On AVX2-only hosts it explicitly targets the AVX2+FMA baseline rather than
`-march=native`. It then restores only the portable Python overlays and runs
`kt doctor`.

`run_adaptive.sh` detects the hardware, runs preflight checks, then applies safe
first-boot settings. For two 48 GiB L40/L40S-class cards:

```text
TP_SIZE=2
GPU_EXPERTS=24
MEM_FRACTION=0.85
MAX_TOTAL_TOKENS=8192
CPUINFER=min(available CPUs minus system reserve, 72)
NUMA_NODES=detected from the host
KT_GPU_PREFILL_THRESHOLD=0
```

The validated `2xl40` profile overrides that generic worker formula with
`CPUINFER=28`, `KT_THREADPOOL_COUNT=2`,
`KT_RAWINT4_BACKEND=avx2_packed`, and
`KT_W4AFP8_GPU_BACKEND=marlin_sm80`. Ada SM89 uses Triton for dense FP8 and
attention; Hopper keeps CUTLASS W4A8 and FlashMLA.

All values remain environment-overridable:

```bash
GPU_EXPERTS=32 MAX_TOTAL_TOKENS=4096 ./run_adaptive.sh
```

## Preflight only

```bash
python scripts/hardware_profile.py --check --adaptive \
  --weights-dir ./weights/GLM-5.2-W4AFP8
```

Adaptive mode conservatively requires 430 GiB system RAM because every expert
must remain CPU-backed for coherent eviction. Static mode computes a lower RAM
floor from `GPU_EXPERTS`. A failed preflight is intentional: storage and swap
are not treated as decode-time RAM.

## Portable CPU and GPU backends

Check the allocated host:

```bash
lscpu | grep -iE 'model name|avx512|vnni'
```

There are two independent dispatch decisions:

- AVX-512 VNNI hosts retain `AVX512RawInt4Packed_MOE` and
  `KT_RAWINT4_BACKEND=avx512_packed`.
- AVX2+FMA hosts can explicitly select the separate
  `AVX2RawInt4Packed_MOE`. Both `avx2` and `avx2_packed` select it. It uses
  signed two's-complement INT4 and TP-sliced copied buffers, including the
  down-projection gather and two NUMA pools.
- Hopper SM90+ retains the existing CUTLASS W4A8 GPU path.
- Ampere/Ada SM80-SM89 selects the separate Marlin W4A16 packed-INT4 path
  with `KT_W4AFP8_GPU_BACKEND=marlin_sm80`. Its activations are BF16; weights
  remain packed INT4 in VRAM.

The AVX2 path remains an explicit experimental opt-in:

```bash
export RUNGLM_ALLOW_AVX2=1
INSTALL_SYSTEM_DEPS=1 ./setup.sh
HF_MAX_WORKERS=16 python int4_scripts/download_w4afp8.py
./run_adaptive.sh
```

The hardware profile then selects `KT_RAWINT4_BACKEND=avx2_packed` and
`KT_KERNEL_CPU_VARIANT=avx2`. This is an experimental compatibility path, not a
near-30 tok/s promise.

### Measured L40 result

Validated on 2× NVIDIA L40 (SM89, 46 GiB each), AMD EPYC 7773X, 28 CPU
workers, two NUMA/thread pools, driver 550.127.08, and a CUDA 12.8 toolkit:

- scalar/SIMD and synthetic RAWINT4 tests: 6/6 passed;
- real W4AFP8 CPU cosine: 0.99992, finite output;
- real W4AFP8 GPU cosine: 0.99999, finite output;
- TP2 server and all CUDA graph captures completed;
- repeated smoke output was coherent after adaptive-cache warm-up;
- first five decode runs: 13.13, 12.85, 10.48, 12.48, 11.75 tok/s
  (median 12.48, range 10.48–13.13);
- warmed five runs: 11.92, 13.35, 13.14, 13.95, 13.48 tok/s
  (median 13.35, range 11.92–13.95).

The measured peak allocation was 40,243 MiB on GPU 0 and 40,109 MiB on GPU 1.
Host use reached approximately 464 GiB, including about 374 GiB RSS in the
primary scheduler process. Adaptive expert swaps occurred during warm-up.

The successful run did not emit a PTX-version or newer-driver error. Earlier
attempts to use Hopper-only components failed separately: the W4A8 CUTLASS
kernel reported TMA descriptor error 801, and FlashMLA reported no kernel image
for SM89. Those are GPU architecture incompatibilities, not AVX2 CPU-kernel
failures; the portable profile avoids both.

Exact validated launch:

```bash
RUNGLM_ALLOW_AVX2=1 \
KT_KERNEL_CPU_VARIANT=avx2 \
KT_RAWINT4_BACKEND=avx2_packed \
KT_W4AFP8_GPU_BACKEND=marlin_sm80 \
FP8_GEMM_BACKEND=triton \
ATTENTION_BACKEND=triton \
CPUINFER=28 \
KT_THREADPOOL_COUNT=2 \
NUMA_NODES="0 1" \
GPU_EXPERTS=24 \
./run_adaptive.sh
```

## Profile policy

Automatic profiles select safe capacity settings, not promised performance.
The repository keeps separate concerns:

- Hardware detection chooses initial VRAM, CPU-worker and context limits.
- The launcher accepts explicit overrides for measured tuning.
- Preflight rejects configurations that cannot hold the required expert set.
- `config.sh` owns all paths.
- Machine-specific extensions are rebuilt locally.

To add another two-GPU family, add a conservative branch to
`scripts/hardware_profile.py`, validate boot/coherence, measure VRAM, then record
the measured preset. Do not infer throughput from VRAM alone.
