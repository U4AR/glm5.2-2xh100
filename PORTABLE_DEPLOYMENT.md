# Portable deployment

The supported portable target is currently **two NVIDIA GPUs on one host**,
sufficient system RAM, Linux, Python 3.12, and a CUDA 12.8+ development image.
The first automatic profile is `2xl40`; the measured reference remains `2xh100`.

## RunPod: 2× L40, 500 GB RAM

Use a RunPod PyTorch/CUDA development template with persistent storage mounted at
`/workspace`. A 500 GB volume is the practical minimum for the approximately
373 GB W4AFP8 checkpoint; 600 GB leaves healthier build/cache headroom.

```bash
cd /workspace
git clone --branch experiment/adaptive-decode-cache --single-branch \
  https://github.com/U4AR/glm5.2-2xh100.git RunGLM
cd RunGLM

# The EPYC 7773X host is AVX2+FMA but has no AVX-512.
export RUNGLM_ALLOW_AVX2=1

# Install host build dependencies and compile an AVX2-local kt-kernel.
INSTALL_SYSTEM_DEPS=1 ./setup.sh

# Resumable and network-filesystem-safe by default.
python int4_scripts/download_w4afp8.py

# Recommended: safe2 routing, MTP depth 3, adaptive decode expert cache.
./run_adaptive.sh
```

`setup.sh` builds kt-kernel on the final pod. It never restores the checked-in
Zen4/Hopper extension. It installs the pinned KTransformers fork, restores only
the portable Python overlays, and runs `kt doctor`.

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
