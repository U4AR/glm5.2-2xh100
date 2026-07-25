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

# Install host build dependencies if the image does not already have them.
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

## CPU backend

The fast packed-INT4 backend needs AVX-512 VNNI. Check the allocated host:

```bash
lscpu | grep -iE 'model name|avx512|vnni'
```

The KTransformers AVX2 fallback can broaden compatibility later, but it is not
expected to preserve the H100/Zen4 throughput. The preflight therefore reports
missing AVX-512 VNNI as an error for the performance-oriented path.

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
