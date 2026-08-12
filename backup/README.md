# backup/ — the work that lives OUTSIDE this git repo

Created 2026-08-12 before tearing the machine down. Everything here is work that
would otherwise be lost, because it lives in the Python venv or in a shallow
clone rather than in any repository.

## sglang-patched/

**20 files** edited in place inside
`.venv/lib/python3.12/site-packages/sglang/`, copied here with their original
relative paths. These are not a diff — they are the patched files themselves, so
they can be dropped straight back over a matching install.

They carry essentially every model-side result this project produced:

| file | what it carries |
|---|---|
| `srt/models/deepseek_v2.py` | the top-K expert substitution (`_kt_topk_experiment`) — GlmMoeDsa routes through here, NOT `glm4_moe.py` |
| `srt/model_executor/cuda_graph_runner.py` | the MTP-under-CUDA-graphs fix (kt verify-batch buffer registration) |
| `srt/layers/attention/nsa_backend.py` | NSA multistep draft metadata fix (long-context MTP corruption) |
| `srt/layers/attention/triton_backend.py` | draft-path buffer fixes |
| `srt/layers/quantization/w4afp8.py` | the `-1` topk remap for ep_size=1, and no-op weight_loader for GPU prefill |
| `srt/layers/moe/kt_ep_wrapper.py` | W4AFP8 detection, bf16 scale buffer, GPU-resident expert overlay |
| `srt/entrypoints/openai/serving_chat.py` | per-request intelligence tier (`<model>-topN`) |
| `srt/managers/*`, `srt/model_executor/*` | keep_k plumbing through ModelWorkerBatch → ForwardBatch |

Restore: install the matching sglang build, then copy these over it. Verify with
a tier A/B (`GLM5.2-top8` vs `GLM5.2-top2`) — if the tiers give identical speed,
the substitution patch did not land.

⚠️ `/tmp/kt_topk_mode` must exist or `-topN` is silently a no-op.

## ktransformers-patches/

The **6 local commits** on the ktransformers fork at `/data/models/RunGLM/ktransformers`,
exported with `git format-patch`. The fork is a **shallow clone**, so it cannot be
pushed to a fresh remote (`did not receive expected object`) — patches are the
portable form.

Base commit: `6c9c95601d97` on `kvcache-ai/ktransformers`.

    git clone https://github.com/kvcache-ai/ktransformers
    cd ktransformers && git checkout 6c9c95601d97
    git am /path/to/ktransformers-patches/*.patch

Contents: the packed-RAWINT4 W4A8 CPU MoE kernel (13.4 tok/s, +23% over
baseline), `write_weights_to_buffer` for W4AFP8 GPU bulk prefill, and the
three-tier expert store with per-expert promote/evict.

Build with `install.sh build` + `CPUINFER_USE_CUDA=1` only — a manual cmake
produces a CPU-only `.so` that silently drops `submit_with_cuda_stream`.

## VERSIONS.txt

Exact package versions these patches were built against. `flashinfer 0.6.3` in
particular is pinned by the build; `apache-tvm-ffi==0.1.11` is required for
tilelang (0.1.12 is broken).

## What is NOT here

* Model weights (373 GB) — `/data/models/glm52-w4afp8`, re-downloadable via
  `int4_scripts/download_w4afp8.py`.
* The two upstream benchmark harnesses (`deep-swe`, `swebenchpro_upstream`) —
  see `.gitignore` for their clone commands.
* Docker images — re-pullable from `public.ecr.aws/d3j8x8q7/swe-bench-202605`.
* Host configuration (rootless Docker host-loopback, data-root, squid port
  patch) — documented in `bench/deepswe/STATUS.md`; all must be redone.
