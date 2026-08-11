# CHANGES — exactly what code in this project was modified

This project runs **GLM-5.2 on 2×H100** by splitting the MoE experts between GPU
and CPU. Almost none of the speed comes from new standalone code; it comes from
**patches to two vendored upstream projects**. This file exists so that anyone
reading the archive can tell, without guessing, which lines are ours.

There are three categories, and only the first two are code we wrote:

1. **Patches to upstream SGLang** — vendored into `.venv/` and force-added to git.
2. **Patches to upstream ktransformers / kt-kernel** — a separate git repo, pinned below.
3. **Our own files** — launchers, benchmarks, UIs, experiment harnesses. All of
   `bench/`, `experiments/`, `int4_scripts/`, `scripts/`, `*_ui.py`, `run_*.sh`,
   `setup*.sh`, `config.sh` and the `*.md` write-ups are ours, written from scratch.

---

## 1. Patched SGLang files

Upstream baseline is the SGLang used by kt-kernel, pinned as a submodule of
ktransformers at **`third_party/sglang @ 51032b712`** ("feat: support end-to-end
KT LoRA serving for Qwen3.5 MoE (#53)"). It is installed into the venv as
`sglang_kt-0.0.0.dev0`.

The complete, machine-generated diff of our copies against that upstream is
checked in at **[`patches/sglang-vs-upstream.patch`](patches/sglang-vs-upstream.patch)**
(6,677 lines). Regenerate it any time with
[`scripts/gen_upstream_patch.sh`](scripts/gen_upstream_patch.sh).

Line counts vs pristine upstream:

| Vendored file (under `.venv/.../site-packages/`) | +added | −removed | What our patch does |
|---|---:|---:|---|
| `sglang/srt/layers/moe/kt_ep_wrapper.py` | 3491 | 28 | **The main one.** GPU/CPU expert split, W4AFP8 GPU bulk prefill, expert placement strategies (static / oracle / adaptive), the decode-time adaptive expert cache, expert prefetch + UVA gather, tier movement. Effectively rewritten. |
| `sglang/srt/models/deepseek_v2.py` | 1597 | 14 | GLM-5.2 (`GlmMoeDsa`) routes through here, not `glm4_moe.py`. Adds the top-K expert **substitution** kernel (`_kt_topk_experiment`), per-request keep-K tiering, and the in-graph next-layer expert **predictor** (`KT_PRED_*`). |
| `sglang/srt/layers/quantization/w4afp8.py` | 245 | 2 | Fixes W4AFP8 GPU MoE at `ep_size=1`: kt marks CPU experts with `-1` in `topk_ids`, and `cutlass_w4a8_moe` only remaps `-1` when EP>1 → NaN → garbage. Plus a no-op `weight_loader` and an idempotency guard on `process_weights_after_loading`. |
| `sglang/srt/layers/attention/nsa_backend.py` | 70 | 176 | NSA sparse attention for GLM-5.2, whose layers **share one indexer per 4 layers** (IndexShare). Also fixes multistep MTP drafting, which was handing every spec step step-0 metadata and corrupting long-context output. |
| `sglang/srt/entrypoints/openai/serving_chat.py` | 61 | 0 | Per-request intelligence tier: OpenAI `model` field `<base>-topN` selects the keep-K expert tier live, per request, mixed-batch. |
| `sglang/srt/configs/model_config.py` | 37 | 0 | GLM-5.2 config plumbing for the NSA index/share pattern. |
| `sglang/srt/entrypoints/http_server.py` | 36 | 0 | Endpoints for the tier + placement controls. |
| `sglang/srt/model_executor/cuda_graph_runner.py` | 29 | 1 | **MTP under CUDA graphs.** The kt verify-batch token count (`bs * num_draft_tokens`) was never registered via `set_capture_batch_sizes`, so its buffer was transient and went stale inside the captured host node → garbage. |
| `sglang/srt/model_executor/forward_batch_info.py` | 28 | 0 | Carries per-token `keep_k` down to the routing kernel. |
| `sglang/srt/mem_cache/memory_pool.py` | 24 | 8 | NSA index-k stub (saves ~4.1 GB/card on the layers that do not own an indexer). |
| `sglang/srt/model_executor/model_runner.py` | 24 | 0 | Hooks for the adaptive expert cache. |
| `sglang/srt/managers/schedule_batch.py` | 16 | 0 | `keep_k` on the request → `ModelWorkerBatch`. |
| `sglang/srt/models/deepseek_common/.../forward_mha.py` | 12 | 8 | NSA path for shared-indexer layers. |
| `sglang/srt/layers/attention/triton_backend.py` | 9 | 0 | Multi-step draft warmup for MTP under graphs. |
| `sglang/srt/server_args.py` | 5 | 2 | New flags (`--kt-expert-placement-strategy`, RAWINT4 backend selection). |
| `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 2 | 0 | NSA KV pool sizing. |
| `kt_kernel/utils/amx.py`, `kt_kernel/utils/loader.py` | 0 | 0 | Identical to our patched kt-kernel source — the changes live in the ktransformers repo (below), not here. |

`.gitignore` force-adds these paths; the header comments there are historical
notes and are **not** a complete list — this table is.

## 2. Patched ktransformers / kt-kernel

Separate repo at `ktransformers/` (gitignored by the parent). Backed up to
**`U4AR/ktransformers`, branch `glm52-work`, commit `49a9bd8`**, forked from
upstream `kvcache-ai/ktransformers @ 6c9c9560`.

Six commits on top of upstream, +4,565 / −49 lines:

| File | What our patch does |
|---|---|
| `kt-kernel/operators/avx2/rawint4_packed_avx512vnni-moe.hpp` | **New kernel.** Packed RAWINT4 W4A8 CPU MoE. Three things had to be right: Zen4 has `avx512_vnni` and **not** `avx_vnni`, so it uses the AVX-512 EVEX form of `_mm256_dpbusd_epi32`; int4 is **two's-complement**, not nibble-8; and `tp=2` needs a **copy** into the TP buffers (zero-copy collapses to tp=1 and segfaults under CUDA graphs). |
| `kt-kernel/operators/avx2/moe_base.hpp` | Per-expert promote/evict hooks; `write_weights_to_buffer` ported from `rawint4-moe.hpp` (it was a throw-stub, which blocked GPU bulk prefill). |
| `kt-kernel/cpu_backend/expert_store_shm.h` | *(new)* Shared-memory expert store so all 256 experts can be staged. The shipped dynamic-update path evicts into **missing** weights, which is silent garbage. |
| `kt-kernel/cpu_backend/cpu_expert_opts.h`, `cpu_expert_profile.h` | *(new)* Per-expert options and timing profile. |
| `kt-kernel/cpu_backend/task_queue.{h,cpp}`, `worker_pool.cpp`, `cpuinfer.h` | Queue-depth control on the CPU submit path. Head-of-line blocking is what made expert streaming cost +31.7% decode. |
| `kt-kernel/ext_bindings.cpp` | Bindings for all of the above, incl. `AVX512RawInt4Packed_MOE`. |
| `kt-kernel/python/utils/amx.py`, `loader.py` | RAWINT4 backend selection, W4AFP8 loader shim, host-RAM expert tier. |
| `*.bak-preC` files | Pre-change snapshots kept deliberately for diffing. Not live code. |

⚠️ **Build note:** build with `install.sh build` + `CPUINFER_USE_CUDA=1`. A manual
`cmake` produces a CPU-only `.so` that silently drops `submit_with_cuda_stream`.
Set `CMAKE_PREFIX_PATH` / `PKG_CONFIG_PATH` to the venv or cmake reports "NUMA not found".

## 3. What was NOT changed

- The model weights are stock GLM-5.2 W4AFP8; only the CPU-side expert repack
  (`int4_scripts/gptq_full_repack.py`) touches them, and it is reversible.
- `glm4_moe.py` was edited early on and is **dead code** — GLM-5.2 (`GlmMoeDsa`)
  dispatches through `deepseek_v2.py`. Do not trust anything routed there.

---

## Reproducing the environment

`setup.sh` + `int4_scripts/download_w4afp8.py` + `run_fast.sh`. Full procedure in
[`SETUP_RUNBOOK.md`](SETUP_RUNBOOK.md) and [`PORTABLE_DEPLOYMENT.md`](PORTABLE_DEPLOYMENT.md).
Weights are not in this repo and never should be.

## Archive notes

- This repo is a **backup of an active research project**, not a released product.
  Many conclusions in the `.md` files are provisional or were later refuted —
  [`STATUS_OF_FINDINGS.md`](STATUS_OF_FINDINGS.md) grades them.
- History was rewritten once before publishing, to remove a machine password that
  had been committed in `6b1e536` and scrubbed in `cc0e49f`. Commit SHAs in old
  write-ups may therefore not resolve; the commit *subjects* still do.
- Paths like `/data/models/…`, `/cache/nvme0` and the 2×H100 / Zen4 assumptions
  are specific to the machine this ran on.
