# Fused NSA metadata-copy kernel — investigation closure (2026-06-24)

**TL;DR: the fused metadata-copy CUDA kernel is NOT our MTP bug and NOT a speedup
lever. It is never called in our config, and even where it runs it only saves
µs-scale launch overhead. The earlier "buggy fused kernel" root-cause was WRONG.**

## What was claimed before
Prior session concluded MTP produced garbage because of a buggy fused NSA
metadata-copy CUDA kernel (`sglang.jit_kernel.fused_metadata_copy`) and that
`SGLANG_USE_FUSED_METADATA_COPY=0` fixed it.

## What is actually true (verified three ways)
1. **Never called at `--speculative-num-steps 1`.** The only production caller of
   `fused_metadata_copy_cuda` is
   `NativeSparseAttnBackend.init_forward_metadata_replay_cuda_graph_from_precomputed`
   (nsa_backend.py:1181). That is invoked ONLY from
   `NativeSparseAttnMultiStepBackend.init_forward_metadata_replay_cuda_graph`
   inside `for i in range(speculative_num_steps - 1)`. We run `num_steps=1` →
   `range(0)` → empty → the fused path never executes.
2. **Live server confirms it.** `get_server_info` → `num_steps=1`; the running
   server log (`serve_mtp_fix_*.log`) has ZERO occurrences of
   "fused metadata" / "VERIFICATION" / "individual copies".
3. **The kernel is correct anyway.** Standalone production-faithful test
   (`scratchpad/test_fused_bug.py`) replicates the exact call-mapping +
   reference-fallback for the *untested* TARGET_VERIFY (mode 1) and DRAFT_EXTEND
   (mode 2) paths over seq_len 100→9000, bs 1/2 → **bitwise-identical** to the
   reference in every case. (Upstream `test_fused_metadata_copy.py` only tests
   DECODE/mode 0, comment: "other modes not fully tested yet" — but verify/draft
   happen to be right.)

## Therefore
- `SGLANG_USE_FUSED_METADATA_COPY=0` is a **no-op** in our setup.
- The real garbage→coherent fix was **`SGLANG_ENABLE_SPEC_V2=True` alone** — it
  flips on the overlap scheduler + the correct spec-v2 verify path. spec-v2 is
  auto-enabled only for `DeepseekV4ForCausalLM` (server_args.py ~1413), NOT our
  `GlmMoeDsa`, so it must be forced.
- Even where the fused kernel DOES run (num_steps≥2 single / ≥4 multi-backend),
  it only collapses a few int32 metadata copies into one launch — negligible vs
  the kt CPU-expert MoE compute that dominates decode. **Fixing/optimising it
  cannot move tok/s.** Option 3 (as originally framed) is a dead end.

## Working baseline config (keep this)
`SGLANG_ENABLE_SPEC_V2=True SPEC_DECODE=1 GPU_EXPERTS=48 ./run_server.sh`
→ coherent, accept ~1.9, ~8.8 tok/s ≈ the no-MTP baseline (8.7). MTP is a WASH
here because the speculative verify tokens cost real kt CPU-expert compute.

## Where real speedup lives (see INT4_PROJECT.md)
The bottleneck is the CPU-expert critical path (only 48/256 experts on GPU).
Real levers: (a) more GPU experts by freeing VRAM, (b) **int4 CPU experts** via
kt — the chosen next project.
