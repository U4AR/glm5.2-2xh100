# Whole-document QA: a 533k-token document held in context

This is the setup for asking questions about one large document with the *entire*
document resident in the model's context — no retriever, no chunking, no
embeddings. GLM-5.2 has a 1,048,576-token position budget, and the document
(946 pages of OCR) tokenizes to **533,594 tokens**, so it fits with room to spare.

The whole thing rests on two properties:

* **NSA sparse attention** makes attention over half a million tokens affordable —
  `O(L·2048)` instead of `O(L²)`. It was believed broken on this box; it wasn't.
  See "NSA / IndexShare" below.
* **The radix cache** makes the document a *prefix*. The first request pays the
  full prefill once; every question after it reuses that KV and only prefills its
  own handful of tokens.

## Run it

```bash
# 1. Server: NSA on, 549k-token KV pool, MTP depth-3.
DISABLE_NSA=0 SPEC_DRAFT_ATTN=nsa GPU_EXPERTS=60 MEM_FRACTION=0.93 \
MAX_TOTAL_TOKENS=548864 CONTEXT_LENGTH=548864 CHUNKED_PREFILL=4096 \
MTP=1 MODE=safe KEEP=8 ./run_fast.sh

# 2. Pay the document prefill once (~26 min). Do this before opening the UI,
#    or the first question is the one that waits.
python3 bench/doc_context_bench.py --warm-only

# 3. The UI.
python3 doc_ui.py 8081 unlimited_ocr.txt
#    -> open forwarded port 8081
```

> **MTP needed a fix to be safe here** (now applied, see "MTP at long context").
> Before it, `MTP=1` corrupted individual tokens at 533k and collapsed into
> `0,0,0,0,...` at temperature 0.

> Any change to `SYSTEM_PROMPT` changes the prefix and forces a fresh 26-minute
> prefill. The prompt deliberately does **not** name the file, so renaming or
> moving the document is safe; only editing the instructions or the text itself
> costs a re-prefill.

`bench/doc_context_bench.py` imports `SYSTEM_PROMPT` from `doc_ui.py` on purpose, so
both send a **byte-identical** prefix and share one cache entry. Change the prompt
in one place only, or you will pay the prefill twice.

## Why these numbers

Per card (95.8 GB), at 533k tokens:

| item | cost |
|---|---|
| MLA KV, fp8 | 43.9 KB/token → 24.0 GB |
| NSA index-k cache (Full layers only) | ~6.0 KB/token → 3.3 GB |
| **KV pool @ 548,864 tokens** | **27.6 GB** |
| model weights @ `GPU_EXPERTS=60` | ~60 GB |
| free (GPU prefill scratch needs ~5 GB) | ~7.5 GB |

Each GPU expert slot costs ~0.675 GB, so the KV pool and the expert count trade
against each other directly. `GPU_EXPERTS=60` is the largest that still leaves
prefill scratch. Without the index-k saving described below it would be ~44.

Context budget: the document plus its instructions is **533,764 tokens**, leaving
**15,100** for questions and answers — roughly 8-10 turns. "New chat" in the UI
drops the turns while keeping the document cached, so resetting is instant.

## Decode throughput

Measured at `GPU_EXPERTS=60`, NSA on, MTP depth-3. The tier is chosen **per request**
via the OpenAI `model` field (`GLM5.2-top4`), so the UI switches it live with no
restart. TTFT is ~2.2s against the cached document prefix:

| routing | tok/s @ 533k | (MTP off) | when |
|---|---|---|---|
| exact (top-8) | **13.1** | 10.1 | default — every routed expert as the router chose |
| top-4 | **20.6** | 13.1 | tail substituted; no quality loss observed |
| top-2 | **27.9** | 15.3 | faster; occasional repetition |

Holding half a million tokens of KV still costs real decode speed on its own —
the same box at a short prompt does ~16.6 / ~27 / ~33.

Full prefill: **1560s** for 533,781 tokens (~344 tok/s), flat per chunk.

## NSA / IndexShare

`DISABLE_NSA=1` used to be the default because NSA produced gibberish past 2048
tokens. The recorded root cause ("the topk kernels hard-assert topk==2048") was
wrong: those are *shape* asserts and this model's `index_topk` **is** 2048.

The actual cause is that GLM-5.2 trains with **IndexShare** — one lightning indexer
per group of 4 layers, placed on the first layer of the group, its top-k reused by
the rest. The checkpoint states this (`indexer_types`, `index_topk_freq=4`) and
ships indexer weights for only **21 of 78** layers (0, 1, 2, then every 4th), plus
the NEXTN layer. This sglang build had no support for any of it and ran all 78
layers through their own indexer — 57 of them scoring with never-initialised
weights. It looked fine below 2048 only because the topk kernel emits the trivial
`[0..L-1]` selection there, which is what made the cliff so abrupt.

Backported in `.venv/.../sglang/srt/`:

* `configs/model_config.py` — `get_nsa_indexer_pattern()`, `nsa_layer_has_own_indexer()`
* `models/deepseek_v2.py` — build the indexer only on Full layers; they stash
  `topk_indices`, Shared layers read it back
* `models/deepseek_common/attention_forward_methods/forward_mha.py` — guard the
  prefill-path indexer call
* `mem_cache/memory_pool.py`, `model_executor/model_runner_kv_cache_mixin.py` —
  Shared layers get a one-page stub index-k buffer instead of a full one (nothing
  reads it), worth **~4 GB/card** at this context length
* `layers/attention/nsa_backend.py` — `NativeSparseAttnMultiStepBackend` was missing
  `on_after_cuda_graph_warmup_pass`, the same gap already patched for the triton
  draft backend; MTP-on-NSA crashes at boot without it

Verified by needle retrieval at 6k / 16k / 27k tokens — exact and coherent, where
the old build emitted gibberish.

### Two traps

* `scripts/hardware_profile.py` exports `ATTENTION_BACKEND=flashmla`, and the
  profile loop in `run_fast.sh` fills in anything unset — which beats
  `run_server_int4.sh`'s `${ATTENTION_BACKEND:-nsa}`. Asking for `DISABLE_NSA=0`
  used to silently still run dense MLA. `run_fast.sh` now claims the variable first.
* MTP's usual `SPEC_DRAFT_ATTN=triton` rejects NSA's `q_rope`. Use
  `SPEC_DRAFT_ATTN=nsa`; the NEXTN layer is itself a DSA layer with its own indexer.

## MTP at long context

With MTP on, output at 533k corrupted individual tokens ("PRVs" came out as `0`)
and, at temperature 0, collapsed into `0,0,0,0,...` within a few tokens. It was
clean at 27k and 126k, so it looked at first like the model simply running out of
road far past its trained context.

It was a real defect, in `NativeSparseAttnMultiStepBackend`: **every speculative
step was handed the same attention metadata**. The whole draft loop is captured in
one CUDA graph, so `eagle_draft_cuda_graph_runner` calls
`init_forward_metadata_replay_cuda_graph` once and the wrapper has to stage all the
steps itself. It computed metadata once from `attn_backends[0]` and copied it to
every step — but draft step *i* attends to the prompt plus the *i* tokens the
earlier steps just drafted, so its sequence is `seq_lens + i` and its page table is
*i* entries longer. Steps 1 and 2 could not see the KV of the tokens they had just
produced.

Two things confirm that was the intended contract:

* `speculative_step_id` is passed to each sub-backend, stored, and then never read
  anywhere.
* The triton draft backend — which did work at depth-3 — does this explicitly:
  `generate_draft_decode_kv_indices` copies `seq_len` entries and then appends
  `iters` more from `token_pool_ptr + seq_len + ...`. The draft slots are already
  reserved in `req_to_token`; NSA simply never widened its slice.

Why it stayed invisible below ~126k: `compute_nsa_seqlens` clamps to
`index_topk` (2048), and the topk selection is the trivial `[0..L-1]` there, so
stale metadata barely matters. Well past 2048 each step's top-k and page mapping
genuinely differ, the draft is scored against the wrong keys, and verification
starts accepting tokens the target model would never emit.

Fix: per-step metadata (`seq_lens + i`, wider page indices) in both the replay and
capture paths. Also removes the shared-source multi-backend fused copy, which is
unusable once each step has its own source. The fused metadata-copy kernel itself
was **not** at fault — run with `SGLANG_VERIFY_FUSED_METADATA_COPY=1` it reports
zero mismatches.

Verified at 533k, temperature 0 and 0.2: correct answers, correct citations, no
degeneration — and MTP is worth +30% to +82% depending on tier.
