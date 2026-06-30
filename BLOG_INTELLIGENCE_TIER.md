# Pick your speed at call time: a per-request "intelligence" dial for GLM-5.2

*A follow-up to [BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md) and
[BLOG_MTP_CUDAGRAPH.md](BLOG_MTP_CUDAGRAPH.md). We already had GLM-5.2 decoding
fast on a 2×H100 box by **substituting** the tail experts (the "top-K" trick).
The catch: the tier was a server-launch flag. You picked top-2 OR top-4 OR
baseline when you started the server, and changing your mind meant a restart.*

*This post makes the tier a property of the **request**, not the server. The
OpenAI `model` field selects it — `GLM5.2-top8`, `GLM5.2-top4`, `GLM5.2-top2`,
`GLM5.2-top0` — live, with no restart, and **different tiers can share one batch
and one CUDA graph**.*

---

## The idea in one line

> The same way you'd switch between `gpt-4` and `gpt-4-mini`, switch between
> "smart GLM" and "fast GLM" by changing the model name — except here both are
> the *same weights*, and the dial is *how many experts per token actually run*.

`GLM5.2-topN` keeps the **N genuinely most-important experts** for each token and
substitutes the other `8−N` with the best GPU-resident experts (see the
[top-2 post](BLOG_TOP2_EXPERTS.md) for why that preserves coherence). `N=8` is the
untouched baseline; `N=2` is the fast default; `N=0` is fastest-but-degrades.

```bash
# smartest (baseline routing)
curl localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top8", ...}'
# fast (default)
curl localhost:8000/v1/chat/completions -d '{"model":"GLM5.2-top2", ...}'
```

Nothing about the server changes between those two calls. They can even be
**in flight at the same time**.

## Why this is harder than a global flag

The tail-substitution is a tiny tensor edit on the router output inside every MoE
layer. The problem is *where* that edit lives at decode time: **inside a captured
CUDA graph**. The graph is recorded once at startup; on every token we just
*replay* it. A Python `if tier == 2` doesn't run on replay — the value is frozen
into the recording. So a global tier flag was the natural design, and changing it
meant re-recording the graph (a restart).

To make the tier vary **per request without re-recording**, it has to become
*data the graph reads*, not *code the graph runs*. Concretely: a per-token
integer `K`, living in a **persistent input buffer** that the graph references,
which we overwrite before each replay — exactly how `input_ids` and `positions`
already work.

## The plumbing (six small, boring edits)

The whole feature is threading one `int` from the HTTP request down to a GPU
buffer, and rewriting the routing edit to read it per-token:

1. **API** (`serving_chat.py`): parse `-topN` off the model name → stash `K` in
   the request's `custom_params`. (Gotcha: `to_sampling_params()` returns a
   *dict* here, not an object — the first boot died with
   `'dict' has no attribute custom_params`.)
2. **Batch** (`schedule_batch.py`): carry a per-request `kt_keep_k` list on the
   `ModelWorkerBatch` (it's already broadcast to every TP worker).
3. **Forward** (`forward_batch_info.py`): expand per-request `K` into a
   **per-token** `int32` tensor in `ForwardBatch.init_new` (`repeat_interleave`
   by tokens-per-request — covers prefill, decode, and MTP verify).
4. **Graph** (`cuda_graph_runner.py`): a persistent `keep_k` buffer (sized
   `max_num_token`, default `-1` = "use server default"), copied in before each
   replay and sliced into the captured `ForwardBatch`. **This is the line that
   makes switching free** — one graph, buffer updated per replay.
5. **Kernel** (`deepseek_v2.py`): rewrite the routing edit to be **branch-free
   and per-token** — rank the 8 selected experts by weight, keep those with
   `rank < K[token]`, fill the rest from the best GPU-resident experts. `K==8`
   is provably the identity (mask all-true, renorm is a no-op).
6. **Discovery** (`http_server.py`): advertise `-top{8,4,2,0}` in `/v1/models`.

A nice property falls out for free: a request with no suffix, a request asking
`-top8`, and a request asking `-top2` are all the *same* captured kernel —
`top8` just produces a mask that changes nothing. No special-casing.

## Does it actually switch — and does it cost anything?

Three things had to be true. All measured on the live server (GLM-5.2 W4AFP8,
2×H100, MTP depth-3 on):

**1. The tier is really active.** Single-stream decode, monotonic with `K`:

| model | tok/s | notes |
|---|---|---|
| `GLM5.2-top8` / bare | ~20 | baseline routing |
| `GLM5.2-top4` | ~27 | clean |
| `GLM5.2-top2` | ~40 | default |
| `GLM5.2-top0` | ~75 | degenerates (documented) |

**2. No quality or speed regression on the full path.** `GLM5.2` (bare) and
`GLM5.2-top8` produce **bit-identical greedy output** — the `K=8` path is the old
baseline, unchanged. The added per-token tensor ops are negligible (top-8 ≈
baseline tok/s).

**3. No graph recapture, ever.** The server logs **zero CUDA-graph captures after
startup**; decode steps run under `cuda graph: True` regardless of which tier each
request asks for. Switching `top8 → top2 → top4` across consecutive requests is
just three different buffer values fed to the same recording.

And the one that matters for serving: **mixed tiers in one batch stay coherent.**
Firing `top8 + top4 + top2 + top0` concurrently, every stream comes back correct —
the per-token `K` means each token in the shared batch is routed by its own
request's dial.

## Batching: what concurrency buys you

With the server configured for concurrency (`cuda_graph_max_bs=8`,
`max_running_requests=8`, `GPU_EXPERTS=96` to free KV headroom), top-2 aggregate
throughput as you add concurrent users:

| concurrent requests | top-2 **+ MTP** (agg tok/s) | top-2 **no-MTP** (agg tok/s) |
|---|---|---|
| 1 | 32 | 21 |
| 2 | 44 | 34 |
| 4 | 56 | 55 |
| 8 | 73 | 78 |

Two takeaways:

- **Batching works with the per-request dial** — aggregate throughput more than
  doubles from 1 → 8 streams (32 → 73 tok/s), and it composes with mixed tiers.
- **MTP is a single-user optimization.** Speculative decode harvests the idle
  headroom of a lone stream (32 vs 21 tok/s at B=1, +56%), but once real
  concurrency fills the pipe its verify cost stops paying for itself — the two
  curves cross around B=4 and no-MTP is slightly ahead at B=8. So: **MTP on for
  interactive single-user, MTP off for a busy multi-user endpoint.** Both are the
  *same code*; it's a launch flag (`MTP=0 ./run_fast.sh`).

(Single-stream top-2 hits ~40 tok/s at the latency-tuned `GPU_EXPERTS=104,
cuda_graph_max_bs=1` config; the table above uses the throughput-tuned config, so
its B=1 number is a touch lower. Same dial, different operating point.)

## The honest caveats

- **Mixed-batch CPU cost is shared.** GLM's CPU-resident experts are computed once
  per layer for the *union* of experts any token in the step needs. So a `top8`
  request riding in a batch makes that step pay roughly baseline CPU cost — the
  fast requests sharing it don't get their *full* speedup while overlapping a slow
  one. Throughput stays bounded between "all-baseline" and "all-top2". This is
  intrinsic to shared-batch MoE, not a bug.
- **`top0` still degrades.** Dropping *all* true experts collapses into repetition
  (the model needs ~8 contributing experts; *which* tail experts is forgiving,
  but the top ones carry the identity). It's exposed because the dial is
  continuous, but it's labeled "may degrade" for a reason.
- **`top0`'s extra CPU-skip isn't wired for mixed batches** (it would need a
  second captured graph), so `top0` here runs without that last ~1.2×.

## How to use it

```bash
./run_fast.sh                       # default: top-2 + MTP, latency-tuned
GPU_EXPERTS=96 MAX_RUNNING=8 CUDA_GRAPH_MAX_BS=8 MTP=0 ./run_fast.sh   # throughput
```

Then just set the model name per call: `GLM5.2-top8` when you want the model's
full attention, `GLM5.2-top2` when you want the answer now. Any `N` in `0..8`
works; the UI ([chat_ui.py](chat_ui.py)) exposes it as an "intelligence" dropdown.

*One model, one set of weights, one running server — and a quality/latency dial
the caller controls, token by token.*
