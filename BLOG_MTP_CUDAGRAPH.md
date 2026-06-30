# Making MTP actually work: how one unregistered buffer cost us 55% of our speed

*A follow-up to [BLOG_TOP2_EXPERTS.md](BLOG_TOP2_EXPERTS.md). We had GLM-5.2
decoding at 22 tok/s on a 2×H100 box with the top-2 expert trick. Multi-Token
Prediction (MTP) was supposed to push it higher — and for weeks it didn't,
producing fluent-looking garbage like "**The CPU metamorphosis metamorph
metamosis…**". This is the story of chasing that garbage to a single line of
buffer-registration code, and walking out with **34 tok/s, coherent**.*

---

## What MTP is supposed to do

GLM-5.2 ships a NEXTN/MTP head: a small draft layer that proposes the next few
tokens, which the full model then *verifies* in one batched forward. If the
draft is good, you emit several tokens per expensive forward pass instead of one.
On a GPU box this is a clean 2–4× win.

On *our* box it's subtler, because the model is split: the draft runs pure-GPU,
but the **verify** runs the real 754B model — whose experts live in CPU DRAM and
are computed by AVX-512 kernels (see [the top-2 post](BLOG_TOP2_EXPERTS.md)). So
verify still pays the CPU-expert tax. The hope: emit 2–3 tokens per CPU-expert
sweep instead of one.

Every prior attempt produced garbage the moment CUDA graphs were on. The
accepted wisdom — recorded across several debugging sessions — was "MTP is a wash
on this stack; the verify forward corrupts under graph replay." That turned out
to be three wrong diagnoses stacked on a real bug.

## The tell: eager works, graphs don't

The single most useful experiment was the cheapest. Turn CUDA graphs **off**
(`DISABLE_CUDA_GRAPH=1`) and run MTP:

```
The CPU, or Central Processing Unit, acts as the "brain" of a computer,
executing instructions and performing calculations…     accept 1.9 ✓
```

Perfect. Coherent, real acceptance. Turn graphs back **on**:

```
The CPU metamorphosis metamorph metamosis the a the the of the CPU metamosis…
```

That cleanly rules out a *lot*: the draft is fine, the verify *math* is fine, the
routing is fine. The bug is specifically in **CUDA-graph capture/replay of the
verify forward**. Eager has no replay, so eager is correct.

## The misdirection: it's not the attention

The obvious suspect was the attention backend's graph metadata — and there's even
an upstream blog (DeepSeek-V4's "in-graph metadata") describing exactly this class
of bug. We chased it hard:

- **flashmla** verify metadata: capture and replay write the same fixed buffers
  in-place. Consistent. Not it.
- **compressed** backend (the upstream fix for this exact problem): asserts
  `head_dim == 512`. GLM-5.2's MLA head_dim is 192. Structurally incompatible —
  that fix is bound to DeepSeek-V4's geometry.
- **`SGLANG_KT_HYBRID_NO_CPU_STREAM=1`** (serialize the CPU path, kill any race):
  still garbage. So it's **not a race** — it's **stale data**.

The decisive isolation: route *every* expert to the GPU (our `KEEP=0` mode, which
skips the CPU-expert path entirely) and run MTP with graphs on:

```
A CPU serves as the central brain of a computer, executing commands…   ✓ 39.85 tok/s
```

**Coherent.** So the corruption isn't the attention and isn't the verify — it's
the **kt CPU-expert path**, specifically for the multi-token verify batch, under
graph replay. Any CPU-resident expert in the verify → garbage; zero CPU experts →
clean.

## The bug: one batch size nobody registered

kt computes CPU experts via `cudaLaunchHostFunc` — a **host node** captured into
the CUDA graph. At replay it re-fires with the argument pointer **frozen at
capture time**. That's fine *if* the pointer aims at a graph-stable buffer that
gets refreshed each step. kt keeps exactly such buffers — but only for batch
sizes it was told to pre-allocate:

```python
# kt_kernel/experts_base.py
if batch_size in cls.capture_bs:
    cls.capture_buffers[batch_size] = cur_buffer   # persistent, graph-stable
else:
    cls.temp_buffer = cur_buffer                   # transient, reallocated per call
```

And who tells kt which sizes to capture? sglang, in one line:

```python
# cuda_graph_runner.py
KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)   # the DECODE batch sizes
```

Here's the bug. The plain decode forward processes `bs` tokens. But the
**TARGET_VERIFY** forward processes `bs × num_draft_tokens` tokens — the engine
even sets `num_tokens_per_bs = speculative_num_draft_tokens` a few lines up. That
larger token count was **never registered** with kt. So the verify forward fell
through to the **transient** buffer, whose pointer goes stale the instant the
graph replays without re-running Python — and the CPU experts dutifully computed
on freed/overwritten memory. Fluent garbage.

This is why `KEEP=0` was the *only* config that ever worked: it skips the CPU
buffer entirely, so there's nothing to go stale.

## The fix: two small, surgical changes (no rebuild)

**1. Register the verify token counts** (`cuda_graph_runner.py`):

```python
_kt_existing = set(KTMoEWrapper.get_capture_batch_sizes() or [])
_kt_token_counts = {bs * self.num_tokens_per_bs for bs in self.capture_bs}
KTMoEWrapper.set_capture_batch_sizes(sorted(_kt_existing | _kt_token_counts))
```

The union matters: `set_capture_batch_sizes` *replaces* the list, and the decode
and verify graph runners are separate objects — without the union, the second
clobbers the first.

**2. A missing no-op hook** (`triton_backend.py`): deeper MTP (`num_steps > 1`)
crashed because `TritonMultiStepDraftBackend` lacked
`on_after_cuda_graph_warmup_pass`, which the draft graph runner calls
unconditionally. We added it (delegating to the per-step sub-backends), unblocking
multi-step drafting.

Both are pure-Python patches to the venv. **No `.so` rebuild.**

A note on the draft backend, since it cost real time: the draft's attention has to
be `triton`. `flashmla` draft crashes (`PrefillMetadata` has no `block_kv_indices`),
`fa3` crashes (`set_mla_kv_buffer` on a `None` rope tensor), `compressed` hits the
`head_dim==512` assert. Only triton survives the draft-extend path on this build.

## The payoff

With the verify buffer registered, MTP is coherent under graphs at **any** routing
— including plain 14.7 tok/s baseline routing. And because deeper drafting just
means a larger (still-registered) verify batch, we could finally sweep depth:

| KEEP=2 config            | accept | tok/s | vs no-MTP |
|--------------------------|:------:|:-----:|:---------:|
| no-MTP (previous default)| 1.0    | 22.0  | —         |
| MTP depth-1              | 1.7    | 29.8  | +36%      |
| **MTP depth-3**          | **2.5**| **34.1** | **+55%** |
| MTP depth-5              | 2.6    | 28.9  | (regresses) |

Plain baseline routing gains too: 14.7 → 17.5 (+19%).

**Depth-3 is the peak, and the reason is the whole point of this box.** The draft
runs on the idle GPU for free, so acceptance keeps climbing with depth (1.7 → 2.5
→ 2.6). But the *verify* runs CPU experts, and a depth-5 draft makes the verify
chew through 6 tokens of CPU-expert compute per step. Past depth-3, that verify
cost outruns the acceptance gain. It's the CPU-bound-box version of the
"depth-4 is unstable" wall that GPU-streaming setups hit for a different reason.

## Where we landed

**`MTP=1 ./run_fast.sh`** (now the default) serves GLM-5.2 at **~34 tok/s,
coherent** on two H100s — **2.3× the 14.7 tok/s** we started the week at, and
**1.55× the shipped top-2** server — by stacking two orthogonal tricks: substitute
the low-weight experts onto the GPU (top-2), and emit multiple tokens per
CPU-expert sweep (MTP depth-3).

The lesson, again: the speedup that looked impossible for weeks was a real,
fixable bug hiding behind three plausible-but-wrong explanations — and the way out
was the cheapest experiment (eager vs graphs) plus refusing to trust a "wash"
verdict that the data didn't actually support.

*Fix committed in `67db44e`. Two-file venv patch, no rebuild. Launch:
`MTP=1 ./run_fast.sh`.*
