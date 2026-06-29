# The top-2 expert trick: 1.5× faster GLM-5.2 decode by *substituting* the tail

*A follow-up to [BLOG.md](BLOG.md). We had GLM-5.2 decoding at 14.7 tok/s on a
2×H100 box. This is the story of squeezing ~1.5× more out of it by routing fewer
experts to the CPU — and, just as importantly, how we kept ourselves from
believing a speedup that was quietly breaking the model.*

---

## Where the time goes

GLM-5.2 routes every token to **8 of 256** experts per layer. On this box only
~104 of those experts fit on the GPUs; the other 152 live in CPU DRAM and are
computed by AVX-512 kernels. Per decode step the engine does, for each of the 75
MoE layers: stage the activations, **submit** the CPU experts on a side stream,
run the GPU experts, then **sync** and merge. Profiling showed the GPU sits
**idle ~63% of every decode step** — ~44 ms of the 70 ms step — stalled once per
MoE layer on that CPU submit/sync handshake.

So the lever is obvious: **put fewer experts on the CPU.** If a token's 8 experts
were all GPU-resident, the CPU path would be empty and the stall would vanish.

The catch: you can't just *drop* the experts that aren't on the GPU. Which leads
to the idea this post is about.

## The idea (and the paper-shaped intuition behind it)

> Keep the genuinely most-important experts; replace the rest with whatever good
> experts are already on the GPU.

Concretely, after the router picks its top-8 for a token: **keep the true top-K**
(highest gate weight) and **substitute the other 8−K** with the highest-scoring
experts that are *already GPU-resident*. The router's own logits tell us which
GPU experts are the "best available" — no similarity table, no weight transfer.

This is a bet that MoE routing is **forgiving about the tail**: the top experts
carry the decision, the rest are refinement, and refinement can be approximated.

## First, measure: is GLM actually top-heavy?

Before writing the hot path, we dumped the router weights over thousands of real
token×layer decisions and asked how much of each token's routed mass lives in its
top-K (weights renormalized to sum to 1):

| keep | share of routed mass |
|---|---|
| top-1 | 26.6% |
| **top-2** | **43.9%** |
| top-3 | 57.1% |
| top-4 | 68.0% |

GLM-5.2 is **not** top-heavy. The top-2 hold under half the mass; the model
genuinely spreads work across its 8 experts. That's an early warning that a naive
"top-2 is all you need" will lose a lot of signal.

## drop vs. sub: the whole ballgame

Two variants, same speed, very different behavior:

- **drop-K**: keep the true top-K, *zero* the rest, renormalize. At K=2 the output
  is **garbage** — degenerate repetition. Two contributing experts is simply too
  few; the 44% mass figure was right.
- **sub-K**: keep the true top-K, *substitute* the rest with the best GPU-resident
  experts. At K=2 the output is **coherent** — the bat-and-ball riddle is solved
  correctly, code and prose come out clean.

Same number of contributing experts (8 either way). The difference is that
substitution keeps *eight* experts in the mix; dropping leaves two. **The model
needs ~8 contributors, but which ones in the tail is negotiable.** That single
contrast is the core finding.

## The speed/quality curve

Routing fewer experts to the CPU does pay off, and the payoff scales with how
aggressively you substitute. Decode throughput, 5-run median, temperature 0,
`GPU_EXPERTS=104`:

| config | tok/s | speedup | quality |
|---|---|---|---|
| baseline (all 8 as routed) | 14.7 | 1.00× | reference |
| **sub-4** (keep top-4, substitute 4) | 18.5 | **1.25×** | clean — no degeneration found |
| **sub-2** (keep top-2, substitute 6) | 22.0 | **1.49×** | mostly clean; rare repetition loops |
| sub-0 + CPU-skip (substitute all 8) | 29.2 | 1.98× | degenerates readily — not viable |

`sub-0` is special: with *every* expert GPU-resident, the CPU path is genuinely
empty, so we can also **skip the per-layer submit/sync entirely** (a one-line guard
in the kt wrapper). That removes the 44 ms stall and reaches ~2×. The 29 vs. the
34 tok/s we'd measured for "skip with baseline routing" is because `sub-0` runs all
8 experts on the GPU instead of ~3 — more GPU GEMM, but the bubble is gone.

## The part where we almost fooled ourselves

`sub-0` passed every quick check: bat-and-ball correct, "capital of Japan? Tokyo,"
a tidy two-sentence Rayleigh-scattering answer. At 2× and "coherent," it was
tempting to call it a win and ship it.

Then we asked it for a one-line Python string reversal. The reasoning trace:

> *…reversing a given string character-wise, reversing the characters in a given
> string, reversing a given string character-wise, reversing the characters in a
> given string, reversing a given string character-wise…*

— forever. `sub-0` **degenerates** into repetition loops; the short prompts had
passed by luck. Dropping *all* the genuine experts loses the routing "identity,"
and on any prompt that needs a few tokens of real work the model falls into a rut.

`sub-2` was much better but not immune: it looped on a haiku (syllable counting).
So we ran the **decisive control** — the *same* prompts on the unmodified baseline.
Baseline counts syllables across varied candidate lines and finishes cleanly; it
does **not** loop. So the loop is real, caused by substitution, and not just "LLMs
are bad at haiku."

The lesson, and the reason this section exists: **a fluent-looking short answer is
not coherence.** You only see the degeneration on prompts that force sustained
generation, and you can only attribute it with a baseline control on identical
inputs. Throughput numbers are easy; quality verdicts are where these projects go
wrong.

## What we'd actually recommend

- **`sub-4` (1.25×)** is the free lunch: no quality regression we could find.
- **`sub-2` (1.5×)** — the "top-2" headline — is a good default if you can tolerate
  the occasional repetition loop on open-ended generation. It keeps the two experts
  that matter most and substitutes the rest.
- **`sub-0` (2×)** is off the table for quality.

The mechanism overturned one of our own earlier conclusions, too: we'd previously
written that "expert placement is decode-speed-neutral." That rested on a reroute
hook sitting in the wrong file (`glm4_moe.py`) for this architecture — GLM-5.2's
MoE actually runs through `deepseek_v2.py`, so the old hook was **dead code that
never executed**. With a working reroute in the right place, cutting CPU experts
per layer from ~5 to ≤2 demonstrably speeds decode up. Placement matters after all.

## The honest ceiling, and the next move

The clean way to get `sub-0`'s speed at `sub-4`'s quality is to **keep the true top
experts but stream their weights onto the GPU** so the CPU path is empty *and* the
real experts are present. That trades the per-layer CPU stall for PCIe traffic and
some CUDA-graph-capture complexity — promising, unbuilt, and the next thing to try.

For now: a measured, coherent **1.25–1.5×** on top of the existing 14.7 tok/s, from
a tiny logical reroute and zero weight movement. Run it with
[`run_fast.sh`](run_fast.sh) (`KEEP=2` by default, `KEEP=4` for the safe setting).

*Reproduce: `KEEP=4 ./run_fast.sh`, or `./run_fast.sh` for the top-2 default. The
reroute is gated by `/tmp/kt_topk_mode`; remove it for plain baseline.*
