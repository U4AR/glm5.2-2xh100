"""Capture-safe chain lookahead: drive the real prefetch from a walked stream.

`bench/chain_predict.py` is the INSTRUMENT -- it scores predictions, runs
arbitrary Python, and only works eager. This file is the MECHANISM: it does the
minimum needed to fire a prefetch and it does it inside the decode CUDA graph,
because a predictor that cannot be captured cannot run during decode at all.

WHAT IT DOES, at the end of every layer L:

    state = hidden_states + residual        # layer L's REAL output stream, free
    for s in 1 .. depth-1:
        state += approx_moe(L+s, post_ln_{L+s}(state))
    prefetch( layer L+depth, top-P of gate_{L+depth}(post_ln(state)) )

Two things about that shape are deliberate.

FIRST STEP IS FREE AND EXACT. At the post hook layer L's true MoE update is
already in hand (`hidden_states + residual` IS the stream entering L+1, minus
its attention), so the walk starts from truth and only approximates the layers
after it. Stage G measured that arm -- `post` -- at 83.1% whole-layer coverage
against `chain`'s 81.9% at depth 1, i.e. the exact first step is BETTER than
approximating it, and it costs nothing. At depth 1 this file therefore does no
expert math at all; the walk only begins at depth 2.

THE APPROXIMATION NEVER TOUCHES THE CPU. It calls `gpu_method.apply` directly
rather than the kt wrapper's `apply`, which would submit and sync a CPU expert
batch per call -- and per Stage E that fixed submit/sync is ~86% of the CPU
pole, so routing the walk through it would cost more than the whole prefetch
saves. Expert ids are substituted onto GPU-resident experts first, then remapped
with kt's own `mask_and_remap_expert_ids`, so the GPU kernel sees exactly what
it sees on the real path.

CAPTURE-SAFETY RULES this file obeys, all of which have bitten this project
before: no `.item()`, no `.cpu()`, no `torch.cuda.synchronize`, no Python branch
on a tensor VALUE, and every module registered at CONSTRUCTION rather than on
first forward -- capture happens before any request is served, so a dict
populated on first forward is empty at capture time and the prefetch is captured
as nothing at all, permanently.

KNOBS
  KT_CHAIN_PF=1            enable
  KT_CHAIN_PF_DEPTH=d      layers ahead to predict and fetch (default 2)
  KT_CHAIN_PF_K=2          expert slots the APPROXIMATE forward uses. 8 keeps
                           every routed slot (most accurate); 2 keeps only the
                           top two, which is ~4x less expert GEMM. The walk is
                           pure overhead, so this is the main cost lever.
  KT_CHAIN_PF_P=2          size of the predicted set handed to the prefetcher.
  KT_CHAIN_PF_SUB=search   "search" = nearest-equivalent substitution (~20
                           kernels/layer, 6.55 ms/step); "drop_raw" = keep the
                           resident slots and leave the weight sum short (~3
                           kernels, 1.55 ms, 2 coverage points behind search);
                           "drop" = the same but renormalised, which MEASURED
                           WORSE THAN NOT WALKING -- kept only so the comparison
                           stays reproducible. See Stages H2 and H3.
"""

import os
from typing import Dict, Optional

import torch

ENABLED = os.environ.get("KT_CHAIN_PF") == "1"
_DEPTH = max(1, int(os.environ.get("KT_CHAIN_PF_DEPTH", "2")))
_WALK_K = int(os.environ.get("KT_CHAIN_PF_K", "2"))
_P = int(os.environ.get("KT_CHAIN_PF_P", "2"))
# Decode-shaped batches only. Without this the walk also runs on every PREFILL
# chunk -- 2048 tokens through an extra MoE per layer, for a prefetch that means
# nothing there. Reading .shape is static metadata, not a tensor value, so the
# guard is capture-safe.
_TMAX = int(os.environ.get("KT_CHAIN_PF_TMAX", "8"))
# Cost decomposition (Stage H2). The walk measured 18.2 ms/step per walked layer
# and NOTHING about that number is attributed yet, so it gets taken apart one
# component at a time rather than explained. Cumulative:
#   1 walk skipped entirely (hook + target prediction only)
#   2 + the walked layer's gate and topk scoring path
#   3 + the resident substitution, id remap and slot masking
#   4 + the GPU expert kernel itself
#   5 + the shared expert
#   6 + the TP all-reduce  (= the real thing)
# Run the ladder with the prefetch's EFFECT off (GATHER=0 ROUTE=0 CPUSKIP=0) so
# that a stage's degraded prediction cannot feed back into coverage and change
# the step time. Then every difference between rows is walk COST and nothing
# else.
_STAGE = int(os.environ.get("KT_CHAIN_PF_STAGE", "6"))
# How the walk forces its routing onto GPU-resident experts.
#   search  the original: substitute each non-resident slot with the best
#           remaining resident expert by router score. ~20 kernels/layer,
#           measured at 6.55 ms/step in Stage H2.
#   drop_raw  keep the slots that are already resident, drop the rest, and do
#           NOT renormalise. ~3 kernels/layer, 1.55 ms/step. Whole-layer
#           coverage 75.0 / 67.4 / 65.0 / 66.4 at d=1..4 against search's
#           77.0 / 70.0 / 66.0 / 67.6 -- ~2 points for 5 ms less. This is the
#           cheap mode to use.
#   drop    identical, but renormalised: 71.4 / 62.3 / 59.7 / 61.7, i.e. BELOW
#           running no routed experts at all and below not walking. The single
#           division is the whole difference. Kept for reproducibility.
_SUB = os.environ.get("KT_CHAIN_PF_SUB", "search").strip().lower()

_MOE: Dict[int, object] = {}
_DL: Dict[int, object] = {}
_WARNED = set()


def register_moe(layer_id: int, moe) -> None:
    """Called from DeepseekV2MoE.__init__ -- see the construction rule above."""
    _MOE[int(layer_id)] = moe


def register_layer(layer_id: int, dlayer) -> None:
    """Called from DeepseekV2DecoderLayer.__init__."""
    _DL[int(layer_id)] = dlayer


def _norm(lid: int, v: torch.Tensor) -> Optional[torch.Tensor]:
    dl = _DL.get(lid)
    ln = getattr(dl, "post_attention_layernorm", None) if dl is not None else None
    if ln is None:
        return None
    out = ln(v)
    return out[0] if isinstance(out, tuple) else out


def _resident_ids(topk_output, logits, mask):
    """Genuine slots where already resident, best resident substitute elsewhere.

    Returns (ids, weights) -- weights renormalised so the approximate update is
    a proper convex combination and the residual stream keeps its scale.
    """
    w = topk_output.topk_weights
    ids = topk_output.topk_ids
    T, E = w.shape
    allow = mask.to(torch.bool)
    keep = allow[ids.long()]
    scores = logits.float()
    neg = torch.finfo(scores.dtype).min
    s = torch.where(allow.view(1, -1).expand(T, -1), scores, neg).clone()
    s.scatter_(1, ids.long(), neg)           # never refill with an existing slot
    fill_s, fill_ids = torch.topk(s, E, dim=-1)
    drop = ~keep
    w_drop = torch.where(drop, w, torch.full_like(w, neg))
    order_d = torch.argsort(w_drop, dim=-1, descending=True)
    drank = torch.empty_like(order_d)
    drank.scatter_(1, order_d, torch.arange(E, device=w.device).expand(T, E))
    new_ids = torch.where(keep, ids,
                          torch.gather(fill_ids, 1, drank).to(ids.dtype))
    new_w = torch.where(keep, w,
                        torch.sigmoid(torch.gather(fill_s, 1, drank)).to(w.dtype))
    return new_ids, new_w


def _approx_update(moe, h: torch.Tensor) -> Optional[torch.Tensor]:
    """Layer `moe`'s contribution to the residual stream, GPU-resident only.

    Mirrors forward_normal's assembly (routed x routed_scaling_factor + shared,
    then the TP all-reduce) but reaches past the kt wrapper straight to the GPU
    expert kernel, so no CPU batch is ever submitted.
    """
    from sglang.srt.layers.moe.kt_ep_wrapper import mask_and_remap_expert_ids
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    qm = getattr(moe.experts, "quant_method", None)
    mask = getattr(qm, "gpu_experts_mask_cuda", None)
    gpu_method = getattr(qm, "gpu_method", None)
    if mask is None or gpu_method is None:
        return None
    if _STAGE <= 1:
        return torch.zeros_like(h)

    logits = moe.gate(h, None)
    if isinstance(logits, tuple):
        logits = logits[0]
    tk = moe.topk(h, logits)
    if _STAGE <= 2:
        return torch.zeros_like(h)
    if _SUB in ("drop", "drop_raw"):
        # THE CHEAP PATH. Stage H2 measured the substitution SEARCH at 6.55
        # ms/step -- 87 us per layer for ~20 tiny kernels (a topk across all 256
        # experts, two argsorts, gathers, scatters), none of which is
        # arithmetic. But the walk does not need the "nearest equivalent"
        # expert; it needs SOME resident-only update. And
        # mask_and_remap_expert_ids ALREADY writes -1 -- the kernel's skip
        # sentinel -- into every non-resident slot. So dropping them is free,
        # and all that is left is to zero those slots' weights and renormalise:
        # one gather, one where, one div, against twenty.
        #
        # No _WALK_K limit here, deliberately: enforcing one needs another
        # argsort, which is the cost being removed. The live-slot count is then
        # whatever the router happened to put on resident experts (~3 of 8 at
        # 100/256 resident) rather than a fixed 2, so this trades a little more
        # expert GEMM for a lot less dispatch -- and Stage H2 measured GEMM as
        # the cheap direction (2 -> 8 slots was only 3.4 ms).
        ids = tk.topk_ids
        remapped = mask_and_remap_expert_ids(ids, mask,
                                             qm.logical_to_gpu_index_cuda)
        w = torch.where(mask.to(torch.bool)[ids.long()], tk.topk_weights,
                        torch.zeros_like(tk.topk_weights))
    else:
        ids, w = _resident_ids(tk, logits, mask)
        remapped = mask_and_remap_expert_ids(ids, mask,
                                             qm.logical_to_gpu_index_cuda)
        E = w.shape[1]
        if _WALK_K < E:
            # Keep only the top _WALK_K slots. -1 AFTER the remap is the
            # kernel's "skip this slot" sentinel; writing it before the remap
            # would index the residency mask at -1 instead. Branch-free, static
            # shape.
            order = torch.argsort(w, dim=-1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order,
                          torch.arange(E, device=w.device).expand(w.shape[0], E))
            live = rank < _WALK_K
            remapped = torch.where(live, remapped, torch.full_like(remapped, -1))
            w = torch.where(live, w, torch.zeros_like(w))
    if _SUB != "drop_raw":
        # MEASURED, not assumed: renormalising after a DROP is what makes the
        # drop bad. Zeroing non-resident slots leaves the routed weights summing
        # to less than one, and rescaling them back to one hands the two or three
        # survivors the full routed magnitude -- an update of about the right
        # size pointing the wrong way. Leaving the sum short instead is the
        # honest statement that the missing experts contributed nothing.
        #
        # Whole-layer coverage at d=1..4 (608 steps, one boot, same trajectory):
        #   drop + renormalise  71.4 / 62.3 / 59.7 / 61.7   <- below doing nothing
        #   drop, no renorm     75.0 / 67.4 / 65.0 / 66.4   <- above shared-only
        #   search              77.0 / 70.0 / 66.0 / 67.6   <- costs 5 ms more
        # Removing this one division recovers 3.6-5.3 points for free.
        #
        # `search` still needs it: with _WALK_K < E the top-K mask zeroes real
        # slots, and there the weight genuinely does belong to the survivors.
        w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-9)
    if _STAGE <= 3:
        return torch.zeros_like(h)

    out = gpu_method.apply(
        moe.experts,
        StandardDispatchOutput(
            hidden_states=h, hidden_states_scale=None,
            topk_output=tk._replace(topk_ids=remapped, topk_weights=w),
        ),
    ).hidden_states * moe.routed_scaling_factor

    if _STAGE <= 4:
        return out
    shared = moe._forward_shared_experts(h, None)
    if shared is not None:
        out = out + shared
    if _STAGE <= 5:
        return out
    if getattr(moe, "tp_size", 1) > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


def emit(lid: int, hidden_states, residual, is_nextn: bool = False) -> None:
    """Layer `lid` is done: walk forward and fire the fetch for lid + depth."""
    if not ENABLED or hidden_states is None or residual is None:
        return
    if is_nextn:
        # The NEXTN/MTP draft is captured into its own single-layer graph, so a
        # fetch issued from it targets a layer whose rejoining wait_event never
        # gets captured -- cudaErrorStreamCaptureUnjoined at capture end. Its
        # predictions are worthless here regardless: the target model re-runs
        # every layer immediately afterwards.
        return
    T = int(hidden_states.shape[0])
    if T == 0 or T > _TMAX:
        return
    lid = int(lid)
    tgt = lid + _DEPTH
    tmoe = _MOE.get(tgt)
    if tmoe is None or getattr(tmoe, "is_hash", False):
        return

    with torch.no_grad():
        # Step 0 is free and exact: this IS layer L's real output stream.
        state = hidden_states + residual
        for s in range(1, _DEPTH):
            smoe = _MOE.get(lid + s)
            if smoe is None or getattr(smoe, "is_hash", False):
                # A dense layer or a hash-routed one in the way. Hash routing
                # needs input_ids, not the hidden state, so it cannot be walked.
                return
            hh = _norm(lid + s, state)
            if hh is None:
                return
            upd = _approx_update(smoe, hh)
            if upd is None:
                return
            state = state + upd

        x = _norm(tgt, state)
        if x is None:
            return
        logits = tmoe.gate(x, None)
        if isinstance(logits, tuple):
            logits = logits[0]
        out = tmoe.topk(x, logits)
        p = min(_P, out.topk_weights.shape[1])
        top = torch.gather(out.topk_ids.long(), 1,
                           torch.topk(out.topk_weights, p, dim=-1).indices)

    from sglang.srt.layers.moe.kt_ep_wrapper import (
        _KT_PREFETCH_ENABLED, _KT_PREFETCH_LAYERS,
    )
    if not _KT_PREFETCH_ENABLED:
        return
    m = _KT_PREFETCH_LAYERS.get(tgt)
    if m is not None:
        m.pf_issue(top)
