"""Does propagating the hidden state through resident-only experts predict
future routing better than just showing the current hidden state to a future
layer's router?

THE TWO CANDIDATE MECHANISMS
----------------------------
Both answer the same question -- "which experts will layer L+d want?" -- from
what is known at layer L, and both have to answer it early enough that a
transfer can be issued and completed before layer L+d's MoE runs.

  DIRECT (shipped, EXPERT_PREFETCH_PLAN.md Stage D).  Take layer L's MoE input
  h_L and feed it straight to layer L+d's gate.  One 6144x256 GEMM.  It ignores
  the fact that d whole layers will modify the residual stream in between, so
  its error is exactly that drift; measured 75-77% top-2 recall at d=1, decaying
  to 43% at d=8.

  CHAIN (this file's subject).  While the CPU is computing layer L's
  non-resident experts, the GPU is idle.  Spend that window running an
  APPROXIMATE layer L: route normally, but substitute every non-resident expert
  with the best GPU-resident one, so the update needs nothing the GPU does not
  already hold.  Add it to the residual stream, renormalise, run layer L+1's
  gate -- and repeat, walking the approximation forward a few layers.  The
  prediction is then made from a hidden state that has actually moved, instead
  of a stale one.

The chain costs a real MoE forward per layer of lookahead, so it only earns its
place if it is MORE ACCURATE.  That is a purely statistical question and it is
answered here, before anything is built on it.

WHAT IS MEASURED, AND AGAINST WHAT
----------------------------------
At each lookahead position d, the fraction of layer L+d's genuine top-K experts
that the prediction named -- the user's "percentage of correct experts".  Recall
is the headline, but it is NOT what a prefetcher is paid in: a fetch only
removes a layer's CPU round-trip when EVERY non-resident expert that layer needs
arrived, so whole-layer coverage is accumulated directly as well (rather than
approximated as recall**D, which Stage D2 showed is pessimistic because a
layer's misses are positively correlated).

ARMS.  Each isolates one term, so a difference can be attributed:

  direct        h_L -> gate_{L+d}.  The shipped predictor.  Baseline.
  renorm        post_ln_{L+d}(r_L) -> gate_{L+d}.  Same stale stream, but
                renormalised with the target's own norm.  Splits "the stream
                drifted" from "the scale was wrong", which the sigmoid/bias/
                group scoring path is not invariant to.
  post          renorm, but from the stream AFTER layer L's real MoE update.
                Free (the update is r_next - r_L, a subtraction).  At d=1 this
                is "know layer L exactly, skip only L+1's attention", so it
                upper-bounds any chain whose first step is layer L.
  chain         The proposal: iterate resident-substituted MoE updates forward
                through L, L+1, ..., L+d-1.
  chain_shared  The same walk with ONLY the shared expert -- no routed experts
                at all.  Costs nothing to run.  If it matches `chain`, the
                routed part of the update is not what carries the signal and
                the mechanism gets an order of magnitude cheaper.
  chain_exact   Opt-in (KT_CHAIN_EXACT=1).  The walk with genuine full routing,
                CPU experts included.  Unshippable by construction -- it needs
                the very results the prefetch exists to avoid waiting for -- but
                it separates "the resident-only approximation is too coarse"
                from "skipping attention is what breaks it".

DEPTH 0 IS THE WIRING SELF-CHECK, not a formality.  At d=0 no update has been
applied, so `direct` scores h_L against layer L's own genuine top-K and MUST
read 100.0%; `renorm` scores post_ln_L(r_L), which must ALSO read 100.0% and is
therefore a direct test that the residual stream this file stitched together is
the one the model actually used.  A low number at d>=1 cannot be attributed to
drift unless both of those are exact.

WHY EAGER.  Everything here runs arbitrary Python per layer per step, so it
cannot be captured; under CUDA graphs it would silently collect nothing during
decode (the known KT_DUMP_TOPK trap).  Run with DISABLE_CUDA_GRAPH=1.  Decode is
slow, which does not matter -- this measures accuracy, not throughput -- and
KT_CHAIN_STRIDE keeps the cost bounded by starting a chain from only every Nth
layer.  Routing statistics are a property of the trajectory, not of how the
kernels were launched, so an eager run measures the same quantity a captured one
would.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

ENABLED = os.environ.get("KT_CHAIN_PRED") == "1"

_DEPTH = int(os.environ.get("KT_CHAIN_DEPTH", "4"))
# Which lookahead positions to SCORE.  The chain has to walk every intermediate
# layer regardless -- there is no way to reach L+16 without passing through
# L+15 -- but scoring is a gate GEMM per arm per depth, so a sparse ladder
# (1,2,3,4,6,8,12,16) covers the decay curve at a fraction of the cost of
# scoring all sixteen.
_DEPTHS = tuple(sorted({int(x) for x in
                        os.environ.get("KT_CHAIN_DEPTHS", "").split(",")
                        if x.strip().isdigit()} - {0}))
if not _DEPTHS:
    _DEPTHS = tuple(range(1, _DEPTH + 1))
_DMAX = max(_DEPTHS)
_K = int(os.environ.get("KT_CHAIN_K", "2"))
_TMAX = int(os.environ.get("KT_CHAIN_TMAX", "8"))
_STRIDE = max(1, int(os.environ.get("KT_CHAIN_STRIDE", "6")))
_P = tuple(sorted({int(x) for x in os.environ.get("KT_CHAIN_P", "2,3,4").split(",")
                   if x.strip().isdigit()} - {0})) or (2,)
_PMAX = max(_P)
_DUMP_EVERY = int(os.environ.get("KT_CHAIN_DUMP_EVERY", "16"))
_OUT = os.environ.get("KT_CHAIN_OUT", "bench/profile_out/chain_predict.json")
_EXACT = os.environ.get("KT_CHAIN_EXACT") == "1"
# How many routed slots the APPROXIMATE forward keeps.  8 (all of them, each
# forced onto a resident expert) is the most faithful resident-only update; 2 is
# ~4x less expert GEMM and is what the capture-safe mechanism would rather run.
# Dropped slots keep their (resident) ids and lose their weight, so this changes
# the ACCURACY exactly as the cheap kernel would while the instrument's own cost
# is unchanged -- accuracy is the only thing measured here.
_WALK_K = int(os.environ.get("KT_CHAIN_WALK_K", "8"))

_ARMS: Tuple[str, ...] = tuple(
    a for a in os.environ.get(
        "KT_CHAIN_ARMS", "direct,renorm,post,chain,chain_drop,chain_shared"
    ).split(",") if a.strip()
)
if _EXACT and "chain_exact" not in _ARMS:
    _ARMS = _ARMS + ("chain_exact",)
# Arms that carry a residual-stream state forward across layers.  `renorm` and
# `post` deliberately do not: they are the "no propagation" controls.
_WALKING = tuple(a for a in _ARMS if a.startswith("chain"))

HIT, TOT, FULL, ACTIVE, NEED, COV, SET, CALLS, BFULL, BCOV = range(10)
_COLS = 10

# layer_id -> module, populated on first sight.  Registration happens before the
# token-count guard so a prefill pass (which is never scored) still wires
# everything up; otherwise the first decode step would find every future layer
# missing and silently emit nothing.
_MOE: Dict[int, object] = {}
_DL: Dict[int, object] = {}
_H: Dict[int, torch.Tensor] = {}          # layer -> its MoE input this pass
# target layer -> {(arm, depth): predicted ids}.  A plain dict, popped on score,
# so a stale entry can never be scored -- there is no validity flag to get wrong.
_PENDING: Dict[int, Dict[Tuple[str, int], torch.Tensor]] = {}
# layer -> this pass's genuine-routing scoring context, so the depth-0
# self-check can be scored in the same pass rather than through the inbox.
_CTX: Dict[int, dict] = {}
# layer -> the set of non-resident experts it needed on the PREVIOUS token.
# Drives both the `persist` arm and the blend column on every other arm.
_PREV: Dict[int, torch.Tensor] = {}
_STATS: Dict[Tuple[str, int, int], torch.Tensor] = {}

_STEP = 0
_FIRST_LID: Optional[int] = None
_RANK = 0
_NORM_ERR: Dict[str, float] = {}
_FAILED: Dict[str, str] = {}


def _log(msg: str) -> None:
    print(f"[kt-chain] {msg}", flush=True)


def _stat(arm: str, d: int, p: int, dev) -> torch.Tensor:
    key = (arm, d, p)
    s = _STATS.get(key)
    if s is None:
        s = torch.zeros(_COLS, dtype=torch.float64, device=dev)
        _STATS[key] = s
    return s


def _norm(lid: int, v: torch.Tensor) -> Optional[torch.Tensor]:
    """Layer `lid`'s own post_attention_layernorm -- the MoE-input norm."""
    dl = _DL.get(lid)
    ln = getattr(dl, "post_attention_layernorm", None) if dl is not None else None
    if ln is None:
        return None
    out = ln(v)
    return out[0] if isinstance(out, tuple) else out


def _topP(moe, h: torch.Tensor, p: int) -> torch.Tensor:
    """Target layer's OWN gate and topk on hidden state `h`.

    Using the target's `topk` module rather than raw gate logits is
    load-bearing: Stage D found the sigmoid/bias/group scoring path accounts for
    most of the apparent prediction error otherwise (depth 0 read 71%, not 100%).
    """
    logits = moe.gate(h, None)
    if isinstance(logits, tuple):
        logits = logits[0]
    out = moe.topk(h, logits)
    p = min(p, out.topk_weights.shape[1])
    return torch.gather(out.topk_ids.long(), 1,
                        torch.topk(out.topk_weights, p, dim=-1).indices)


def _restrict_resident(topk_output, logits: torch.Tensor,
                       mask: torch.Tensor) -> None:
    """Force every routed slot onto a GPU-resident expert, in place.

    Keeps the genuine choice wherever it is already resident and substitutes the
    rest with the best resident expert not already selected, by router score,
    highest-weight dropped slot first -- the same "nearest equivalent" rule the
    shipped substitution tiers use.  Weights are renormalised, so the update is
    a proper convex combination and the residual stream stays in scale.
    """
    w = topk_output.topk_weights
    ids = topk_output.topk_ids
    T, E = w.shape
    allow = mask.to(torch.bool)
    keep = allow[ids.long()]
    scores = logits.float()
    neg = torch.finfo(scores.dtype).min
    s = torch.where(allow.view(1, -1).expand(T, -1), scores, neg).clone()
    s.scatter_(1, ids.long(), neg)          # never refill with an existing slot
    fill_s, fill_ids = torch.topk(s, E, dim=-1)
    fill_w = torch.sigmoid(fill_s)
    drop = ~keep
    w_drop = torch.where(drop, w, torch.full_like(w, neg))
    order_d = torch.argsort(w_drop, dim=-1, descending=True)
    drank = torch.empty_like(order_d)
    drank.scatter_(1, order_d, torch.arange(E, device=w.device).expand(T, E))
    sel_ids = torch.gather(fill_ids, 1, drank)
    sel_w = torch.gather(fill_w, 1, drank)
    new_w = torch.where(keep, w, sel_w.to(w.dtype))
    new_w = new_w / torch.clamp(new_w.sum(dim=-1, keepdim=True), min=1e-9)
    topk_output.topk_ids.copy_(torch.where(keep, ids, sel_ids.to(ids.dtype)))
    topk_output.topk_weights.copy_(new_w)


def _moe_update(moe, h: torch.Tensor, kind: str) -> Optional[torch.Tensor]:
    """The contribution layer `moe` would add to the residual stream for `h`.

    Mirrors forward_normal's assembly exactly: routed output scaled by
    routed_scaling_factor (the kt path applies it outside the kernel), plus the
    shared expert, then the TP all-reduce.  `kind` selects how much of the real
    routing is honoured.
    """
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    shared = moe._forward_shared_experts(h, None)
    if kind == "shared":
        out = shared
        if out is None:
            return None
    else:
        logits = moe.gate(h, None)
        if isinstance(logits, tuple):
            logits = logits[0]
        tk = moe.topk(h, logits)
        if kind == "resident_drop":
            # The CHEAP substitution: keep the routed slots that are already
            # GPU-resident, drop the rest, renormalise. No search over the 256.
            # Zeroing the weight is accuracy-equivalent to the mechanism's -1
            # skip sentinel -- the dropped expert contributes nothing either way
            # -- and the instrument only measures accuracy, never cost.
            #
            # This is a REAL fidelity question, not a free lunch. `resident`
            # puts a stand-in with similar router mass where the missing expert
            # was; `resident_drop` removes it and rescales what is left. Which
            # lands closer to the true hidden state decides whether the 5.0 ms
            # the drop saves is worth having.
            qm = getattr(moe.experts, "quant_method", None)
            mask = getattr(qm, "gpu_experts_mask_cuda", None)
            if mask is None:
                return None
            w = tk.topk_weights
            keep = mask.to(torch.bool)[tk.topk_ids.long()]
            nw = torch.where(keep, w, torch.zeros_like(w))
            tk.topk_weights.copy_(
                nw / torch.clamp(nw.sum(dim=-1, keepdim=True), min=1e-9))
        elif kind == "resident_drop_raw":
            # Drop, but do NOT renormalise. `resident_drop` measured WORSE than
            # `shared`, which runs no routed experts at all -- so an update of
            # the wrong magnitude may be costing more than a missing one. The
            # renormalisation is the only thing that inflates magnitude: it
            # rescales two or three surviving experts up to carry the full
            # routed weight. This arm removes exactly that and nothing else, so
            # the two together say whether the renormalise is the culprit.
            qm = getattr(moe.experts, "quant_method", None)
            mask = getattr(qm, "gpu_experts_mask_cuda", None)
            if mask is None:
                return None
            w = tk.topk_weights
            keep = mask.to(torch.bool)[tk.topk_ids.long()]
            tk.topk_weights.copy_(torch.where(keep, w, torch.zeros_like(w)))
        elif kind == "resident":
            qm = getattr(moe.experts, "quant_method", None)
            mask = getattr(qm, "gpu_experts_mask_cuda", None)
            if mask is None:
                return None
            _restrict_resident(tk, logits, mask)
            if _WALK_K < tk.topk_weights.shape[1]:
                w = tk.topk_weights
                E = w.shape[1]
                order = torch.argsort(w, dim=-1, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(1, order,
                              torch.arange(E, device=w.device).expand(w.shape[0], E))
                nw = torch.where(rank < _WALK_K, w, torch.zeros_like(w))
                tk.topk_weights.copy_(
                    nw / torch.clamp(nw.sum(dim=-1, keepdim=True), min=1e-9))
        out = moe.experts(h, tk) * moe.routed_scaling_factor
        if shared is not None:
            out = out + shared
    if getattr(moe, "tp_size", 1) > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


def _accum(arm: str, d: int, pred_all: torch.Tensor, ctx: dict) -> None:
    """Score one prediction against a layer's genuine routing."""
    true_ids, need_b, n_need = ctx["true"], ctx["need_b"], ctx["n_need"]
    pred_all = pred_all[: ctx["T"]]
    eq = true_ids.unsqueeze(-1) == pred_all.unsqueeze(1)
    pset = torch.zeros(ctx["n_exp"], device=true_ids.device, dtype=torch.float32)
    done = 0
    for P in _P:
        pp = min(P, pred_all.shape[1])
        # Set overlap on the TRUE side: with a superset (P > k) precision and
        # recall diverge, and recall is what a prefetcher is graded on.
        hit = eq[:, :, :pp].any(-1).sum().to(torch.float32)
        if pp > done:
            new = pred_all[:, done:pp].reshape(-1)
            pset.index_add_(0, new, torch.ones_like(new, dtype=torch.float32))
            done = pp
        psb = pset > 0
        cov = (need_b & psb).sum().to(torch.float32)
        full = ((cov >= n_need) & (n_need > 0)).to(torch.float32)
        # BLEND: union the prediction with what THIS layer needed on the
        # previous token. The two signals are near independent -- a lookahead is
        # reading the residual stream, persistence is reading this layer's own
        # recent history -- so their union can cover layers neither covers
        # alone, and it costs no compute at all, only the extra bytes.
        bl = psb | ctx["prev_b"]
        bcov = (need_b & bl).sum().to(torch.float32)
        bfull = ((bcov >= n_need) & (n_need > 0)).to(torch.float32)
        _stat(arm, d, P, true_ids.device).add_(torch.stack([
            hit,
            hit.new_full((), float(ctx["T"] * ctx["k"])),
            full, ctx["active"], n_need, cov,
            (psb & ctx["nonres"]).sum().to(torch.float32),
            hit.new_ones(()),
            bfull, bcov,
        ]).to(torch.float64))


def score(moe, hidden_states, router_logits, topk_output, quant_method) -> None:
    """Score every prediction aimed at this layer, then stash its MoE input.

    Called from forward_normal BEFORE any substitution mutates the routing, so
    the target is the router's genuine choice.
    """
    global _STEP, _FIRST_LID, _RANK
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    lid = int(getattr(moe, "layer_id", -1))
    if lid < 0:
        return
    _MOE[lid] = moe
    pend = _PENDING.pop(lid, None)
    _CTX.pop(lid, None)
    T = int(hidden_states.shape[0])
    if T == 0 or T > _TMAX:
        _H.pop(lid, None)
        return
    _H[lid] = hidden_states.detach()

    if _FIRST_LID is None:
        _FIRST_LID = lid
        try:
            _RANK = int(getattr(quant_method, "tp_rank", 0) or 0)
        except Exception:
            _RANK = 0
    if lid == _FIRST_LID:
        _STEP += 1
        if _DUMP_EVERY > 0 and _STEP % _DUMP_EVERY == 0:
            dump()

    mask = getattr(quant_method, "gpu_experts_mask_cuda", None)
    if mask is None or router_logits is None:
        return

    with torch.no_grad():
        w = topk_output.topk_weights
        ids = topk_output.topk_ids.long()
        k = min(_K, w.shape[1])
        true_ids = torch.gather(ids, 1, torch.topk(w, k, dim=-1).indices)
        n_exp = int(router_logits.shape[1])
        allow = mask.to(torch.bool)
        # Only the non-resident members of the genuine top-K cost a CPU
        # round-trip; scoring coverage against all of them would flatter it.
        need = torch.zeros(n_exp, device=w.device, dtype=torch.float32)
        need.index_add_(0, true_ids.reshape(-1),
                        (~allow[true_ids]).float().reshape(-1))
        need_b = need > 0
        n_need = need_b.sum().to(torch.float32)
        prev_b = _PREV.get(lid)
        if prev_b is None:
            prev_b = torch.zeros(n_exp, device=w.device, dtype=torch.bool)
        ctx = {
            "true": true_ids, "need_b": need_b, "n_need": n_need,
            "active": (n_need > 0).to(torch.float32), "nonres": ~allow,
            "n_exp": n_exp, "T": T, "k": k, "prev_b": prev_b,
        }
        # Kept for THIS layer's own emit, later in the same pass: the depth-0
        # self-check must be scored IMMEDIATELY. Routing it through _PENDING
        # like the lookahead arms leaves it sitting until the NEXT forward pass,
        # where it silently stops being a self-check and becomes a persistence
        # measurement -- which is exactly what happened first time round. It
        # read 26.5%, i.e. Stage D's independently measured 27.1% persistence,
        # instead of the 100% a wiring check has to read.
        _CTX[lid] = ctx
        for (arm, d), pred_all in (pend or {}).items():
            _accum(arm, d, pred_all, ctx)
        # AFTER scoring, so every arm above was blended against the PREVIOUS
        # token's demand rather than this one's.
        _PREV[lid] = need_b


def emit(lid: int, hidden_states, residual, dlayer) -> None:
    """Layer `lid` has finished.  Walk the approximations forward and predict.

    `hidden_states + residual` is the stream entering layer lid+1 minus that
    layer's attention -- exactly what a real prefetcher would hold at this
    instant, which is why every arm is scored from this one point.  The only
    caveat is that the shipped predictor fires slightly earlier (at lid's MoE
    INPUT, buying lid's own MoE as extra shadow); `direct` here uses the
    identical signal h_L, so the accuracy comparison is unaffected -- only the
    transfer's lead time would differ.
    """
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    lid = int(lid)
    _DL[lid] = dlayer
    if (lid % _STRIDE) != 0 or residual is None or hidden_states is None:
        return
    moe = _MOE.get(lid)
    h_L = _H.get(lid)
    r_L = getattr(dlayer, "_kt_chain_resid", None)
    if moe is None or h_L is None or r_L is None:
        return
    T = int(h_L.shape[0])
    if T == 0 or T > _TMAX:
        return

    with torch.no_grad():
        r_L = r_L.detach()
        r_next = (hidden_states + residual).detach()      # exact update, free

        # d = 0: the wiring self-check, scored NOW against this layer's own
        # genuine routing.  `direct` tests the scoring path; `renorm` tests that
        # r_L really is the stream the model normalised.  Both must read 100.0%
        # or nothing else in the table means anything.
        ctx = _CTX.get(lid)
        # Layer L's OWN top-P on its own input -- exact by construction, since
        # no drift has happened yet.  Used three ways: the depth-0 self-check,
        # the `persist` arm, and the `prevlayer` carry below.
        own = _predict(lid, h_L)
        if ctx is not None:
            p = own
            if p is not None:
                _accum("direct", 0, p, ctx)
                # Free baseline, same tensor, no extra work: what this layer
                # routed to on THIS step, scored against what it routes to on
                # the next.  That is the persistence predictor -- "fetch what we
                # needed last time" -- which Stage D measured at 27.1% and which
                # any lookahead has to beat to be worth its cost.
                _PENDING.setdefault(lid, {})[("persist", 1)] = p
            x = _norm(lid, r_L)
            if x is not None:
                _check_norm(lid, x, h_L)
                q = _predict(lid, x)
                if q is not None:
                    _accum("renorm", 0, q, ctx)

        # `chain_post` starts from the stream AFTER layer L's real MoE update
        # -- which is free at this hook -- and only approximates the layers
        # after it.  That is exactly what the capture-safe mechanism does, and
        # at d=1 it degenerates to `post`, giving a built-in cross-check.
        state = {a: (r_next if a == "chain_post" else r_L) for a in _WALKING}
        for d in range(1, _DMAX + 1):
            src = lid + d - 1
            tgt = lid + d
            smoe = _MOE.get(src)
            for arm in list(state):
                if smoe is None:
                    continue
                # `chain_post` already carries layer L's REAL update in its
                # state, so the d == 1 iteration -- whose source layer IS L --
                # must not add an approximate one on top.  Its walk begins at
                # L+1.  (Without this the arm double-applies layer L and reads
                # WORSE than `post`, which is the defect TODO item 6 records.)
                if arm == "chain_post" and d == 1:
                    continue
                # d == 1 reuses h_L; deeper steps must renormalise the
                # approximation with the source layer's own norm.
                hh = h_L if d == 1 else _norm(src, state[arm])
                if hh is None:
                    continue
                kind = {"chain": "resident", "chain_post": "resident",
                        "chain_drop": "resident_drop",
                        "chain_drop_raw": "resident_drop_raw",
                        "chain_shared": "shared",
                        "chain_exact": "exact"}[arm]
                try:
                    upd = _moe_update(smoe, hh, kind)
                except Exception as exc:                  # pragma: no cover
                    if arm not in _FAILED:
                        _FAILED[arm] = repr(exc)
                        _log(f"arm {arm} disabled at layer {src}: {exc!r}")
                    state.pop(arm, None)
                    continue
                if upd is not None:
                    state[arm] = state[arm] + upd

            # The walk above is unconditional -- L+16 is only reachable through
            # L+15 -- but only the ladder's rungs are scored.
            if d not in _DEPTHS or _MOE.get(tgt) is None:
                continue
            # CROSS-LAYER CARRY: predict layer L+d's experts as the ones layer L
            # itself just used.  No router pass at all -- it reuses the tensor
            # the depth-0 check already computed.  Distinct from `persist`, which
            # carries a layer's own choice forward in TIME; this carries a
            # neighbour's choice forward in DEPTH, and adjacent MoE layers may
            # well agree with each other more than adjacent tokens do.
            if own is not None:
                _PENDING.setdefault(tgt, {})[("prevlayer", d)] = own
            if "direct" in _ARMS:
                _emit_one("direct", d, tgt, h_L)
            for arm, v in (("renorm", r_L), ("post", r_next)):
                if arm in _ARMS:
                    x = _norm(tgt, v)
                    if x is not None:
                        _emit_one(arm, d, tgt, x)
            for arm, v in state.items():
                x = _norm(tgt, v)
                if x is not None:
                    _emit_one(arm, d, tgt, x)


def _predict(tgt: int, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Run layer `tgt`'s gate on hidden state `x`; None if it cannot be done."""
    moe = _MOE.get(tgt)
    if moe is None or getattr(moe, "is_hash", False):
        return None     # hash routing needs input_ids, not the hidden state
    try:
        return _topP(moe, x, _PMAX)
    except Exception as exc:                              # pragma: no cover
        if "predict" not in _FAILED:
            _FAILED["predict"] = repr(exc)
            _log(f"gate on layer {tgt} failed: {exc!r}")
        return None


def _emit_one(arm: str, d: int, tgt: int, x: torch.Tensor) -> None:
    top = _predict(tgt, x)
    if top is not None:
        _PENDING.setdefault(tgt, {})[(arm, d)] = top


def _check_norm(lid: int, x: torch.Tensor, h: torch.Tensor) -> None:
    """post_ln(r_L) must reproduce the MoE input the layer actually used."""
    if "resid" in _NORM_ERR:
        return
    num = (x - h).abs().max().item()
    den = h.abs().max().item() + 1e-9
    _NORM_ERR["resid"] = num / den
    _log(f"residual reconstruction rel-err = {num/den:.3e} (layer {lid})")


def dump() -> None:
    if not _STATS:
        return
    rows = []
    for (arm, d, p), s in sorted(_STATS.items()):
        v = s.to("cpu").tolist()
        rows.append({
            "arm": arm, "depth": d, "P": p,
            "hit": v[HIT], "tot": v[TOT], "full": v[FULL], "active": v[ACTIVE],
            "need": v[NEED], "cov": v[COV], "set": v[SET], "calls": v[CALLS],
            "bfull": v[BFULL], "bcov": v[BCOV],
        })
    out = {
        "version": 2, "steps": _STEP, "rank": _RANK, "K": _K,
        "depths": list(_DEPTHS), "stride": _STRIDE, "P": list(_P),
        "arms": list(_ARMS), "resid_rel_err": _NORM_ERR.get("resid"),
        "failed": dict(_FAILED), "rows": rows,
    }
    path = _OUT if _OUT.endswith(".json") else _OUT + ".json"
    path = path[:-5] + f".rank{_RANK}.json"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, path)
    except Exception as exc:                              # pragma: no cover
        _log(f"dump to {path} failed: {exc!r}")
