#!/usr/bin/env python3
"""The split policy: given measured constants, how many experts should stream?

This module is deliberately pure Python with no torch and no CUDA. It is the
one place that decides, and it is shared by three callers that must never
disagree: the calibration harness (which reports what the profile implies), the
unit tests, and the serving path (which turns `k` into an actual transfer).

The whole design is machine-relative. Nothing here assumes PCIe is slow or that
the CPU is fast -- those are inputs. On this box (2xH100, PCIe Gen5, 56 GB/s per
card, a 72-core CPU holding 152 of 256 experts) the answer at the default tier
is "stream nothing". On a machine with a fast host-to-device link (GH200's
900 GB/s C2C is ~16x this box's per-card PCIe) or a slower CPU, the same
formula returns a much larger `k` from the same code path. That is the point:
the policy is calibrated, not tuned.

The model, per layer per step:

    cpu_ms(k)   = c_per_unit * (U - reuse * k)     work left on the CPU
    link_ms(k)  = k * bytes_per_expert / bw       time to ship k experts
    extra_ms(k) = k * bytes_per_expert * contention_ms_per_gb

CPU and link overlap (kt already submits the CPU job, runs GPU work, then
syncs), so a layer costs `max(cpu_ms, link_ms) + extra_ms`. Sweeping k and
taking the minimum handles both the interior optimum and the two corners --
including k = D, which removes the CPU from the layer entirely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerCost:
    k: int              # distinct experts to stream
    layer_ms: float     # predicted per-layer cost at this k
    cpu_ms: float
    link_ms: float
    extra_ms: float


@dataclass(frozen=True)
class Constants:
    """Everything the decision needs, all of it measured per host."""

    cpu_ms_per_layer: float       # CPU expert pole for this tier, per layer
    units_per_layer: float        # U: expert-token work items per layer per step
    distinct_per_layer: float     # D: distinct experts per layer per step
    bytes_per_expert_gb: float    # per card, TP-sharded
    link_gbs: float               # achieved H2D per card under load
    contention_ms_per_gb: float   # decode ms paid per GB streamed
    gpu_ms_per_expert: float = 0.0  # GPU MoE marginal cost per streamed expert
    fixed_ms_per_layer: float = 0.0  # submit/sync cost removed only at k == D

    @property
    def reuse(self) -> float:
        d = self.distinct_per_layer
        return self.units_per_layer / d if d > 0 else 1.0

    @property
    def cpu_ms_per_unit(self) -> float:
        u = self.units_per_layer
        return self.cpu_ms_per_layer / u if u > 0 else 0.0


def cost_at(c: Constants, k: int) -> LayerCost:
    """Predicted per-layer cost when `k` distinct experts are streamed."""
    k = max(0, min(k, int(round(c.distinct_per_layer))))
    units_left = max(c.units_per_layer - c.reuse * k, 0.0)
    cpu_ms = c.cpu_ms_per_unit * units_left
    if k > 0 and units_left <= 0.0:
        # The layer routes nothing to the CPU any more, so its submit/sync goes
        # away too. This is the corner the plan insisted the policy must be able
        # to reach, not just interior splits.
        cpu_ms = 0.0
    elif c.cpu_ms_per_layer > 0:
        cpu_ms += c.fixed_ms_per_layer
    gb = k * c.bytes_per_expert_gb
    link_ms = gb / c.link_gbs * 1000.0 if c.link_gbs > 0 else float("inf")
    extra_ms = gb * c.contention_ms_per_gb + k * c.gpu_ms_per_expert
    return LayerCost(k, max(cpu_ms, link_ms) + extra_ms, cpu_ms, link_ms, extra_ms)


def choose_k(c: Constants) -> LayerCost:
    """Pick the k that minimizes predicted layer time. Ties go to less streaming."""
    best = cost_at(c, 0)
    for k in range(1, int(round(c.distinct_per_layer)) + 1):
        cand = cost_at(c, k)
        if cand.layer_ms < best.layer_ms - 1e-9:
            best = cand
    return best


def predict_step(c: Constants, n_layers: int, step_ms: float) -> dict:
    """Turn the per-layer decision into a predicted step time and speedup."""
    base = cost_at(c, 0)
    best = choose_k(c)
    path_ms = base.layer_ms * n_layers          # what the CPU experts cost now
    rest_ms = step_ms - path_ms                 # attention, dense trunk, MTP, launch
    # A step cannot be shorter than its own expert path. If it is, the caller
    # paired a per-layer pole with a step time from a different configuration,
    # and the honest response is to say so -- the alternative is reporting a
    # 1e11x speedup off inconsistent inputs.
    consistent = rest_ms >= 0
    new_step = max(rest_ms, 0.0) + best.layer_ms * n_layers
    return {
        "k": best.k,
        "layer_ms_before": base.layer_ms,
        "layer_ms_after": best.layer_ms,
        "saved_ms_per_step": (base.layer_ms - best.layer_ms) * n_layers,
        "step_ms_before": step_ms,
        "step_ms_after": new_step,
        "speedup": step_ms / new_step if new_step > 0 else float("inf"),
        "bound_by": "link" if best.link_ms >= best.cpu_ms else "cpu",
        "inputs_consistent": consistent,
    }


def break_even_cpu_ms_per_layer(c: Constants, hi: float = 1e4) -> float:
    """Least CPU work a layer must have before streaming one expert can pay.

    Derived by bisection on `choose_k` itself rather than by a second closed
    form. A parallel derivation is how the two drift apart: the obvious formula
    ("one transfer must save more CPU time than it costs") ignores that CPU and
    link OVERLAP, so it demanded 2.85 ms/layer on this box while the optimizer
    was correctly streaming at 1.26. Whatever the model becomes, this stays
    consistent with the decision the runtime actually makes.

    Returns inf when no amount of CPU work makes streaming worthwhile (a link
    so slow that one transfer costs more than the work it can ever displace).
    """
    from dataclasses import replace

    if c.link_gbs <= 0 or c.distinct_per_layer <= 0 or c.units_per_layer <= 0:
        return float("inf")
    if choose_k(replace(c, cpu_ms_per_layer=hi)).k == 0:
        return float("inf")
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if choose_k(replace(c, cpu_ms_per_layer=mid)).k > 0:
            hi = mid
        else:
            lo = mid
    return hi
