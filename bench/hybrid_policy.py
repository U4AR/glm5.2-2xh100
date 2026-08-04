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
    saved = (base.layer_ms - best.layer_ms) * n_layers
    new_step = step_ms - saved
    return {
        "k": best.k,
        "layer_ms_before": base.layer_ms,
        "layer_ms_after": best.layer_ms,
        "saved_ms_per_step": saved,
        "step_ms_before": step_ms,
        "step_ms_after": new_step,
        "speedup": step_ms / new_step if new_step > 0 else float("inf"),
        "bound_by": "link" if best.link_ms >= best.cpu_ms else "cpu",
    }


def break_even_cpu_ms_per_layer(c: Constants) -> float:
    """Least CPU work a layer must have before streaming one expert can pay.

    Streaming is worth starting only when removing `reuse` units of CPU work
    saves more than one expert's transfer costs. Below this the answer is k=0
    no matter how the rest is tuned -- which is the situation at safe2 on this
    box, and the number to check first on any new machine.
    """
    gb = c.bytes_per_expert_gb
    one_transfer = gb / c.link_gbs * 1000.0 + gb * c.contention_ms_per_gb + c.gpu_ms_per_expert
    # Removing k=1 leaves (U - reuse) units; it pays when the old pole exceeds
    # the new max(cpu, link) + extra.
    if c.reuse <= 0 or c.units_per_layer <= 0:
        return float("inf")
    return one_transfer * c.units_per_layer / c.reuse
