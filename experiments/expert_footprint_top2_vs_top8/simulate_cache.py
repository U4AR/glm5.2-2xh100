#!/usr/bin/env python3
"""Offline replay simulator for adaptive routed-expert GPU caching.

The replay unit is one captured token-layer row. Expert ids are sorted by router
weight before scoring because the captured ids are not guaranteed to arrive in
rank order.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from _consts import (
    N_ROUTED_EXPERTS,
    NUM_EXPERTS_PER_TOKEN,
    PER_EXPERT_MIB,
    TOP2_K,
)


DEFAULT_CAPACITIES = (32, 64, 96, 104, 128, 160, 192)
DEFAULT_UPDATE_INTERVALS = (16, 32, 64)
DEFAULT_TASK_ORDER = (
    "exp3_llm-inference-batching-scheduler.pt",
    "exp3_largest-eigenval.pt",
    "exp3_fix-git.pt",
    "exp3_compile-compcert.pt",
    "exp3_git-multibranch.pt",
)
RANK_CREDIT_WITH_TAIL = (1.20, 1.00, 0.35, 0.25, 0.15, 0.08, 0.04, 0.02)
RANK_CREDIT_TOP2_ONLY = (1.20, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def load_trace(path: Path) -> list:
    return torch.load(path, map_location="cpu", weights_only=False)


def iter_sorted_weighted_rows(trace: list) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    for rec in trace:
        if len(rec) == 3:
            layer_idx, ids, weights = rec
        elif len(rec) == 2:
            layer_idx, ids = rec
            weights = None
        else:
            continue

        ids_t = torch.as_tensor(ids).view(-1, NUM_EXPERTS_PER_TOKEN).to(torch.long)
        if weights is None:
            weights_t = torch.arange(
                NUM_EXPERTS_PER_TOKEN, 0, -1, dtype=torch.float32
            ).expand_as(ids_t)
        else:
            weights_t = torch.as_tensor(weights).view(ids_t.shape).to(torch.float32)
        order = torch.argsort(weights_t, dim=-1, descending=True)
        sorted_ids = torch.gather(ids_t, 1, order)
        sorted_weights = torch.gather(weights_t, 1, order)
        for row_ids, row_weights in zip(sorted_ids, sorted_weights):
            yield int(layer_idx), row_ids, row_weights


@dataclasses.dataclass
class LayerPolicyState:
    resident: torch.Tensor
    score: torch.Tensor
    count: torch.Tensor
    last_seen: torch.Tensor
    event: int = 0


class Policy:
    name = "base"

    def __init__(
        self,
        capacity: int,
        *,
        update_interval: int,
        max_swaps: int,
        half_life: float,
        resident_bonus: float,
        promotion_cost: float,
    ):
        self.capacity = min(capacity, N_ROUTED_EXPERTS)
        self.update_interval = update_interval
        self.max_swaps = max_swaps
        self.half_life = half_life
        self.resident_bonus = resident_bonus
        self.promotion_cost = promotion_cost
        self.layers: dict[int, LayerPolicyState] = {}
        self.top2_hits = 0
        self.top2_total = 0
        self.top8_hits = 0
        self.top8_total = 0
        self.swaps = 0
        self.rows = 0
        rank_credit = getattr(self, "rank_credit", None)
        self.rank_credit_tensor = (
            torch.tensor(rank_credit, dtype=torch.float32)
            if rank_credit is not None
            else None
        )

    def _layer(self, layer_idx: int) -> LayerPolicyState:
        state = self.layers.get(layer_idx)
        if state is None:
            resident = torch.zeros(N_ROUTED_EXPERTS, dtype=torch.bool)
            resident[: self.capacity] = True
            state = LayerPolicyState(
                resident=resident,
                score=torch.zeros(N_ROUTED_EXPERTS, dtype=torch.float32),
                count=torch.zeros(N_ROUTED_EXPERTS, dtype=torch.int64),
                last_seen=torch.full((N_ROUTED_EXPERTS,), -1, dtype=torch.int64),
            )
            self.layers[layer_idx] = state
        return state

    def observe(self, layer_idx: int, ids: torch.Tensor, weights: torch.Tensor) -> None:
        state = self._layer(layer_idx)
        self.rows += 1
        ids_t = ids.to(dtype=torch.long)
        weights_t = weights.to(dtype=torch.float32)
        top2 = ids_t[:TOP2_K]
        top2_valid = (top2 >= 0) & (top2 < N_ROUTED_EXPERTS)
        top8_valid = (ids_t >= 0) & (ids_t < N_ROUTED_EXPERTS)
        self.top2_hits += int(state.resident[top2[top2_valid]].sum().item())
        self.top2_total += int(top2_valid.sum().item())
        self.top8_hits += int(state.resident[ids_t[top8_valid]].sum().item())
        self.top8_total += int(top8_valid.sum().item())

        state.event += 1
        self._update_scores(state, ids_t, weights_t)
        if state.event % self.update_interval == 0:
            self._refresh_residents(state)

    def _update_scores(
        self, state: LayerPolicyState, ids: torch.Tensor, weights: torch.Tensor
    ) -> None:
        raise NotImplementedError

    def _effective_scores(self, state: LayerPolicyState) -> torch.Tensor:
        effective = state.score.clone()
        effective[state.resident] += self.resident_bonus
        effective[~state.resident] -= self.promotion_cost
        return effective

    def _refresh_residents(self, state: LayerPolicyState) -> None:
        if self.capacity >= N_ROUTED_EXPERTS:
            return
        effective = self._effective_scores(state)
        desired = torch.topk(effective, k=self.capacity, largest=True, sorted=True).indices
        desired_mask = torch.zeros(N_ROUTED_EXPERTS, dtype=torch.bool)
        desired_mask[desired] = True
        additions = torch.where(desired_mask & ~state.resident)[0]
        evictions = torch.where(state.resident & ~desired_mask)[0]
        if additions.numel() > 0:
            additions = additions[torch.argsort(effective[additions], descending=True)]
        if evictions.numel() > 0:
            evictions = evictions[torch.argsort(effective[evictions], descending=False)]
        swap_count = min(self.max_swaps, int(additions.numel()), int(evictions.numel()))
        if swap_count <= 0:
            return
        state.resident[evictions[:swap_count]] = False
        state.resident[additions[:swap_count]] = True
        self.swaps += swap_count

    def metrics(self) -> dict[str, float | int | str]:
        top2_misses = self.top2_total - self.top2_hits
        top8_misses = self.top8_total - self.top8_hits
        return {
            "policy": self.name,
            "capacity": self.capacity,
            "update_interval": self.update_interval,
            "top2_hit_rate": self.top2_hits / max(1, self.top2_total),
            "top8_hit_rate": self.top8_hits / max(1, self.top8_total),
            "top2_misses_per_row": top2_misses / max(1, self.rows),
            "top8_misses_per_row": top8_misses / max(1, self.rows),
            "swaps": self.swaps,
            "copy_mib": self.swaps * PER_EXPERT_MIB,
            "rows": self.rows,
        }


class StaticUniformPolicy(Policy):
    name = "static_uniform"

    def _update_scores(
        self, state: LayerPolicyState, ids: torch.Tensor, weights: torch.Tensor
    ) -> None:
        return

    def _refresh_residents(self, state: LayerPolicyState) -> None:
        return


class LRUPolicy(Policy):
    name = "lru"

    def _update_scores(
        self, state: LayerPolicyState, ids: torch.Tensor, weights: torch.Tensor
    ) -> None:
        valid = ids[(ids >= 0) & (ids < N_ROUTED_EXPERTS)]
        if valid.numel() == 0:
            return
        state.last_seen[valid] = state.event
        state.score[valid] = float(state.event)


class FrequencyPolicy(Policy):
    name = "frequency"

    def _update_scores(
        self, state: LayerPolicyState, ids: torch.Tensor, weights: torch.Tensor
    ) -> None:
        valid = ids[(ids >= 0) & (ids < N_ROUTED_EXPERTS)]
        if valid.numel() == 0:
            return
        state.count.index_add_(0, valid, torch.ones_like(valid, dtype=torch.int64))
        state.score.copy_(state.count.to(torch.float32))


class WeightedRouterPolicy(Policy):
    name = "weighted_top2"
    rank_credit = RANK_CREDIT_TOP2_ONLY

    def _update_scores(
        self, state: LayerPolicyState, ids: torch.Tensor, weights: torch.Tensor
    ) -> None:
        decay = math.pow(0.5, 1.0 / max(1.0, self.half_life))
        state.score.mul_(decay)
        valid_mask = (ids >= 0) & (ids < N_ROUTED_EXPERTS)
        if not bool(valid_mask.any().item()):
            return
        weight_sum = weights.clamp_min(0.0).sum().clamp_min(1e-12)
        assert self.rank_credit_tensor is not None
        credits = (
            self.rank_credit_tensor[: ids.numel()]
            * weights.clamp_min(0.0)
            / weight_sum
        )
        valid = ids[valid_mask]
        state.score.index_add_(0, valid, credits[valid_mask])
        state.count.index_add_(0, valid, torch.ones_like(valid, dtype=torch.int64))
        state.last_seen[valid] = state.event


class WeightedRouterWithTailPolicy(WeightedRouterPolicy):
    name = "weighted_with_tail"
    rank_credit = RANK_CREDIT_WITH_TAIL


POLICIES = (
    StaticUniformPolicy,
    LRUPolicy,
    FrequencyPolicy,
    WeightedRouterPolicy,
    WeightedRouterWithTailPolicy,
)


def default_trace_paths(runs_dir: Path) -> list[Path]:
    paths = [runs_dir / name for name in DEFAULT_TASK_ORDER if (runs_dir / name).is_file()]
    if paths:
        return paths
    return sorted(runs_dir.glob("exp3_*.pt"))


def run_simulation(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    traces = [(path.name, load_trace(path)) for path in args.traces]
    results = []
    for capacity in args.capacities:
        for update_interval in args.update_intervals:
            policies = [
                policy_cls(
                    capacity,
                    update_interval=update_interval,
                    max_swaps=args.max_swaps,
                    half_life=args.half_life,
                    resident_bonus=args.resident_bonus,
                    promotion_cost=args.promotion_cost,
                )
                for policy_cls in POLICIES
            ]
            for task_name, trace in traces:
                for layer_idx, ids, weights in iter_sorted_weighted_rows(trace):
                    for policy in policies:
                        policy.observe(layer_idx, ids, weights)
                for policy in policies:
                    metric = policy.metrics()
                    metric["scope"] = task_name
                    metric["through_task"] = task_name
                    results.append(dict(metric))
            for policy in policies:
                metric = policy.metrics()
                metric["scope"] = "all_tasks"
                metric["through_task"] = traces[-1][0] if traces else ""
                results.append(dict(metric))
    return results


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def write_markdown(results: list[dict[str, float | int | str]], path: Path) -> None:
    all_rows = [row for row in results if row["scope"] == "all_tasks"]
    all_rows.sort(key=lambda r: (int(r["capacity"]), int(r["update_interval"]), str(r["policy"])))
    lines = [
        "# Adaptive Expert Cache Simulation",
        "",
        "Replay unit: one token-layer row from the captured prefill traces. Hits are",
        "counted before the policy sees the row, then the policy updates state and may",
        "refresh residents at its update interval.",
        "",
        "| capacity/layer | interval | policy | top2 hit | top8 hit | top2 misses/row | swaps | copy GiB |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        copy_gib = float(row["copy_mib"]) / 1024.0
        lines.append(
            f"| {row['capacity']} | {row['update_interval']} | {row['policy']} | "
            f"{fmt_pct(float(row['top2_hit_rate']))} | "
            f"{fmt_pct(float(row['top8_hit_rate']))} | "
            f"{float(row['top2_misses_per_row']):.3f} | "
            f"{int(row['swaps'])} | {copy_gib:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    return (
        '<polyline fill="none" stroke="'
        + color
        + '" stroke-width="2" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        + '" />'
    )


def write_hit_rate_svg(results: list[dict[str, float | int | str]], path: Path) -> None:
    rows = [
        row
        for row in results
        if row["scope"] == "all_tasks" and int(row["update_interval"]) == 32
    ]
    by_policy: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    width, height = 920, 520
    left, right, top, bottom = 70, 30, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    capacities = sorted({int(row["capacity"]) for row in rows})
    if not capacities:
        return
    min_cap, max_cap = min(capacities), max(capacities)

    def xscale(cap: int) -> float:
        if min_cap == max_cap:
            return left + plot_w / 2
        return left + (cap - min_cap) / (max_cap - min_cap) * plot_w

    def yscale(rate: float) -> float:
        return top + (1.0 - rate) * plot_h

    colors = {
        "static_uniform": "#666666",
        "lru": "#0072b2",
        "frequency": "#009e73",
        "weighted_top2": "#d55e00",
        "weighted_with_tail": "#cc79a7",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" />',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" />',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">GPU experts per layer</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})" font-family="sans-serif" font-size="14">Top-2 hit rate</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yscale(tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333" />')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.2f}</text>')
    for cap in capacities:
        x = xscale(cap)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333" />')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{cap}</text>')
    legend_y = top + 10
    for idx, (policy, policy_rows) in enumerate(sorted(by_policy.items())):
        policy_rows.sort(key=lambda r: int(r["capacity"]))
        color = colors.get(policy, "#000000")
        points = [
            (xscale(int(row["capacity"])), yscale(float(row["top2_hit_rate"])))
            for row in policy_rows
        ]
        parts.append(_svg_polyline(points, color))
        lx = left + 12
        ly = legend_y + idx * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{color}" stroke-width="2" />')
        parts.append(f'<text x="{lx + 26}" y="{ly + 4}" font-family="sans-serif" font-size="12">{policy}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    runs_dir = here / "runs"
    parser.add_argument("--runs-dir", type=Path, default=runs_dir)
    parser.add_argument("--out-dir", type=Path, default=runs_dir)
    parser.add_argument("--capacity", dest="capacities", type=int, nargs="*", default=list(DEFAULT_CAPACITIES))
    parser.add_argument("--update-interval", dest="update_intervals", type=int, nargs="*", default=list(DEFAULT_UPDATE_INTERVALS))
    parser.add_argument("--max-swaps", type=int, default=8)
    parser.add_argument("--half-life", type=float, default=96.0)
    parser.add_argument("--resident-bonus", type=float, default=0.25)
    parser.add_argument("--promotion-cost", type=float, default=0.10)
    parser.add_argument("traces", type=Path, nargs="*")
    args = parser.parse_args()
    if not args.traces:
        args.traces = default_trace_paths(args.runs_dir)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = run_simulation(args)
    json_path = args.out_dir / "cache_simulation.json"
    md_path = args.out_dir / "cache_simulation.md"
    svg_path = args.out_dir / "figs" / "cache_top2_hit_rate_vs_capacity.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, md_path)
    write_hit_rate_svg(results, svg_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
