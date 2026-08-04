#!/usr/bin/env python3
"""Stage 0: score an expert placement in milliseconds, not in points.

`EXPERT_PREFETCH_PLAN.md` needs a cheap, deterministic verdict on "did the
right experts end up in the right tier", because tok/s is noisy and confounded
-- accept length moves with the tier, and background load can fake a regression
outright (an 5.8h runaway `ugrep` once dragged decode 14 -> 0.77 tok/s and
produced a completely believable "too slow" conclusion).

WHY NOT A WEIGHTED HIT COUNT (10 for GPU / 2 for RAM / 1 for SSD):

  * It is not linear. A layer costs `max(cpu, gpu) + fixed submit/sync`. The
    FIRST non-resident expert in a layer buys nearly the whole penalty; the
    second through eighth ride the same submit/sync almost for free. A linear
    per-expert score therefore reports big wins where the clock does not move,
    which is exactly the trap behind "oracle coverage 69% -> 92% barely moves
    tok/s" and behind the voided SSD ladder.
  * It mixes two currencies. GPU-vs-RAM is LATENCY. RAM-vs-SSD is FIDELITY: an
    SSD-tier expert is never fetched on the critical path, it is dropped and a
    resident stand-in is substituted (kt_ep_wrapper.py, three-tier store). Add
    them into one number and a configuration can buy score by degrading its own
    output.

So the score is a PAIR, and the two halves are never summed:

  predicted_ms_per_step   what this placement costs, in milliseconds, from a
                          two-term model fitted to the measured tier ladder
  routing_fidelity        what fraction of the router's genuine top-K demand
                          actually got honored

The model:

    step_ms = floor + a_active * (active layer-calls / step)
                    + a_unit   * (CPU expert-token units / step)

`a_active` is the per-layer fixed cost of touching the CPU path at all
(submit + sync + wake); `a_unit` is the marginal cost of one more expert-token
on it. Both are FITTED, not assumed, and `--calibrate` reports the residual
against the measured ladder. An instrument that cannot retrodict a ladder it
has already seen has no business refereeing new work, so that residual is the
gate.

Counters come from the live server (KT_PLACE_SCORE=1), accumulated IN the
decode CUDA graph. That matters more than it sounds: `KT_DUMP_TOPK` writes from
Python inside the captured region, which does not re-execute on replay, so it
silently only ever sees PREFILL. Prefill is the wrong regime here -- the
adaptive cache has promoted the hot experts to VRAM, so residency correlates
strongly with router rank, and a prefill-derived estimate of "how much lands on
the CPU" is biased low.

    # calibrate the instrument against a measured ladder, then use it
    .venv/bin/python bench/placement_score.py --calibrate --tiers 0,1,2,4,6,8
    .venv/bin/python bench/placement_score.py --tier 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib import request

import torch

RESET_FLAG = os.environ.get("KT_PLACE_RESET_FLAG", "/tmp/kt_place_reset")
DUMP_PT = os.environ.get("KT_PLACE_DUMP_PT", "/tmp/kt_place.pt")

# Slot layout of the per-layer buffer; mirrors kt_ep_wrapper._KT_PLACE_SLOTS.
CALLS, TOKENS, CPU_UNITS, DISTINCT, ACTIVE, KEPT, MASS_KEPT, MASS_TOT, \
    UNREACH, RESIDENT_KEPT, PERSIST_HIT, PREV_DISTINCT = range(12)
N_SLOTS = 12

PROMPT = (
    "Write a detailed technical essay about how modern CPUs and GPUs differ in "
    "their approach to parallelism, memory hierarchy, and scheduling."
)


def run_decode(url: str, model: str, tokens: int) -> dict:
    """One streaming completion; returns median ms/step and accept length."""
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = request.Request(f"{url}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
    stamps, usage = [], {}
    with request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            d = (obj.get("delta") or (obj.get("choices") or [{}])[0].get("delta") or {})
            if d.get("content") or d.get("reasoning_content"):
                stamps.append(time.perf_counter())
    if len(stamps) < 5:
        raise RuntimeError(f"{model}: only {len(stamps)} steps, cannot measure")
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    n_tok = (usage or {}).get("completion_tokens") or len(stamps)
    return {"ms_per_step": statistics.median(gaps),
            "accept": n_tok / len(stamps),
            "n_steps": len(stamps),
            "tok_per_s": 1000.0 * (n_tok / len(stamps)) / statistics.median(gaps)}


def snapshot() -> dict:
    """Read the server's cumulative counters.

    Deliberately NOT a reset-and-measure handshake. The counters are zeroed
    only on the forward thread, so a reset issued against an idle server sits
    unconsumed until traffic arrives -- and a warmup request large enough to
    consume it would then pollute the very window being measured. Taking two
    cumulative snapshots and subtracting has neither problem, and the step
    count travels in the same snapshot as the counters so the two can never
    disagree about which window they describe.
    """
    if not Path(DUMP_PT).is_file():
        raise SystemExit(
            f"no counter dump at {DUMP_PT}. Is the server running with "
            "KT_PLACE_SCORE=1? (the instrument is compiled out otherwise)")
    snap = torch.load(DUMP_PT, map_location="cpu", weights_only=False)
    tot = torch.zeros(N_SLOTS, dtype=torch.float64)
    for buf in snap["layers"].values():
        tot += buf
    return {"steps": int(snap["steps"]), "total": tot,
            "boot": snap.get("boot", ""),
            "n_layers": len(snap["layers"]),
            "counts": snap.get("counts") or {},
            "pred": snap.get("pred") or {}}


def delta(before: dict, after: dict) -> dict:
    steps = after["steps"] - before["steps"]
    if steps <= 0:
        raise SystemExit(
            "the counters did not advance across the request. Either the "
            "server is not ticking (KT_PLACE_SCORE unset) or the snapshot "
            "period is longer than the run -- lower KT_PLACE_DUMP_EVERY.")
    # Per-expert demand is differenced too: the raw counters are cumulative
    # from boot, so a shared histogram would blend every tier already measured
    # and quietly overweight the ones that fire the most experts.
    counts = {lid: c - before["counts"].get(lid, torch.zeros_like(c))
              for lid, c in after["counts"].items()}
    pred = {}
    depths = after["pred"].get("depths") if after["pred"] else None
    if depths:
        acc = {}
        for lid, st in after["pred"].items():
            if lid == "depths":
                continue
            prev = before["pred"].get(lid)
            d = st - prev if prev is not None else st
            for slot, n in enumerate(depths):
                h, t = float(d[slot, 0]), float(d[slot, 1])
                a = acc.setdefault(n, [0.0, 0.0])
                a[0] += h
                a[1] += t
        pred = {n: (h / t if t else 0.0) for n, (h, t) in acc.items()}
    return {"steps": steps, "total": after["total"] - before["total"],
            "n_layers": after["n_layers"], "counts": counts, "pred": pred}


def wait_for_new_snapshot(before: dict, timeout_s: float = 60.0) -> dict:
    """Block until the server writes a snapshot newer than `before`.

    Without this the 'after' read can land on the same periodic dump as the
    'before' read and report a zero-length window.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = snapshot()
        if s["boot"] != before["boot"]:
            raise SystemExit(
                "the server restarted mid-measurement; the counters belong to "
                "a different boot and cannot be differenced")
        if s["steps"] > before["steps"]:
            return s
        time.sleep(0.1)
    raise SystemExit("server stopped writing counter snapshots")


def fresh_baseline(url: str, model: str, timeout_s: float = 120.0) -> dict:
    """Get a 'before' snapshot that is guaranteed to come from THIS server.

    The dump file outlives the server, so a snapshot taken right after a
    restart can carry the previous run's much larger step count -- against
    which the new server's counters never appear to advance. Drive decode until
    the file's boot id stops changing, and use that as the baseline.
    """
    deadline = time.time() + timeout_s
    boot = None
    while time.time() < deadline:
        run_decode(url, model, 64)
        s = snapshot()
        if boot is not None and s["boot"] == boot:
            return s
        boot = s["boot"]
    raise SystemExit("could not establish a counter baseline for this server")


def features(c: dict) -> dict:
    """Per-step aggregates -- the model's independent variables."""
    s, t = c["steps"], c["total"]
    active = float(t[ACTIVE]) / s
    units = float(t[CPU_UNITS]) / s
    distinct = float(t[DISTINCT]) / s
    mass_tot = float(t[MASS_TOT])
    return {
        "active_layer_calls_per_step": active,
        "cpu_units_per_step": units,
        "distinct_per_step": distinct,
        # One transfer serves `reuse` units of CPU work. It is >1 only because
        # MTP puts several tokens in a step; at batch 1 without MTP it is 1.0.
        "reuse": units / distinct if distinct else 0.0,
        "routing_fidelity": float(t[MASS_KEPT]) / mass_tot if mass_tot else 0.0,
        "unreachable_per_step": float(t[UNREACH]) / s,
        "resident_frac_of_kept": (float(t[RESIDENT_KEPT]) / float(t[KEPT])
                                  if float(t[KEPT]) else 0.0),
        "layer_calls_per_step": float(t[CALLS]) / s,
        # Accuracy of the cheapest next-step predictor: "fetch whatever this
        # layer needed last step". Prefetch built on recent history cannot beat
        # this without a genuinely better model, so it is the number that
        # decides whether the prefetch stage is worth building.
        "persistence": (float(t[PERSIST_HIT]) / float(t[DISTINCT])
                        if float(t[DISTINCT]) else 0.0),
    }


def fit(rows: list[dict], floor_ms: float) -> dict:
    """Least squares for (a_active, a_unit) with the floor pinned.

    The floor is pinned rather than fitted because it is directly measured:
    tier 0 routes nothing to the CPU, so its step time IS the model's intercept.
    Fitting three parameters to five points would let a bad intercept absorb the
    structure the score is supposed to expose.
    """
    import numpy as np
    A = np.array([[r["active_layer_calls_per_step"], r["cpu_units_per_step"]]
                  for r in rows], dtype=np.float64)
    y = np.array([r["ms_per_step"] - floor_ms for r in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"a_active_ms": float(coef[0]), "a_unit_ms": float(coef[1])}


def predict(f: dict, coef: dict, floor_ms: float) -> float:
    return (floor_ms
            + coef["a_active_ms"] * f["active_layer_calls_per_step"]
            + coef["a_unit_ms"] * f["cpu_units_per_step"])


def window_analysis(counts: dict, sizes: list[int],
                    expert_mb: float = 9.7) -> dict:
    """How concentrated is CPU-expert demand, per layer?

    A streaming or prefetch scheme cannot hold every non-resident expert
    somewhere fast; it holds a WINDOW of the hottest ones per layer. An expert
    outside the window just takes the existing CPU path, so this is not a
    correctness question -- it is the fraction of the predicted speedup that
    survives.

    Read this against the convexity result in Stage A2: contention is ~20x
    cheaper once a layer routes NOTHING to the CPU, so what matters is not mean
    coverage but the fraction of LAYERS fully covered. Both are reported, and
    they can differ a lot.
    """
    if not counts:
        raise SystemExit("no per-expert counts in the measurement window")
    out = {}
    for W in sizes:
        hit = tot = 0.0
        full_layers = n_layers = 0
        for lid, c in counts.items():
            s = float(c.sum())
            if s <= 0:
                continue
            n_layers += 1
            tot += s
            nz = int((c > 0).sum())
            k = min(W, nz)
            hit += float(torch.topk(c, k).values.sum()) if k else 0.0
            if nz <= W:
                full_layers += 1
        out[W] = {
            "demand_covered": hit / tot if tot else 0.0,
            "layers_fully_covered": full_layers / n_layers if n_layers else 0.0,
            # Both TP halves, all layers with demand.
            "pinned_gb": W * n_layers * expert_mb / 1024.0 * 2,
            "n_layers": n_layers,
        }
    return out


def measure_tier(url: str, model: str, tier: int, tokens: int,
                 warmup: bool) -> dict:
    name = f"{model}-top{tier}"
    before = fresh_baseline(url, name) if warmup else snapshot()
    r = run_decode(url, name, tokens)
    after = wait_for_new_snapshot(before)
    c = delta(before, after)
    row = {"tier": tier, **r, **features(c), "counter_steps": c["steps"],
           "n_layers": c["n_layers"],
           "lookahead_accuracy": {str(k): v for k, v in c["pred"].items()}}
    return row, c["counts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--tiers", default="", help="calibration ladder")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--coef", default="bench/profile_out/placement_model.json")
    ap.add_argument("--windows", default="",
                    help="also report streaming-window coverage, e.g. 4,8,16,32")
    ap.add_argument("--link-gbs", type=float, default=61.3,
                    help="measured AGGREGATE host->device GB/s under decode "
                         "(bench/pcie_contention.py, queue depth 1)")
    ap.add_argument("--expert-mb", type=float, default=9.7,
                    help="TP-sharded bytes of one expert in one layer, per card")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def report_budget(r: dict, floor: float) -> dict:
        """Can the link carry a FULL elimination of the CPU path in one step?

        This is the question Stage A2's convexity result forces. Partial
        coverage keeps the CPU path alive and pays the expensive contention
        regime, so what matters is whether every distinct non-resident expert
        can be moved inside one step. Both TP halves must move, so the bill is
        `distinct x expert_mb x 2`.
        """
        gb = r["distinct_per_step"] * args.expert_mb * 2 / 1024.0
        link_ms = gb / args.link_gbs * 1000.0
        # The floor is the honest denominator: after elimination the step is
        # the top0 floor, not today's slower step.
        duty = link_ms / floor if floor else 0.0
        print(f"\ntransfer budget for FULL elimination: "
              f"{gb:.2f} GB/step, {link_ms:.1f} ms at {args.link_gbs:.1f} GB/s "
              f"aggregate -> link duty {duty*100:.0f}% of a {floor:.1f} ms step")
        if duty > 1.0:
            print("  the link CANNOT cover every layer; coverage caps at "
                  f"~{100/duty:.0f}% of layers and the rest keep the CPU path")
        return {"gb_per_step": gb, "link_ms": link_ms, "duty": duty}

    def report_windows(counts):
        if not args.windows:
            return {}
        sizes = [int(x) for x in args.windows.split(",") if x.strip()]
        wa = window_analysis(counts, sizes, args.expert_mb)
        print(f"\n{'window':>7} {'pinned RAM':>11} {'demand covered':>15} "
              f"{'layers fully covered':>21}")
        for W, r in sorted(wa.items()):
            print(f"{W:>7} {r['pinned_gb']:>9.1f} GB {r['demand_covered']*100:>14.1f}% "
                  f"{r['layers_fully_covered']*100:>20.1f}%")
        return wa

    if args.calibrate:
        tiers = [int(t) for t in (args.tiers or "0,1,2,4,6,8").split(",")]
        if 0 not in tiers:
            raise SystemExit("calibration needs tier 0: it measures the floor")
        rows, counts_by_tier = [], {}
        for t in tiers:
            r, cnt = measure_tier(args.url, args.model, t, args.tokens,
                                  not args.no_warmup)
            rows.append(r)
            counts_by_tier[t] = cnt
            print(f"  top{t}: {r['ms_per_step']:7.2f} ms/step  "
                  f"accept {r['accept']:.3f}  {r['tok_per_s']:6.2f} tok/s  |  "
                  f"active {r['active_layer_calls_per_step']:6.2f}  "
                  f"units {r['cpu_units_per_step']:7.2f}  "
                  f"distinct {r['distinct_per_step']:6.2f}  "
                  f"fidelity {r['routing_fidelity']:.3f}")

        floor = next(r for r in rows if r["tier"] == 0)["ms_per_step"]
        train = [r for r in rows if r["tier"] != 0]
        coef = fit(train, floor)

        print(f"\nfitted: floor {floor:.2f} ms  "
              f"a_active {coef['a_active_ms']:.4f} ms/active-layer-call  "
              f"a_unit {coef['a_unit_ms']:.4f} ms/expert-token")
        print(f"\n{'tier':>5} {'measured':>10} {'predicted':>10} {'err':>8} "
              f"{'fixed share':>12}")
        worst = 0.0
        for r in rows:
            p = predict(r, coef, floor)
            err = (p - r["ms_per_step"]) / r["ms_per_step"] * 100.0
            worst = max(worst, abs(err))
            fixed = coef["a_active_ms"] * r["active_layer_calls_per_step"]
            marg = coef["a_unit_ms"] * r["cpu_units_per_step"]
            share = fixed / (fixed + marg) * 100.0 if (fixed + marg) > 0 else 0.0
            print(f"{r['tier']:>5} {r['ms_per_step']:>10.2f} {p:>10.2f} "
                  f"{err:>7.1f}% {share:>11.0f}%")

        print(f"\n{'tier':>5} {'persistence':>12}  (accuracy of 'fetch what this "
              f"layer needed last step')")
        for r in rows:
            if r["distinct_per_step"] > 0:
                print(f"{r['tier']:>5} {r['persistence']*100:>11.1f}%")
        la = next((r["lookahead_accuracy"] for r in rows
                   if r.get("lookahead_accuracy")), None)
        if la:
            print("\nlookahead prediction: run layer L+n's router early, on "
                  "layer L's hidden state")
            print(f"{'depth n':>8} {'top-K recall':>13}")
            for n in sorted(la, key=int):
                tag = "  (scoring-path check, no lead time)" if n == "0" else ""
                print(f"{n:>8} {la[n]*100:>12.1f}%{tag}")
        budgets = {}
        for r in rows:
            if r["distinct_per_step"] > 0:
                print(f"\n[top{r['tier']}]", end="")
                budgets[str(r["tier"])] = report_budget(r, floor)
        # Window sizing is reported for the SHIPPED tier: it is the one a
        # streaming window would actually be built for.
        ship = 2 if 2 in counts_by_tier else max(counts_by_tier)
        print(f"\nstreaming-window sizing at top{ship}:", end="")
        wa = report_windows(counts_by_tier[ship])
        out = {"ts": time.time(), "floor_ms": floor, **coef,
               "budgets": budgets, "window_tier": ship,
               "worst_abs_err_pct": worst, "rows": rows,
               "windows": {str(k): v for k, v in wa.items()}}
        Path(args.coef).parent.mkdir(parents=True, exist_ok=True)
        Path(args.coef).write_text(json.dumps(out, indent=1))
        print(f"\nworst retrodiction error {worst:.1f}% "
              f"({'PASS' if worst <= 10.0 else 'FAIL'}, gate is 10%)")
        print(f"wrote {args.coef}")
        return 0 if worst <= 10.0 else 1

    cpath = Path(args.coef)
    if not cpath.is_file():
        raise SystemExit(f"no fitted model at {cpath}; run --calibrate first")
    m = json.loads(cpath.read_text())
    r, counts = measure_tier(args.url, args.model, args.tier, args.tokens,
                             not args.no_warmup)
    p = predict(r, m, m["floor_ms"])
    print(f"top{args.tier}: predicted {p:.2f} ms/step "
          f"(measured {r['ms_per_step']:.2f}), "
          f"routing_fidelity {r['routing_fidelity']:.4f}")
    print(f"  active layer-calls/step {r['active_layer_calls_per_step']:.2f}"
          f"   CPU units/step {r['cpu_units_per_step']:.2f}"
          f"   distinct/step {r['distinct_per_step']:.2f}"
          f"   reuse {r['reuse']:.2f}")
    if r["unreachable_per_step"] > 0:
        print(f"  WARNING: {r['unreachable_per_step']:.2f} genuine top-K slots "
              "per step were UNREACHABLE (SSD tier) and were substituted. "
              "That is a fidelity loss, not a latency -- it is why the two "
              "halves of the score are never summed.")
    print(f"  persistence {r['persistence']*100:.1f}%")
    report_budget(r, m["floor_ms"])
    report_windows(counts)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"predicted_ms_per_step": p, "model": m, "measured": r}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
