#!/usr/bin/env python3
"""Phase 2: measure this machine, write a profile, say what it implies.

Produces `bench/profile_out/hybrid_profile.<hostname>.json`, read by the server
at boot and never measured in-process. Every constant is measured on the host it
describes, so the same code makes a different -- and correct -- decision on a
different machine:

  * a box with a fast host-to-device link (GH200 C2C, ~900 GB/s vs this box's
    56 GB/s per card) makes transfers ~16x cheaper and streams far more
  * a box with fewer GPU-resident experts has a bigger CPU pole and streams more
  * a box with a faster CPU, or one holding most experts in VRAM, streams less
    or nothing at all

The profile records what it was measured against (host, GPUs, model, expert
size, tier), so a stale profile is detected rather than silently trusted.

    .venv/bin/python bench/calibrate_hybrid.py --tiers 2,4,6,8

Requires a running server; the CPU curve and the contention constant are
measured live, because both depend on the deployed configuration (GPU_EXPERTS,
MTP depth, TP size) and not on the hardware alone.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_policy import Constants, break_even_cpu_ms_per_layer, predict_step  # noqa: E402

PROMPT = (
    "Write a detailed technical essay about how modern CPUs and GPUs differ in "
    "their approach to parallelism, memory hierarchy, and scheduling."
)


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def host_identity() -> dict:
    gpus = sh(["nvidia-smi", "--query-gpu=name,pci.bus_id,memory.total",
               "--format=csv,noheader"]).splitlines()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": platform.processor() or sh(["bash", "-lc",
                                           "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip(),
        "gpus": [g.strip() for g in gpus],
        "n_gpus": len(gpus),
    }


def decode_step_ms(url: str, model: str, tokens: int, runs: int) -> dict:
    """Median ms/step and accept length for one tier."""
    out = []
    for _ in range(runs):
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
                ch = (obj.get("choices") or [{}])[0].get("delta") or {}
                if ch.get("content") or ch.get("reasoning_content"):
                    stamps.append(time.perf_counter())
        if len(stamps) < 5:
            raise RuntimeError("too few steps to measure")
        gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
        n_tok = (usage or {}).get("completion_tokens") or len(stamps)
        out.append({"ms": statistics.median(gaps),
                    "accept": n_tok / len(stamps)})
    return {"ms_per_step": statistics.median([o["ms"] for o in out]),
            "accept": statistics.median([o["accept"] for o in out]),
            "runs": len(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tiers", default="2,4,6,8")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--n-layers", type=int, default=75)
    ap.add_argument("--verify-tokens", type=int, default=4,
                    help="tokens per step (MTP draft tokens)")
    ap.add_argument("--expert-mb", type=float, default=9.7,
                    help="TP-sharded bytes of one expert in one layer, per card")
    ap.add_argument("--distinct", default="bench/profile_out/phase2_distinct_experts.json",
                    help="output of bench/measure_distinct_experts.py")
    ap.add_argument("--contention", default="bench/profile_out/phase1b_safe_qd1.json",
                    help="output of bench/pcie_contention.py at queue depth 1")
    ap.add_argument("--mechanism", default="bench/profile_out/phase3_mechanism.json",
                    help="output of bench/stream_mechanism_spike.py")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
    ident = host_identity()

    # ---- the CPU cost curve, measured on the deployed configuration ---------
    print("measuring the CPU expert cost curve (this is deployment-specific)...")
    curve = {}
    floor = decode_step_ms(args.url, f"{args.model}-top0", args.tokens, args.runs)
    print(f"  top0 (no CPU experts): {floor['ms_per_step']:.2f} ms/step")
    for t in tiers:
        r = decode_step_ms(args.url, f"{args.model}-top{t}", args.tokens, args.runs)
        cpu_ms = r["ms_per_step"] - floor["ms_per_step"]
        curve[t] = {"ms_per_step": r["ms_per_step"], "accept": r["accept"],
                    "cpu_path_ms": cpu_ms,
                    "cpu_ms_per_layer": cpu_ms / args.n_layers}
        print(f"  top{t}: {r['ms_per_step']:8.2f} ms/step   CPU path {cpu_ms:7.2f} ms"
              f"   ({cpu_ms / args.n_layers:.3f} ms/layer)")

    # ---- transport constants ----------------------------------------------
    link_gbs, contention = None, None
    cpath = Path(args.contention)
    if cpath.is_file():
        cj = json.loads(cpath.read_text())
        summ = cj.get("summary") or {}
        # Per card, under load, at the shallow queue depth the design mandates.
        percard = [v["loaded_busy_gbs"] / max(ident["n_gpus"], 1) for v in summ.values()]
        link_gbs = statistics.median(percard) if percard else None
        contention = statistics.median([v["decode_ms_per_gb"] for v in summ.values()
                                        if v.get("decode_ms_per_gb")])
    if link_gbs is None:
        raise SystemExit(
            f"no transport measurement at {cpath}. Run:\n"
            "  .venv/bin/python bench/pcie_contention.py --tiers 8,2 --round-mb 9.7 "
            "--out bench/profile_out/phase1b_safe_qd1.json")

    # ---- routing constants -------------------------------------------------
    dpath = Path(args.distinct)
    if not dpath.is_file():
        raise SystemExit(
            f"no routing measurement at {dpath}. Run:\n"
            "  .venv/bin/python bench/measure_distinct_experts.py")
    dj = json.loads(dpath.read_text())["per_tier"]

    # The chosen mechanism's own costs, measured rather than assumed: the
    # in-graph gather achieves slightly less than a raw DMA, and the streamed
    # bytes still need a GPU-side repack into the cutlass layout.
    gpu_ms_per_expert, mech_note = 0.0, "not measured"
    mpath = Path(args.mechanism)
    if mpath.is_file():
        mj = json.loads(mpath.read_text())
        gpu_ms_per_expert = float(mj.get("repack", {}).get("ms_median", 0.0))
        a = mj.get("A") or {}
        if a.get("gbs"):
            link_gbs = min(link_gbs, float(a["gbs"]))
            mech_note = (f"gather {a['ms_per_expert']:.3f} ms/expert, "
                         f"repack {gpu_ms_per_expert:.3f} ms/expert")

    bytes_gb = args.expert_mb / 1024.0
    print(f"\nlink {link_gbs:.1f} GB/s per card, contention {contention:.2f} ms/GB, "
          f"expert {args.expert_mb:.1f} MB; mechanism: {mech_note}")

    # ---- what it implies, per tier ----------------------------------------
    print(f"\n{'tier':>5} {'CPU ms/layer':>13} {'D':>6} {'reuse':>6} {'stream k':>9} "
          f"{'step ms':>9} {'-> ms':>8} {'speedup':>8} {'bound':>6}")
    decisions = {}
    for t in tiers:
        if str(t) not in dj:
            continue
        d = dj[str(t)]
        c = Constants(
            cpu_ms_per_layer=curve[t]["cpu_ms_per_layer"],
            units_per_layer=d["U_mean"],
            distinct_per_layer=d["D_mean"],
            bytes_per_expert_gb=bytes_gb,
            link_gbs=link_gbs,
            contention_ms_per_gb=contention,
            gpu_ms_per_expert=gpu_ms_per_expert,
        )
        p = predict_step(c, args.n_layers, curve[t]["ms_per_step"])
        be = break_even_cpu_ms_per_layer(c)
        decisions[t] = {**p, "break_even_cpu_ms_per_layer": be,
                        "D": d["D_mean"], "U": d["U_mean"], "reuse": d["reuse"]}
        print(f"{t:>5} {c.cpu_ms_per_layer:>13.3f} {d['D_mean']:>6.2f} "
              f"{d['reuse']:>6.2f} {p['k']:>9} {p['step_ms_before']:>9.2f} "
              f"{p['step_ms_after']:>8.2f} {p['speedup']:>8.3f}x {p['bound_by']:>6}")
        if p["k"] == 0:
            print(f"        -> stream nothing: needs {be:.3f} ms/layer of CPU work "
                  f"to pay, has {c.cpu_ms_per_layer:.3f}")

    profile = {
        "version": 1,
        "ts": time.time(),
        "host": ident,
        "model": args.model,
        "n_layers": args.n_layers,
        "verify_tokens": args.verify_tokens,
        "bytes_per_expert_per_card_gb": bytes_gb,
        "link_gbs_per_card": link_gbs,
        "contention_ms_per_gb": contention,
        "gpu_ms_per_expert": gpu_ms_per_expert,
        "mechanism": mech_note,
        "cpu_curve": {str(k): v for k, v in curve.items()},
        "floor_ms_per_step": floor["ms_per_step"],
        "routing": dj,
        "decisions": {str(k): v for k, v in decisions.items()},
    }
    out = Path(args.out or
               f"bench/profile_out/hybrid_profile.{ident['hostname']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    print(f"\nwrote {out}")

    streaming_tiers = [t for t, d in decisions.items() if d["k"] > 0]
    if not streaming_tiers:
        print("\nThis machine streams nothing at any measured tier. That is a valid\n"
              "answer, not a failure: the CPU pole is smaller than one transfer.")
    else:
        best = max(streaming_tiers, key=lambda t: decisions[t]["speedup"])
        print(f"\nBest tier for streaming here: top{best} at "
              f"{decisions[best]['speedup']:.3f}x (stream {decisions[best]['k']}/"
              f"{decisions[best]['D']:.1f} distinct experts per layer).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
