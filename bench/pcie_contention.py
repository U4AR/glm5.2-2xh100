#!/usr/bin/env python3
"""Phase 1b: does PCIe H2D survive while the CPU expert path is decoding?

The streaming-hybrid headline (EXPERT_STREAMING_PLAN.md 1.3/1.4) adds ~107 GB/s
of "otherwise idle" PCIe delivery on top of the ~138 GB/s the CPU expert path
already pulls out of DRAM. Both come out of the same memory system: the DMA
engine reads the pinned host pages the CPU cores are also hammering. If that
sum does not hold, the entire design is capped before a line of it is written.

So this measures the two directions together, which is the only way the answer
is meaningful:

  1. H2D GB/s per card with the server idle              (the 53.4 GB/s claim)
  2. H2D GB/s per card *while* the server decodes at a   (contention on DMA)
     given expert tier
  3. the decode's own ms/step during that same window    (contention on decode)

A win requires both to survive. Bandwidth that only exists when the CPU is idle
is not a budget streaming can spend, and bandwidth bought by slowing the CPU
expert path is double counting.

Each GPU sits on its own NUMA node here, so one worker process per card is
pinned (cpunodebind+membind) to that card's node -- an unbound run measures
cross-socket traffic instead of the transfer streaming would actually issue.

    .venv/bin/python bench/pcie_contention.py --tiers 8,2 \
        --out bench/profile_out/phase1b_contention.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import request

PROMPT = (
    "Write a detailed technical essay about how modern CPUs and GPUs differ in "
    "their approach to parallelism, memory hierarchy, and scheduling."
)

# One expert's TP-sharded weights per layer, per card (plan 1.3). The realistic
# streaming transfer is this size, not a 256 MB block: DMA efficiency at 9.7 MB
# is what the design actually gets to spend.
EXPERT_MB = 9.7


# --------------------------------------------------------------------------
# worker: measured H2D on one card, meant to be run under numactl
# --------------------------------------------------------------------------
def worker(args) -> int:
    import torch

    torch.cuda.set_device(args.gpu)
    chunk = int(args.chunk_mb * 1024 * 1024)
    n_chunks = max(int(args.buf_mb * 1024 * 1024) // chunk, 4)
    host = torch.empty(n_chunks * chunk, dtype=torch.uint8, pin_memory=True)
    host.random_(0, 255)  # fault the pages in before timing
    # Two landing slots, as the real thing would double-buffer.
    dev = [torch.empty(chunk, dtype=torch.uint8, device=f"cuda:{args.gpu}")
           for _ in range(2)]
    stream = torch.cuda.Stream(device=args.gpu)

    # A round is a fixed batch of chunk copies; the host buffer is walked
    # linearly and wraps, so no chunk is served from cache twice in a row.
    # Queue depth, not chunk size, is what a latency-critical transfer waits
    # behind: a round is submitted back to back and only then synchronized.
    per_round = max(int(args.round_mb * 1024 * 1024 // chunk), 1)
    if args.start_at:
        time.sleep(max(args.start_at - time.time(), 0))

    windows = []
    i = 0
    t_begin = time.perf_counter()
    deadline = t_begin + args.seconds
    with torch.cuda.stream(stream):
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            if args.spin_only:
                # Control: burn the same core in the same synchronize-style spin
                # without moving a byte, to separate "the DMA cost bandwidth" from
                # "the benchmark stole a core from the CPU expert threads".
                while time.perf_counter() - t0 < 0.02:
                    pass
            else:
                for _ in range(per_round):
                    src = host[(i % n_chunks) * chunk:(i % n_chunks + 1) * chunk]
                    dev[i % 2].copy_(src, non_blocking=True)
                    i += 1
                stream.synchronize()
            dt = time.perf_counter() - t0
            windows.append(per_round * chunk / dt / 1e9)
            # Duty cycle < 1 idles the link between rounds. Streaming would not
            # saturate PCIe continuously, so the exchange rate has to be read at
            # the demand the design would actually place on it, not only at 100%.
            if args.duty < 1.0:
                time.sleep(dt * (1.0 / args.duty - 1.0))
    total_s = time.perf_counter() - t_begin

    out = {
        "gpu": args.gpu,
        "chunk_mb": args.chunk_mb,
        "buf_mb": n_chunks * chunk / 1024 / 1024,
        "duty": args.duty,
        "round_mb": args.round_mb,
        "spin_only": args.spin_only,
        "windows": len(windows),
        # gbs_median is the rate while the link is busy; gbs_effective is the
        # rate averaged over the whole window, i.e. the demand actually placed.
        "gbs_effective": i * chunk / total_s / 1e9,
        "gbs_median": statistics.median(windows) if not args.spin_only else 0.0,
        "gbs_mean": statistics.mean(windows) if not args.spin_only else 0.0,
        "gbs_min": min(windows) if not args.spin_only else 0.0,
        "gbs_max": max(windows) if not args.spin_only else 0.0,
        "bytes_moved_gb": i * chunk / 1e9,
        "seconds": total_s,
    }
    print("WORKER_JSON " + json.dumps(out))
    return 0


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------
def gpu_numa_nodes() -> dict[int, int]:
    nodes = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout
    except Exception:
        return nodes
    for line in out.strip().splitlines():
        idx, bus = [x.strip() for x in line.split(",")]
        # nvidia-smi prints 00000000:0X:00.0; sysfs wants 0000:0X:00.0
        addr = bus.lower()
        if len(addr.split(":")[0]) > 4:
            addr = addr[len(addr.split(":")[0]) - 4:]
        p = Path(f"/sys/bus/pci/devices/{addr}/numa_node")
        if p.is_file():
            n = int(p.read_text().strip())
            if n >= 0:
                nodes[int(idx)] = n
    return nodes


def run_workers(gpus, nodes, seconds, chunk_mb, buf_mb, start_at,
                duty=1.0, spin_only=False, round_mb=64.0) -> list[dict]:
    """One process per card, started against a common wall-clock instant."""
    have_numactl = shutil.which("numactl") is not None
    procs = []
    for g in gpus:
        cmd = []
        if have_numactl and g in nodes:
            cmd += ["numactl", f"--cpunodebind={nodes[g]}", f"--membind={nodes[g]}"]
        cmd += [sys.executable, __file__, "--worker", "--gpu", str(g),
                "--seconds", str(seconds), "--chunk-mb", str(chunk_mb),
                "--buf-mb", str(buf_mb), "--start-at", f"{start_at:.3f}",
                "--duty", str(duty), "--round-mb", str(round_mb)]
        if spin_only:
            cmd.append("--spin-only")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    results = []
    for p in procs:
        out, err = p.communicate()
        got = None
        for line in out.splitlines():
            if line.startswith("WORKER_JSON "):
                got = json.loads(line[len("WORKER_JSON "):])
        if got is None:
            raise RuntimeError(f"H2D worker failed:\n{err[-2000:]}")
        results.append(got)
    return sorted(results, key=lambda r: r["gpu"])


def decode_stream(url: str, model: str, tokens: int, sink: dict) -> None:
    """Stream a completion, recording a perf_counter stamp per emitted chunk.

    One SSE chunk is one accepted token; steps are recovered from the stamps,
    so the same run yields both the load and the ms/step it ran at.
    """
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = request.Request(f"{url}/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
    stamps, usage = [], {}
    try:
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
    except Exception as e:
        sink["error"] = repr(e)
    sink["stamps"] = stamps
    sink["usage"] = usage


def step_ms(stamps: list[float], lo: float | None = None,
            hi: float | None = None) -> dict:
    """Median gap between emitted tokens, optionally restricted to a window.

    Restricting to [lo, hi] is the point: it separates the steps that ran
    against live DMA from the ones before and after it.
    """
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))
            if (lo is None or stamps[i - 1] >= lo) and (hi is None or stamps[i] <= hi)]
    if len(gaps) < 5:
        return {"n": len(gaps), "ms_per_step": float("nan")}
    return {"n": len(gaps), "ms_per_step": statistics.median(gaps),
            "ms_per_step_mean": statistics.mean(gaps)}


def health(url: str) -> None:
    with request.urlopen(f"{url}/health_generate", timeout=60):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--start-at", type=float, default=0.0)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="GLM5.2")
    ap.add_argument("--tiers", default="8,2",
                    help="expert tiers to load the CPU path with")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--chunk-mb", type=float, default=EXPERT_MB)
    ap.add_argument("--round-mb", type=float, default=64.0,
                    help="bytes submitted back to back before synchronizing")
    ap.add_argument("--buf-mb", type=float, default=512.0)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--load-tokens", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--duty", type=float, default=1.0,
                    help="fraction of the window the link is kept busy")
    ap.add_argument("--spin-only", action="store_true",
                    help="control: burn the same cores, move no bytes")
    ap.add_argument("--skip-idle", action="store_true")
    ap.add_argument("--out", default="bench/profile_out/phase1b_contention.json")
    args = ap.parse_args()

    if args.worker:
        return worker(args)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    nodes = gpu_numa_nodes()
    print(f"GPU->NUMA node map: {nodes or 'unknown (running unbound)'}")
    health(args.url)

    report = {"ts": time.time(), "args": vars(args), "numa": nodes,
              "idle": [], "loaded": {}, "decode_alone": {}}

    if not args.skip_idle:
        print(f"\n--- H2D with the server idle ({args.seconds:.0f}s x {args.repeats}) ---")
        for r in range(args.repeats):
            res = run_workers(gpus, nodes, args.seconds, args.chunk_mb,
                              args.buf_mb, time.time() + 1.0,
                              args.duty, args.spin_only, args.round_mb)
            report["idle"].append(res)
            agg = sum(x["gbs_effective"] for x in res)
            print("  rep {}: {}  aggregate {:.1f} GB/s effective".format(
                r + 1,
                "  ".join(f"gpu{x['gpu']} {x['gbs_effective']:.1f} GB/s" for x in res),
                agg))

    for tier in [int(t) for t in args.tiers.split(",") if t.strip()]:
        model = f"{args.model}-top{tier}"
        print(f"\n--- tier top{tier} ---")

        # (a) decode alone, same prompt, same length: the reference the loaded
        #     decode is compared against.
        alone = []
        for r in range(args.repeats):
            sink: dict = {}
            decode_stream(args.url, model, args.load_tokens, sink)
            if sink.get("error"):
                raise RuntimeError(sink["error"])
            s = step_ms(sink["stamps"])
            alone.append(s["ms_per_step"])
            print(f"  decode alone  rep {r + 1}: {s['ms_per_step']:7.2f} ms/step "
                  f"({s['n']} steps)")
        report["decode_alone"][tier] = alone

        # (b) decode and DMA at the same time. The DMA window starts a few
        #     seconds in so it lands in steady-state decode, and both sides are
        #     read only over that window.
        loaded = []
        for r in range(args.repeats):
            sink = {}
            th = threading.Thread(target=decode_stream,
                                  args=(args.url, model, args.load_tokens, sink))
            th.start()
            time.sleep(4.0)
            t_lo = time.perf_counter()
            wall_start = time.time() + 0.5
            res = run_workers(gpus, nodes, args.seconds, args.chunk_mb,
                              args.buf_mb, wall_start, args.duty, args.spin_only,
                              args.round_mb)
            t_hi = time.perf_counter()
            # If the generation finished early the tail of the DMA window ran
            # against an idle server, which would inflate the "under load"
            # bandwidth. Record it rather than silently averaging it in.
            still_decoding = th.is_alive()
            th.join(timeout=600)
            if sink.get("error"):
                raise RuntimeError(sink["error"])
            win = step_ms(sink["stamps"], t_lo + 0.5, t_hi)
            agg = sum(x["gbs_effective"] for x in res)
            loaded.append({"h2d": res, "aggregate_gbs": agg,
                           "busy_gbs": sum(x["gbs_median"] for x in res),
                           "decode_ms_per_step": win["ms_per_step"],
                           "decode_steps_in_window": win["n"],
                           "decode_covered_window": still_decoding})
            print("  loaded       rep {}: {}  aggregate {:.1f} GB/s   "
                  "decode {:.2f} ms/step ({} steps in window){}".format(
                      r + 1,
                      "  ".join(f"gpu{x['gpu']} {x['gbs_effective']:.1f}" for x in res),
                      agg, win["ms_per_step"], win["n"],
                      "" if still_decoding else "   WARNING: decode ended inside the window"))
        report["loaded"][tier] = loaded

    # ---- verdict -----------------------------------------------------------
    idle_agg = statistics.median([sum(x["gbs_effective"] for x in rep)
                                  for rep in report["idle"]]) if report["idle"] else float("nan")
    print("\n================ Phase 1b ================")
    print(f"idle aggregate H2D:        {idle_agg:6.1f} GB/s "
          f"({idle_agg / len(gpus):.1f} GB/s per card)")
    for tier, runs in report["loaded"].items():
        load_agg = statistics.median([r["aggregate_gbs"] for r in runs])
        busy_agg = statistics.median([r["busy_gbs"] for r in runs])
        alone_ms = statistics.median(report["decode_alone"][tier])
        load_ms = statistics.median([r["decode_ms_per_step"] for r in runs])
        keep = load_agg / idle_agg * 100 if idle_agg == idle_agg else float("nan")
        slow = (load_ms / alone_ms - 1) * 100
        # The exchange rate is what the split policy actually needs: how many ms
        # of decode step does a GB of streamed weight cost? Bandwidth that is
        # paid for out of the decode's own step time is not free capacity.
        gb_per_step = load_agg * load_ms / 1000.0
        ms_per_gb = (load_ms - alone_ms) / gb_per_step if gb_per_step else float("nan")
        print(f"top{tier}: H2D under decode  {load_agg:6.1f} GB/s effective "
              f"({keep:5.1f}% of idle, {busy_agg:.1f} while busy)   "
              f"decode {alone_ms:7.2f} -> {load_ms:7.2f} ms/step ({slow:+.1f}%)")
        print(f"       moved {gb_per_step:.2f} GB per step -> "
              f"{ms_per_gb:.2f} ms of decode per GB streamed")
        report.setdefault("summary", {})[tier] = {
            "idle_aggregate_gbs": idle_agg, "loaded_aggregate_gbs": load_agg,
            "loaded_busy_gbs": busy_agg,
            "pcie_retained_pct": keep, "decode_ms_alone": alone_ms,
            "decode_ms_loaded": load_ms, "decode_slowdown_pct": slow,
            "gb_moved_per_step": gb_per_step, "decode_ms_per_gb": ms_per_gb,
            # MB / (GB/s) is already milliseconds.
            "expert_transfer_ms": (args.chunk_mb / (busy_agg / len(gpus))
                                   if busy_agg else None),
        }
        if busy_agg:
            print(f"       one {args.chunk_mb:.1f} MB expert per card at the busy "
                  f"rate: {args.chunk_mb / (busy_agg / len(gpus)):.3f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
