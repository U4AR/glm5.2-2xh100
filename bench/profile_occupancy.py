#!/usr/bin/env python3
"""Profile the *running* GLM-5.2 server: where does a decoded token's time go?

Answers two questions in one pass, against a live server (nothing is restarted):

  1. Occupancy of every resource in the loop -- GPU compute, GPU (HBM) bandwidth,
     CPU compute, host DRAM bandwidth, SSD bandwidth.
  2. Average time per token, split by the stage that consumed it, and which of
     those stages overlap.

Method notes (this box has no uncore PMU and perf_event_paranoid=4, so there are
no hardware bandwidth counters):
  * GPU compute / HBM occupancy come from NVML duty-cycle sampling, cross-checked
    against an analytic bytes-per-token model of the weights actually read.
  * Host DRAM peak is *measured* with a STREAM-triad (bench/stream.c), then the
    decode-time demand is derived from the CPU-expert weight traffic.
  * SSD bandwidth is read straight out of /proc/diskstats.
  * The per-stage split comes from the server's own torch profiler
    (/start_profile), so it is real kernel time, not a guess.

Usage:
    .venv/bin/python bench/profile_occupancy.py                 # full run
    .venv/bin/python bench/profile_occupancy.py --tokens 400
    .venv/bin/python bench/profile_occupancy.py --no-stream     # skip STREAM
    .venv/bin/python bench/profile_occupancy.py --no-torch      # skip torch profiler

Writes <out>/results.json and <out>/report.html (a self-contained diagram).
Pass --no-report for JSON only; bench/profile_report.py can re-render it later.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import urllib.request

REPO = Path(__file__).resolve().parent.parent
SECTOR = 512  # /proc/diskstats counts 512B sectors regardless of physical size


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sh(cmd: str, timeout: int = 60) -> str:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------
# counter sampling
# --------------------------------------------------------------------------
@dataclass
class Sample:
    t: float
    cpu_busy_jiffies: float
    cpu_total_jiffies: float
    per_node_busy: dict
    disk: dict            # dev -> (read_sectors, write_sectors)
    mem_available_kb: int
    pgmajfault: int
    pswpin: int
    pswpout: int


class Sampler(threading.Thread):
    """Samples /proc counters + NVML at a fixed cadence, timestamped so any
    sub-window (e.g. the decode-only phase) can be sliced out afterwards."""

    def __init__(self, disks, period=0.2):
        super().__init__(daemon=True)
        self.period = period
        self.disks = disks
        self.samples: list[Sample] = []
        self.gpu: list[tuple] = []           # (t, idx, sm%, mem%, W, MHz)
        self._stop = threading.Event()
        self._nvsmi = None

    # ---- /proc readers -------------------------------------------------
    def _read_cpu(self):
        busy = total = 0.0
        per_node = {0: 0.0, 1: 0.0}
        node_of = self.node_of
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                if parts[0] == "cpu":
                    v = [float(x) for x in parts[1:]]
                    total = sum(v)
                    busy = total - v[3] - v[4]        # minus idle, iowait
                else:
                    cid = int(parts[0][3:])
                    v = [float(x) for x in parts[1:]]
                    n = node_of.get(cid, 0)
                    per_node[n] = per_node.get(n, 0.0) + (sum(v) - v[3] - v[4])
        return busy, total, per_node

    def _read_disk(self):
        out = {}
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) > 9 and p[2] in self.disks:
                    out[p[2]] = (int(p[5]), int(p[9]))   # sectors read, written
        return out

    def _read_mem(self):
        avail = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1])
                    break
        maj = swin = swout = 0
        with open("/proc/vmstat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                if k == "pgmajfault":
                    maj = int(v)
                elif k == "pswpin":
                    swin = int(v)
                elif k == "pswpout":
                    swout = int(v)
        return avail, maj, swin, swout

    # ---- NVML via nvidia-smi streaming --------------------------------
    def _start_nvsmi(self):
        ms = int(self.period * 1000)
        self._nvsmi = subprocess.Popen(
            "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,"
            f"power.draw,clocks.sm --format=csv,noheader,nounits -lms {ms}",
            shell=True, stdout=subprocess.PIPE, text=True,
        )

        def pump():
            for line in self._nvsmi.stdout:
                try:
                    p = [x.strip() for x in line.split(",")]
                    self.gpu.append(
                        (time.time(), int(p[0]), float(p[1]), float(p[2]),
                         float(p[3]), float(p[4]))
                    )
                except Exception:
                    pass

        threading.Thread(target=pump, daemon=True).start()

    def run(self):
        self.node_of = numa_map()
        self._start_nvsmi()
        while not self._stop.is_set():
            busy, total, per_node = self._read_cpu()
            avail, maj, swin, swout = self._read_mem()
            self.samples.append(
                Sample(time.time(), busy, total, per_node,
                       self._read_disk(), avail, maj, swin, swout)
            )
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._nvsmi:
            self._nvsmi.terminate()

    # ---- window analysis ----------------------------------------------
    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x.t <= t1]
        if len(s) < 2:
            return {}
        a, b = s[0], s[-1]
        dt = b.t - a.t
        d_total = b.cpu_total_jiffies - a.cpu_total_jiffies
        d_busy = b.cpu_busy_jiffies - a.cpu_busy_jiffies
        ncpu = os.cpu_count()

        disk = {}
        for dev in self.disks:
            if dev in a.disk and dev in b.disk:
                dr = (b.disk[dev][0] - a.disk[dev][0]) * SECTOR
                dw = (b.disk[dev][1] - a.disk[dev][1]) * SECTOR
                disk[dev] = {
                    "read_MB": dr / 1e6, "write_MB": dw / 1e6,
                    "read_MBs": dr / 1e6 / dt, "write_MBs": dw / 1e6 / dt,
                }

        g = [x for x in self.gpu if t0 <= x[0] <= t1]
        gpus = {}
        for idx in sorted({x[1] for x in g}):
            rows = [x for x in g if x[1] == idx]
            gpus[idx] = {
                "sm_util_mean": statistics.mean(r[2] for r in rows),
                "sm_util_p95": sorted(r[2] for r in rows)[int(0.95 * (len(rows) - 1))],
                "mem_util_mean": statistics.mean(r[3] for r in rows),
                "mem_util_p95": sorted(r[3] for r in rows)[int(0.95 * (len(rows) - 1))],
                "power_W_mean": statistics.mean(r[4] for r in rows),
                "sm_clock_MHz": statistics.mean(r[5] for r in rows),
                "n": len(rows),
            }

        node_busy = {}
        for n in a.per_node_busy:
            node_busy[n] = ((b.per_node_busy[n] - a.per_node_busy[n]) / d_total * ncpu
                            if d_total else 0)

        return {
            "seconds": dt,
            "cpu_busy_frac": d_busy / d_total if d_total else 0,
            "cpu_cores_busy": d_busy / d_total * ncpu if d_total else 0,
            "cpu_cores_busy_per_node": node_busy,
            "n_cores": ncpu,
            "disk": disk,
            "gpu": gpus,
            "mem_available_MB": b.mem_available_kb / 1024,
            "pgmajfault": b.pgmajfault - a.pgmajfault,
            "pswpin": b.pswpin - a.pswpin,
            "pswpout": b.pswpout - a.pswpout,
        }


def numa_map() -> dict:
    m = {}
    for nd in glob.glob("/sys/devices/system/node/node*/cpulist"):
        n = int(re.search(r"node(\d+)", nd).group(1))
        for part in open(nd).read().strip().split(","):
            if "-" in part:
                a, b = part.split("-")
                for c in range(int(a), int(b) + 1):
                    m[c] = n
            elif part:
                m[int(part)] = n
    return m


# --------------------------------------------------------------------------
# STREAM triad -- measured host DRAM peak
# --------------------------------------------------------------------------
STREAM_C = r"""
/* Two kernels:
     read  -- pure streaming read (8 B/elem).  This is the right ceiling for
              MoE weight streaming, which never writes the weights back.
     triad -- classic STREAM triad. Counted at 32 B/elem, not 24: the store to
              c[] pulls the line in first (write-allocate / RFO), so the DRAM
              actually moves 4 words per element, not 3.
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char** argv){
    size_t N = (argc>1)? (size_t)atoll(argv[1]) : (size_t)200000000;
    int iters = (argc>2)? atoi(argv[2]) : 8;
    double *a=aligned_alloc(64,N*8),*b=aligned_alloc(64,N*8),*c=aligned_alloc(64,N*8);
    #pragma omp parallel for
    for(size_t i=0;i<N;i++){a[i]=1.0;b[i]=2.0;c[i]=0.0;}
    double best_t=0, best_r=0;
    for(int it=0; it<iters; it++){
        double t0=omp_get_wtime();
        #pragma omp parallel for
        for(size_t i=0;i<N;i++) c[i]=a[i]+3.0*b[i];
        double dt=omp_get_wtime()-t0;
        double gbs = 32.0*(double)N/dt/1e9;
        if(gbs>best_t) best_t=gbs;

        /* integer accumulate: FP '+' is not associative so gcc refuses to
           vectorise a double reduction without -ffast-math, which would make
           this latency-bound instead of bandwidth-bound. */
        unsigned long long s=0; unsigned long long *ai=(unsigned long long*)a;
        t0=omp_get_wtime();
        #pragma omp parallel for reduction(+:s)
        for(size_t i=0;i<N;i++) s+=ai[i];
        dt=omp_get_wtime()-t0;
        gbs = 8.0*(double)N/dt/1e9;
        if(gbs>best_r) best_r=gbs;
        if(s==42) printf("x");   /* keep the reduction alive */
    }
    printf("%.1f %.1f\n", best_t, best_r);
    return 0;
}
"""


def stream_peak(outdir: Path, threads: int | None = None,
                gb: float = 1.0) -> tuple[float, float] | None:
    """Build + run the bandwidth probe. Returns (triad_GBs, read_GBs)."""
    src = outdir / "stream.c"
    exe = outdir / "stream"
    if not exe.exists():
        src.write_text(STREAM_C)
        rc = subprocess.run(
            f"gcc -O3 -march=native -fopenmp {src} -o {exe}",
            shell=True, capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print("  ! stream build failed:", rc.stderr[-300:], file=sys.stderr)
            return None
    n = int(gb * 1e9 / 8)
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([str(exe), str(n), "8"], capture_output=True, text=True,
                       env=env, timeout=300)
    try:
        triad, read = r.stdout.strip().split()
        return float(triad), float(read)
    except Exception:
        return None


# --------------------------------------------------------------------------
# decode workload
# --------------------------------------------------------------------------
def decode_run(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Streaming completion.

    One SSE chunk == one *forward step*, not one token: with MTP/NEXTN a step
    emits `accept_length` tokens at once. We therefore report step time and
    token time separately, using the server's own usage counter for the true
    token count.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_start = time.time()
    stamps, text, usage = [], [], {}
    resp = post(f"{url}/v1/chat/completions", body)
    for raw in resp:
        line = raw.decode().strip()
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
        piece = ch.get("content") or ch.get("reasoning_content") or ""
        if piece:
            stamps.append(time.time())
            text.append(piece)
    t_end = time.time()
    if len(stamps) < 3:
        raise RuntimeError("decode produced too few steps to profile")

    ttft = stamps[0] - t_start
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    gaps_s = sorted(gaps)
    decode_s = stamps[-1] - stamps[0]
    n_steps = len(stamps)
    n_tok = usage.get("completion_tokens") or n_steps
    # tokens produced after the first step (the window the gaps cover)
    tok_after_first = max(1, n_tok - n_tok / n_steps)
    accept = n_tok / n_steps

    return {
        "t_start": t_start,
        "t_first_token": stamps[0],
        "t_end": t_end,
        "ttft_s": ttft,
        "usage": usage,
        "n_steps": n_steps,
        "n_tokens": n_tok,
        "accept_length": accept,
        "decode_seconds": decode_s,
        "ms_per_step_mean": statistics.mean(gaps),
        "ms_per_step_median": statistics.median(gaps),
        "ms_per_step_p10": gaps_s[int(0.10 * (len(gaps_s) - 1))],
        "ms_per_step_p90": gaps_s[int(0.90 * (len(gaps_s) - 1))],
        "ms_per_token_mean": decode_s * 1000 / tok_after_first,
        "tok_per_s": tok_after_first / decode_s if decode_s else 0,
        "steps_per_s": (n_steps - 1) / decode_s if decode_s else 0,
        "gaps_ms": gaps,
        "sample_text": "".join(text)[:200],
    }


# --------------------------------------------------------------------------
# torch profiler -> per-stage breakdown
# --------------------------------------------------------------------------
# Ordered classification rules; first match wins.
KERNEL_RULES = [
    ("comm (NCCL all-reduce)", r"nccl|all_?reduce|allgather|reduce_scatter"),
    ("attention (NSA / MLA)",  r"nsa|flash|mla|fmha|attn|attention|index_?topk|paged"),
    ("MoE experts (GPU int4)", r"w4a8|w4afp8|cutlass.*(moe|group)|moe|grouped_gemm|group_gemm|silu_and_mul|topk_softmax|moe_align|scatter|gather"),
    ("dense GEMM (attn/MLP)",  r"gemm|cutlass|sm90|sm80|cublas|deepgemm|matmul|dot|nvjet"),
    ("norm / rope / elemwise", r"norm|rope|rotary|elementwise|vectorized|add|mul|cast|convert|copy_|fill|memset"),
    ("sampling / draft logic", r"sampl|softmax|argmax|topk|sort|cumsum|gather_next|verify"),
    ("memcpy H<->D",           r"memcpy"),
]


def classify(name: str) -> str:
    low = name.lower()
    for label, pat in KERNEL_RULES:
        if re.search(pat, low):
            return label
    return "other GPU kernels"


def torch_profile(url: str, model: str, prompt: str, tokens: int,
                  outdir: Path, num_steps: int) -> dict:
    """Drive the server's own torch profiler over a short decode and parse the
    resulting chrome trace into per-stage GPU time."""
    trace_dir = outdir / "torch_trace"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    before = set(glob.glob(str(trace_dir / "*")))
    try:
        post(f"{url}/start_profile", {
            "output_dir": str(trace_dir),
            "num_steps": num_steps,
            "activities": ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": False,
        }, timeout=60).read()
    except Exception as e:
        return {"error": f"start_profile failed: {e}"}

    run = decode_run(url, model, prompt, tokens)

    # profiler auto-stops after num_steps; make sure, then wait for files
    try:
        post(f"{url}/stop_profile", {}, timeout=120).read()
    except Exception:
        pass

    files = []
    for _ in range(120):
        time.sleep(1.0)
        files = [f for f in glob.glob(str(trace_dir / "**"), recursive=True)
                 if f not in before and f.endswith((".json", ".json.gz"))]
        if files and all(os.path.getsize(f) > 0 for f in files):
            time.sleep(2.0)   # let the writer finish flushing
            break
    if not files:
        return {"error": "no trace files produced", "decode": run}

    per_rank = {}
    for f in sorted(files):
        try:
            per_rank[os.path.basename(f)] = parse_trace(f)
        except Exception as e:
            per_rank[os.path.basename(f)] = {"error": str(e)}
    return {"decode": run, "ranks": per_rank, "files": files}


def parse_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    evs = doc.get("traceEvents", [])

    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu = [e for e in evs if e.get("cat") in gpu_cats and e.get("ph") == "X"]
    if not gpu:
        return {"error": "no GPU events in trace"}

    # per-stage GPU time
    stage = {}
    for e in gpu:
        lbl = classify(e.get("name", ""))
        s = stage.setdefault(lbl, {"us": 0.0, "count": 0})
        s["us"] += e.get("dur", 0)
        s["count"] += 1

    # busy wall time = union of kernel intervals on the compute streams
    def union(intervals):
        if not intervals:
            return 0.0
        intervals.sort()
        tot, cs, ce = 0.0, *intervals[0]
        for s, e in intervals[1:]:
            if s > ce:
                tot += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        return tot + (ce - cs)

    iv = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in gpu]
    busy_us = union(iv)
    span_us = max(e["ts"] + e.get("dur", 0) for e in gpu) - min(e["ts"] for e in gpu)

    # CPU-side: what the launching thread spent, and how long it blocked
    cpu = {}
    for e in evs:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("cuda_runtime", "cuda_driver"):
            n = e.get("name", "")
            c = cpu.setdefault(n, {"us": 0.0, "count": 0})
            c["us"] += e.get("dur", 0)
            c["count"] += 1
    cpu_top = dict(sorted(cpu.items(), key=lambda kv: -kv[1]["us"])[:12])

    sync_us = sum(v["us"] for k, v in cpu.items()
                  if re.search(r"synchronize|EventQuery|StreamWait", k, re.I))

    # count decode steps via the graph launches / kernel replays
    graph_launch = sum(v["count"] for k, v in cpu.items()
                       if re.search(r"GraphLaunch", k, re.I))

    return {
        "gpu_busy_ms": busy_us / 1000,
        "gpu_span_ms": span_us / 1000,
        "gpu_busy_frac": busy_us / span_us if span_us else 0,
        "gpu_kernel_ms_total": sum(s["us"] for s in stage.values()) / 1000,
        "stages": {k: {"ms": v["us"] / 1000, "count": v["count"]}
                   for k, v in sorted(stage.items(), key=lambda kv: -kv[1]["us"])},
        "cpu_runtime_top": {k: {"ms": v["us"] / 1000, "count": v["count"]}
                            for k, v in cpu_top.items()},
        "cpu_blocked_ms": sync_us / 1000,
        "cuda_graph_launches": graph_launch,
        "n_gpu_events": len(gpu),
    }


# --------------------------------------------------------------------------
# analytic bytes/token model
# --------------------------------------------------------------------------
def analytic_model(cfg: dict, args) -> dict:
    H = cfg["hidden_size"]
    Im = cfg["moe_intermediate_size"]
    L = cfg["num_hidden_layers"]
    dense_L = cfg.get("first_k_dense_replace", 0)
    moe_L = L - dense_L
    E = cfg["n_routed_experts"]
    K = cfg["num_experts_per_tok"]
    shared = cfg.get("n_shared_experts", 0) or 0
    I_dense = cfg["intermediate_size"]

    # one routed expert = gate + up + down
    p_expert = 3 * H * Im
    b_expert_int4 = p_expert * 0.5 + (p_expert / 128) * 2      # +fp16 group scales
    b_expert_fp8 = p_expert * 1.0 + (p_expert / 128) * 2

    gpu_experts = args.gpu_experts
    # coverage: fraction of the top-K activations that land on a GPU-resident
    # expert. hotcore placement skews this above the uniform gpu_experts/E.
    cov_uniform = gpu_experts / E
    cov = args.coverage if args.coverage is not None else cov_uniform

    k_gpu = K * cov
    k_cpu = K - k_gpu

    # per token, per MoE layer
    gpu_moe_bytes = k_gpu * b_expert_int4
    cpu_moe_bytes = k_cpu * b_expert_int4

    # dense/attention weights live on GPU, fp8, split over TP
    attn_params = (
        H * cfg.get("q_lora_rank", 0) + cfg.get("q_lora_rank", 0) *
        cfg["num_attention_heads"] * (cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"])
        + H * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"])
        + cfg["kv_lora_rank"] * cfg["num_attention_heads"] *
        (cfg["qk_nope_head_dim"] + cfg["v_head_dim"])
        + cfg["num_attention_heads"] * cfg["v_head_dim"] * H
    )
    shared_params = shared * 3 * H * Im
    dense_mlp_params = 3 * H * I_dense

    gpu_bytes_per_token = (
        moe_L * (gpu_moe_bytes + (attn_params + shared_params) * 1.0)
        + dense_L * (attn_params + dense_mlp_params) * 1.0
    )
    cpu_bytes_per_token = moe_L * cpu_moe_bytes

    # FLOPs (2 per MAC)
    gpu_flops = 2 * (
        moe_L * (k_gpu * p_expert + attn_params + shared_params)
        + dense_L * (attn_params + dense_mlp_params)
    )
    cpu_ops = 2 * moe_L * k_cpu * p_expert       # int8 MAC ops

    return {
        "moe_layers": moe_L, "dense_layers": dense_L,
        "params_per_expert_M": p_expert / 1e6,
        "bytes_per_expert_int4_MB": b_expert_int4 / 1e6,
        "gpu_experts_resident": gpu_experts,
        "coverage_assumed": cov,
        "coverage_uniform": cov_uniform,
        "experts_on_gpu_per_token_per_layer": k_gpu,
        "experts_on_cpu_per_token_per_layer": k_cpu,
        "gpu_bytes_per_token_MB": gpu_bytes_per_token / 1e6,
        "cpu_bytes_per_token_MB": cpu_bytes_per_token / 1e6,
        "gpu_flops_per_token_G": gpu_flops / 1e9,
        "cpu_ops_per_token_G": cpu_ops / 1e9,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how a modern out-of-order CPU "
    "pipeline works, covering fetch, decode, rename, dispatch, execution ports, "
    "the reorder buffer, load/store queues, and retirement. Be thorough."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="default: first model on server")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--torch-tokens", type=int, default=120)
    ap.add_argument("--torch-steps", type=int, default=200)
    ap.add_argument("--idle-seconds", type=float, default=8.0)
    ap.add_argument("--gpu-experts", type=int,
                    default=int(os.environ.get("GPU_EXPERTS", 60)))
    ap.add_argument("--coverage", type=float, default=None,
                    help="measured fraction of top-K hits served by GPU-resident "
                         "experts (default: uniform gpu_experts/n_experts)")
    ap.add_argument("--hbm-peak-gbs", type=float, default=3900.0,
                    help="per-GPU HBM peak GB/s (H100 NVL = 3900)")
    ap.add_argument("--gpu-fp8-tflops", type=float, default=1979.0,
                    help="per-GPU dense FP8 tensor TFLOP/s")
    ap.add_argument("--settle-tries", type=int, default=4,
                    help="warmup decodes used to reach a steady state")
    ap.add_argument("--settle-tol-pct", type=float, default=8.0)
    ap.add_argument("--cpu-ghz", type=float, default=3.0,
                    help="sustained all-core clock, for the int8 VNNI peak")
    ap.add_argument("--cpu-threads", type=int,
                    default=int(os.environ.get("KT_CPUINFER", 72)),
                    help="kt cpuinfer worker threads")
    ap.add_argument("--canary-threads", type=int, default=4,
                    help="threads for the DRAM contention probe (0 disables)")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-report", action="store_true",
                    help="skip rendering report.html (JSON only)")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--out", default=str(REPO / "bench" / "profile_out"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"ts": time.time(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "argv": sys.argv, "args": vars(args)}}

    # ---------------- preflight ----------------
    print("[1/7] preflight")
    models = get_json(f"{args.url}/v1/models")["data"]
    model = args.model or models[0]["id"]
    res["meta"]["model"] = model
    print(f"      model = {model}")

    srv = get_json(f"{args.url}/get_server_info")
    keep = {k: srv.get(k) for k in
            ("model_path", "max_total_tokens", "tp_size", "attention_backend",
             "speculative_algorithm", "speculative_num_steps",
             "speculative_num_draft_tokens", "kt_num_gpu_experts", "kt_method",
             "kt_cpuinfer", "mem_fraction_static", "context_length")}
    res["server"] = {k: v for k, v in keep.items() if v is not None}
    if srv.get("kt_num_gpu_experts"):
        args.gpu_experts = srv["kt_num_gpu_experts"]
    if srv.get("kt_cpuinfer"):
        args.cpu_threads = srv["kt_cpuinfer"]

    # tier sentinel (KEEP mode) if present
    tf = os.environ.get("KT_TOPK_MODE_FILE", "/tmp/kt_topk_mode")
    res["server"]["topk_mode"] = (Path(tf).read_text().strip()
                                  if Path(tf).exists() else None)

    cfg_path = Path(srv.get("model_path", "")) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    res["model_config"] = {k: cfg.get(k) for k in
                           ("num_hidden_layers", "hidden_size", "intermediate_size",
                            "moe_intermediate_size", "n_routed_experts",
                            "num_experts_per_tok", "n_shared_experts",
                            "first_k_dense_replace", "num_nextn_predict_layers")}

    res["host"] = {
        "cpu_model": sh("lscpu | awk -F: '/Model name/{print $2}' | head -1"),
        "n_cores": os.cpu_count(),
        "numa_nodes": len(set(numa_map().values())),
        "mem_total_GB": int(sh("awk '/MemTotal/{print $2}' /proc/meminfo")) / 1e6,
        "gpus": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
        "perf_paranoid": sh("cat /proc/sys/kernel/perf_event_paranoid"),
    }

    disks = [d for d in sh("lsblk -dn -o NAME").split()
             if d.startswith(("nvme", "sd"))]
    res["host"]["disks"] = disks

    # ---------------- idle baseline ----------------
    print(f"[2/7] idle baseline ({args.idle_seconds:.0f}s)")
    s = Sampler(disks)
    s.start()
    time.sleep(args.idle_seconds)
    t_idle_end = time.time()
    res["idle"] = s.window(t_idle_end - args.idle_seconds + 0.5, t_idle_end)

    # ---------------- STREAM ceiling ----------------
    # Ordering matters in both directions. STREAM only reads a true ceiling on a
    # quiet box, so it goes first -- but its multi-GB allocation forces reclaim
    # on a box with a few GB free (the rest is expert weights + page cache),
    # which depresses decode afterwards. Hence: probe first, then settle until
    # decode is reproducible again, and only then take the headline numbers.
    if args.no_stream:
        res["stream"] = {"skipped": True}
        print("[3a/7] STREAM peak: skipped")
    else:
        print("[3a/7] STREAM probe (host DRAM ceiling, server idle)")
        best_t = best_r = 0.0
        for _ in range(2):                     # best of two attempts
            a = stream_peak(out) or (0, 0)
            best_t, best_r = max(best_t, a[0] or 0), max(best_r, a[1] or 0)
        half = stream_peak(out, threads=max(1, os.cpu_count() // 2)) or (0, 0)
        res["stream"] = {
            "triad_GBs_all_cores": best_t, "read_GBs_all_cores": best_r,
            "triad_GBs_half_cores": half[0], "read_GBs_half_cores": half[1],
            "peak_GBs_all_cores": best_r,      # read peak = weight-stream ceiling
            "threads_all": os.cpu_count(),
        }
        limited = ("BW-limited" if half[1] and best_r and half[1] > 0.9 * best_r
                   else "thread-limited")
        print(f"      read {best_r:.1f} GB/s | triad {best_t:.1f} GB/s (all cores); "
              f"read {half[1]:.1f} GB/s @half -> {limited}")

    # ---------------- settle ----------------
    print("[3b/7] settling until decode is reproducible")
    prev, settled = None, False
    for i in range(args.settle_tries):
        w = decode_run(args.url, model, DEFAULT_PROMPT, 80)
        rate = w["tok_per_s"]
        drift = abs(rate - prev) / prev * 100 if prev else None
        print(f"      warmup {i+1}: {rate:5.2f} tok/s" +
              (f"  ({drift:.0f}% vs previous)" if drift is not None else ""))
        if drift is not None and drift < args.settle_tol_pct:
            settled = True
            break
        prev = rate
        time.sleep(2.0)
    res["settled"] = settled
    res["settle_tok_s"] = prev if prev else None
    res["settle_tokens"] = 80
    if not settled:
        print("      ! never settled; headline numbers may not be steady-state")

    # ---------------- loaded decode ----------------
    print(f"[3/7] decode run ({args.tokens} tokens)")
    run = decode_run(args.url, model, DEFAULT_PROMPT, args.tokens)
    res["decode"] = {k: v for k, v in run.items() if k != "gaps_ms"}
    res["decode"]["gaps_ms"] = run["gaps_ms"]
    # occupancy over the decode-only window (excludes prefill/TTFT)
    res["loaded"] = s.window(run["t_first_token"] + 0.3, run["t_end"])
    res["prefill"] = s.window(run["t_start"], run["t_first_token"])
    print(f"      {run['tok_per_s']:.2f} tok/s  "
          f"({run['ms_per_token_mean']:.1f} ms/token), TTFT {run['ttft_s']*1000:.0f} ms")

    # ---------------- torch profiler ----------------
    if args.no_torch:
        res["torch"] = {"skipped": True}
        print("[4/7] torch profiler: skipped")
    else:
        print(f"[4/7] torch profiler ({args.torch_tokens} tokens) -- server keeps running")
        try:
            res["torch"] = torch_profile(args.url, model, DEFAULT_PROMPT,
                                         args.torch_tokens, out, args.torch_steps)
        except Exception as e:
            res["torch"] = {"error": repr(e)}
        if "error" in res.get("torch", {}):
            print("      ! ", res["torch"]["error"])

    # ---------------- DRAM contention canary ----------------
    # The analytic bytes/token model assumes a coverage figure. This measures
    # the pressure directly: how much read bandwidth a small probe can still
    # get while the server decodes, vs. while it is idle. A big drop means the
    # CPU expert path really is leaning on the memory controller.
    if args.no_stream or args.canary_threads <= 0:
        res["canary"] = {"skipped": True}
        print("[5/7] DRAM contention canary: skipped")
    else:
        nthr = args.canary_threads
        print(f"[5/7] DRAM contention canary ({nthr} threads)")
        idle_c = stream_peak(out, threads=nthr, gb=1.0) or (None, None)
        holder = {}

        def _bg():
            try:
                holder["run"] = decode_run(args.url, model, DEFAULT_PROMPT,
                                           args.tokens)
            except Exception as e:
                holder["err"] = repr(e)

        th = threading.Thread(target=_bg, daemon=True)
        th.start()
        time.sleep(4.0)                       # get past prefill
        load_c = stream_peak(out, threads=nthr, gb=1.0) or (None, None)
        th.join(timeout=300)
        res["canary"] = {
            "threads": nthr,
            "idle_read_GBs": idle_c[1], "loaded_read_GBs": load_c[1],
            "idle_triad_GBs": idle_c[0], "loaded_triad_GBs": load_c[0],
            "read_retained_pct": (load_c[1] / idle_c[1] * 100)
            if idle_c[1] and load_c[1] else None,
            "decode_tok_s_during_canary": holder.get("run", {}).get("tok_per_s"),
        }
        print(f"      probe read BW: {idle_c[1]} GB/s idle -> {load_c[1]} GB/s "
              f"under load ({res['canary']['read_retained_pct']:.0f}% retained)")

    # ---------------- recovery check ----------------
    # The probes above disturb the page cache. Re-measure so the report can say
    # whether the headline decode number is still reproducible afterwards.
    try:
        time.sleep(3.0)
        chk = decode_run(args.url, model, DEFAULT_PROMPT, 80)
        res["recheck"] = {"tok_per_s": chk["tok_per_s"],
                          "ms_per_step_mean": chk["ms_per_step_mean"],
                          "ttft_s": chk["ttft_s"]}
        # Compare against the settle warmup, which used the SAME token count.
        # Comparing an 80-token recheck to the 400-token headline run would
        # just measure the context-length effect, not drift.
        base = res.get("settle_tok_s") or run["tok_per_s"]
        drift = abs(chk["tok_per_s"] - base) / base * 100
        res["recheck"]["drift_pct"] = drift
        if drift > 20:
            res["recheck"]["warning"] = (
                "decode speed moved >20% between the start and end of the run; "
                "the box was not in a steady state")
            print(f"      ! recheck drift {drift:.0f}% "
                  f"({base:.1f} -> {chk['tok_per_s']:.1f} tok/s at 80 tokens)")
        else:
            print(f"      recheck ok: {chk['tok_per_s']:.2f} tok/s "
                  f"({drift:.0f}% drift)")
    except Exception as e:
        res["recheck"] = {"error": repr(e)}

    s.stop()

    # ---------------- analysis ----------------
    print("[7/7] analysis")
    res["analytic"] = analytic_model(cfg, args) if cfg else {}
    res["occupancy"] = occupancy(res, args)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print_report(res)
    print(f"\nwrote {out/'results.json'}")

    # ---------------- diagram ----------------
    if args.no_report:
        print(f"render the diagram with:  .venv/bin/python bench/profile_report.py "
              f"{out/'results.json'}")
    else:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import profile_report
            html_path = out / "report.html"
            html_path.write_text(profile_report.build(res))
            print(f"wrote {html_path}")
        except Exception as e:
            print(f"! could not render the diagram: {e!r}")
            print(f"  retry with: .venv/bin/python bench/profile_report.py "
                  f"{out/'results.json'}")


def occupancy(res: dict, args) -> dict:
    """Fold measurements + analytics into one occupancy table."""
    o = {}
    dec = res.get("decode", {})
    load = res.get("loaded", {})
    an = res.get("analytic", {})
    tps = dec.get("tok_per_s", 0)
    ngpu = len(load.get("gpu", {})) or 1

    # --- GPU compute
    sm = [g["sm_util_mean"] for g in load.get("gpu", {}).values()]
    # Use the *compute* rank (the lower duty cycle). The other rank sits at
    # ~90% because it spins inside the all-reduce waiting for rank 0 -- taking
    # the max would report spin-wait as GPU utilisation.
    fracs = [r["gpu_busy_frac"] for r in (res.get("torch", {}) or {})
             .get("ranks", {}).values()
             if isinstance(r, dict) and "gpu_busy_frac" in r]
    gpu_busy_frac = min(fracs) if fracs else None
    achieved_tflops = an.get("gpu_flops_per_token_G", 0) * tps / 1e3 / ngpu
    o["gpu_compute"] = {
        "duty_cycle_pct": statistics.mean(sm) if sm else None,
        "kernel_busy_pct": gpu_busy_frac * 100 if gpu_busy_frac else None,
        "achieved_TFLOPs_per_gpu": achieved_tflops,
        "peak_TFLOPs_per_gpu": args.gpu_fp8_tflops,
        "flops_util_pct": achieved_tflops / args.gpu_fp8_tflops * 100
        if args.gpu_fp8_tflops else None,
    }

    # --- GPU bandwidth
    mem = [g["mem_util_mean"] for g in load.get("gpu", {}).values()]
    gbs = an.get("gpu_bytes_per_token_MB", 0) * tps / 1e3 / ngpu   # GB/s per GPU
    o["gpu_bandwidth"] = {
        "duty_cycle_pct": statistics.mean(mem) if mem else None,
        "achieved_GBs_per_gpu": gbs,
        "peak_GBs_per_gpu": args.hbm_peak_gbs,
        "bw_util_pct": gbs / args.hbm_peak_gbs * 100,
    }

    # --- CPU compute
    cores = load.get("cpu_cores_busy", 0)
    ncores = load.get("n_cores", os.cpu_count())
    # Zen4 AVX-512 VNNI: one 512-bit VPDPBUSD/cycle = 64 int8 MACs = 128 ops
    ghz = args.cpu_ghz
    peak_tops = args.cpu_threads * ghz * 128 / 1e3
    ach_tops = an.get("cpu_ops_per_token_G", 0) * tps / 1e3
    o["cpu_compute"] = {
        "cores_busy": cores,
        "n_cores": ncores,
        "core_util_pct": cores / ncores * 100 if ncores else None,
        "cores_busy_per_node": load.get("cpu_cores_busy_per_node"),
        "achieved_TOPS_int8": ach_tops,
        "peak_TOPS_int8": peak_tops,
        "tops_util_pct": ach_tops / peak_tops * 100 if peak_tops else None,
        "note": "core_util_pct counts cores as busy while they stall on DRAM "
                "(and while kt worker threads spin), so it is occupancy, not "
                "useful arithmetic -- compare with tops_util_pct",
    }

    # --- host DRAM bandwidth
    st = res.get("stream", {}) or {}
    peak = max([v for v in (st.get("read_GBs_all_cores"),
                            st.get("read_GBs_half_cores"),
                            st.get("triad_GBs_all_cores")) if v] or [0]) or None
    dram = an.get("cpu_bytes_per_token_MB", 0) * tps / 1e3
    can = res.get("canary", {}) or {}
    o["ram_bandwidth"] = {
        "achieved_GBs": dram,
        "measured_peak_GBs": peak,
        "bw_util_pct": (dram / peak * 100) if peak else None,
        "canary_read_retained_pct": can.get("read_retained_pct"),
        "note": "achieved_GBs is analytic (bytes/token x tok/s) at the assumed "
                "GPU-expert coverage; the canary is the direct measurement",
    }

    # --- SSD
    tot_r = sum(d["read_MBs"] for d in load.get("disk", {}).values())
    tot_w = sum(d["write_MBs"] for d in load.get("disk", {}).values())
    o["ssd_bandwidth"] = {
        "read_MBs": tot_r, "write_MBs": tot_w,
        "per_device": load.get("disk", {}),
        "major_faults": load.get("pgmajfault"),
        "swap_in_pages": load.get("pswpin"), "swap_out_pages": load.get("pswpout"),
    }

    o["budget"] = time_budget(res)
    return o


def time_budget(res: dict) -> dict:
    """Split one forward step into GPU-busy vs CPU-critical-path.

    The kt hybrid MoE submits the CPU experts asynchronously and runs the GPU
    experts concurrently, so a layer costs max(cpu, gpu). Whatever the GPU is
    *not* busy with during a step is therefore time the step spent waiting on
    the CPU expert path.
    """
    dec = res.get("decode", {})
    t = res.get("torch", {}) or {}
    step_ms = dec.get("ms_per_step_mean")
    if not step_ms or "ranks" not in t:
        return {"note": "torch profiler unavailable; no stage split"}

    # rank 0 owns the CPU experts, so it is the compute rank; rank 1 mostly
    # spins in NCCL waiting for it.
    ranks = {k: v for k, v in t["ranks"].items() if "gpu_busy_ms" in v}
    if not ranks:
        return {"note": "no usable trace"}
    r0k = sorted(ranks, key=lambda k: ranks[k]["gpu_busy_frac"])[0]
    r0 = ranks[r0k]
    n_steps = t["decode"]["n_steps"]
    prof_step_ms = r0["gpu_span_ms"] / n_steps
    scale = step_ms / prof_step_ms if prof_step_ms else 1.0

    gpu_busy_step = r0["gpu_busy_ms"] / n_steps * scale
    stages = {k: v["ms"] / n_steps * scale for k, v in r0["stages"].items()}
    accept = dec.get("accept_length", 1.0)

    return {
        "compute_rank": r0k,
        "profiled_step_ms": prof_step_ms,
        "measured_step_ms": step_ms,
        "profiler_overhead_x": prof_step_ms / step_ms if step_ms else None,
        "gpu_busy_ms_per_step": gpu_busy_step,
        "gpu_busy_pct_of_step": r0["gpu_busy_frac"] * 100,
        "cpu_critical_ms_per_step": step_ms - gpu_busy_step,
        "cpu_critical_pct_of_step": (step_ms - gpu_busy_step) / step_ms * 100,
        "stage_ms_per_step": stages,
        "accept_length": accept,
        "ms_per_token": step_ms / accept if accept else None,
        "gpu_busy_ms_per_token": gpu_busy_step / accept if accept else None,
        "cpu_critical_ms_per_token": (step_ms - gpu_busy_step) / accept if accept else None,
        "spin_rank": {k: v["gpu_busy_frac"] * 100 for k, v in ranks.items()},
    }


def bar(pct, width=28):
    if pct is None:
        return "  n/a"
    n = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * n + "·" * (width - n)


def print_report(res):
    o = res["occupancy"]
    d = res["decode"]
    print("\n" + "=" * 78)
    print(f"  {res['meta']['model']}  |  {res['meta']['date']}")
    print("=" * 78)
    print(f"  {d['tok_per_s']:.2f} tok/s   {d['ms_per_token_mean']:.1f} ms/token   "
          f"|  forward step {d['ms_per_step_mean']:.1f} ms "
          f"(median {d['ms_per_step_median']:.1f}, "
          f"p10 {d['ms_per_step_p10']:.1f} / p90 {d['ms_per_step_p90']:.1f}), "
          f"{d['accept_length']:.2f} tok/step")
    print(f"  {d['n_tokens']} tokens in {d['n_steps']} steps, "
          f"TTFT {d['ttft_s']*1000:.0f} ms")
    print("-" * 78)
    rows = [
        ("GPU compute (duty cycle)", o["gpu_compute"]["duty_cycle_pct"],
         f"{o['gpu_compute']['achieved_TFLOPs_per_gpu']:.1f} of "
         f"{o['gpu_compute']['peak_TFLOPs_per_gpu']:.0f} TFLOP/s"),
        ("GPU compute (FLOP util)", o["gpu_compute"]["flops_util_pct"], ""),
        ("GPU bandwidth (duty cycle)", o["gpu_bandwidth"]["duty_cycle_pct"],
         f"{o['gpu_bandwidth']['achieved_GBs_per_gpu']:.0f} of "
         f"{o['gpu_bandwidth']['peak_GBs_per_gpu']:.0f} GB/s"),
        ("GPU bandwidth (of peak)", o["gpu_bandwidth"]["bw_util_pct"], ""),
        ("CPU cores occupied", o["cpu_compute"]["core_util_pct"],
         f"{o['cpu_compute']['cores_busy']:.1f} of {o['cpu_compute']['n_cores']} cores"),
        ("CPU arithmetic (int8 VNNI)", o["cpu_compute"]["tops_util_pct"],
         f"{o['cpu_compute']['achieved_TOPS_int8']:.2f} of "
         f"{o['cpu_compute']['peak_TOPS_int8']:.0f} TOPS"),
        ("RAM bandwidth (read peak)", o["ram_bandwidth"]["bw_util_pct"],
         f"{o['ram_bandwidth']['achieved_GBs']:.0f} of "
         f"{o['ram_bandwidth']['measured_peak_GBs'] or 0:.0f} GB/s"),
        ("SSD bandwidth", None,
         f"{o['ssd_bandwidth']['read_MBs']:.1f} MB/s read, "
         f"{o['ssd_bandwidth']['write_MBs']:.1f} MB/s write, "
         f"{o['ssd_bandwidth']['major_faults']} major faults"),
    ]
    for name, pct, note in rows:
        p = f"{pct:5.1f}%" if pct is not None else "   -- "
        print(f"  {name:<28} {bar(pct)} {p}  {note}")
    c = res.get("canary", {})
    if c.get("read_retained_pct"):
        print(f"  {'DRAM headroom (probe)':<28} {bar(100-c['read_retained_pct'])} "
              f"{100-c['read_retained_pct']:5.1f}%  a {c['threads']}-thread probe "
              f"keeps {c['idle_read_GBs']:.0f}->{c['loaded_read_GBs']:.0f} GB/s "
              f"under load")
    print("-" * 78)

    g = res.get("loaded", {}).get("gpu", {})
    if g:
        print("  per GPU: " + " | ".join(
            f"gpu{k} sm {v['sm_util_mean']:.0f}% mem {v['mem_util_mean']:.1f}% "
            f"{v['power_W_mean']:.0f}W" for k, v in g.items()))

    b = o.get("budget", {})
    if "gpu_busy_ms_per_step" in b:
        print(f"  TIME BUDGET  (one {b['measured_step_ms']:.0f} ms forward step "
              f"= {b['accept_length']:.2f} tokens = "
              f"{b['ms_per_token']:.1f} ms/token)")
        print(f"    GPU busy (overlapped)      {b['gpu_busy_ms_per_step']:7.1f} ms"
              f"  {b['gpu_busy_pct_of_step']:5.1f}%")
        print(f"    CPU experts (critical path){b['cpu_critical_ms_per_step']:7.1f} ms"
              f"  {b['cpu_critical_pct_of_step']:5.1f}%")
        print(f"    (profiler slowed steps {b['profiler_overhead_x']:.2f}x; "
              f"stage times below rescaled to the unprofiled step)")
        print("    GPU stages, ms per step:")
        for k, v in sorted(b["stage_ms_per_step"].items(), key=lambda kv: -kv[1]):
            print(f"      {k:<28} {v:6.2f} ms")
        print("    GPU kernel duty per rank: " +
              ", ".join(f"{k.split('-TP-')[-1][0]}={v:.0f}%"
                        for k, v in b["spin_rank"].items()) +
              "   (the high one is NCCL spin-wait, not work)")
    print("=" * 78)


if __name__ == "__main__":
    main()
