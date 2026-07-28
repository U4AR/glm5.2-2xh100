#!/usr/bin/env python3
"""Aggregate the cost-decomposition ladder into one attributable table.

Each rung adds exactly one mechanism to the rung below it, so the drop between
adjacent rungs is that mechanism's price:

    A frozen   nothing
    B count    + per-step in-graph demand counters (no ticks)
    C decide   + full selection every 32 steps, ZERO moves allowed
    D ram      + RAM<->SSD movement
    D0 nopf    D with the prefetch lookahead off
    E gpu_inc  + RAM<->GPU movement via the stable-slot swap
    F gpu_full + RAM<->GPU movement via the full restage

Throughput is reported as the median of three median-of-12 blocks taken after
warm-up, against a measured within-boot noise floor of sd 0.57 tok/s -- so a
gap under ~1.2 tok/s between rungs of the same pass is not resolvable, and any
gap at all needs to survive both passes to count.

Accept length is reported beside it because tok/s = accept_len / step_time:
if a rung loses throughput while accept_len falls, the cost is degraded draft
quality rather than time spent blocked, and no `took=` accounting can see it.

Run: .venv/bin/python experiments/expert_tiering_ssd/summarize_decompose.py
"""
import glob
import os
import re
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/decompose.log"
ORDER = ["A_frozen", "B_count", "C_decide", "D_ram", "D0_nopf", "E_gpu_inc",
         "F_gpu_full"]


def parse_runlog(path):
    """Steady-state accept length and server-reported throughput from a server
    log. The first 20% of decode lines are dropped: they cover the ramp, not
    the configuration."""
    acc, thr = [], []
    for line in open(path, errors="ignore"):
        m = re.search(r'accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)', line)
        if m:
            acc.append(float(m.group(1)))
            thr.append(float(m.group(2)))
    if len(acc) < 30:
        return None, None
    k = len(acc) // 5
    return statistics.mean(acc[k:]), statistics.mean(thr[k:])


def parse_phases(path):
    """Mean ms per phase of a tier visit, plus the w4afp8 stream/post split."""
    ph = {}
    for line in open(path, errors="ignore"):
        p = re.search(r'\[kt-tier-prof\] layer=\d+ (?:NOCHANGE )?total=[\d.]+ms (.*)', line)
        if p:
            for seg in p.group(1).split():
                k, _, v = seg.partition('=')
                ph.setdefault(k, []).append(float(v))
    w = {}
    for line in open(path, errors="ignore"):
        m = re.search(r'\[kt-w4afp8-phases\] experts=(\S+) stream=([\d.]+)ms '
                      r'post\+overlay=([\d.]+)ms', line)
        if m:
            w.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3))))
    return ph, w


# Throughput blocks, per pass and rung.
runs = {}
cur = None
for line in open(LOG, errors="ignore"):
    m = re.match(r'=== \[(p\d)\] (\S+) ::', line)
    if m:
        cur = (m.group(1), m.group(2))
        runs.setdefault(cur, [])
        continue
    m = re.match(r'\s+c\d median ([\d.]+) tok/s', line)
    if m and cur:
        runs[cur].append(float(m.group(1)))

passes = sorted({k[0] for k in runs})
print(f"{'rung':12s}" + "".join(f"{p+' tok/s':>14s}" for p in passes)
      + f"{'accept':>9s}{'probe':>8s}")

probes = {}
cur = None
for line in open(LOG, errors="ignore"):
    m = re.match(r'=== \[(p\d)\] (\S+) ::', line)
    if m:
        cur = (m.group(1), m.group(2)); continue
    m = re.match(r"\s+probe '?(.*?)'?\s*$", line)
    if m and cur:
        probes[cur] = m.group(1)[:6]

for rung in ORDER:
    cells = ""
    for p in passes:
        v = runs.get((p, rung))
        cells += f"{statistics.median(v):14.2f}" if v else f"{'-':>14s}"
    accs = []
    for p in passes:
        f = f"logs/dc_{p}_{rung}.log"
        if os.path.exists(f):
            a, _ = parse_runlog(f)
            if a:
                accs.append(a)
    a = f"{statistics.mean(accs):9.3f}" if accs else f"{'-':>9s}"
    pr = probes.get((passes[0], rung), "-")
    print(f"{rung:12s}{cells}{a}{pr:>8s}")

print("\nphase profile (mean ms per visit, only rungs that tick):")
for f in sorted(glob.glob("logs/dc_*.log")):
    ph, w = parse_phases(f)
    if not ph:
        continue
    tag = os.path.basename(f)[3:-4]
    keys = ["d2h", "select", "bcast", "diff", "gpu_incr", "gpu_full", "evict",
            "roi", "promote", "sync", "log"]
    body = "  ".join(f"{k}={statistics.mean(ph[k]):.1f}" for k in keys if k in ph)
    print(f"  {tag:16s} {body}")
    for experts, vals in sorted(w.items()):
        s = statistics.mean(v[0] for v in vals)
        po = statistics.mean(v[1] for v in vals)
        print(f"  {'':16s}   w4afp8 experts={experts}: stream={s:.1f}ms "
              f"post+overlay={po:.1f}ms  (n={len(vals)})")

print("\nNoise floor: within-boot sd 0.57 tok/s (9 median-of-12 blocks).")
print("Gaps under ~1.2 tok/s within a pass are not resolvable; gaps that do not")
print("reproduce in BOTH passes are boot-to-boot variation, not the mechanism.")
