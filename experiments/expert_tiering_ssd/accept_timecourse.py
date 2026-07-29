#!/usr/bin/env python3
"""Is the accept-length loss under movement TRANSIENT or STEADY-STATE?

The decomposition ladder showed that ~85% of what tier movement costs is
degraded MTP draft acceptance (accept length 2.693 frozen -> 2.318 moving),
not time spent moving. Two mechanisms survive scrutiny, and they call for
completely different fixes:

  (1) STEADY-STATE. The drifted resident set is a worse model than the
      warm-start oracle set it replaced, so the fixed draft head agrees with it
      less often. Fix: don't drift (or drift toward a better objective).

  (2) TRANSIENT. Each swap briefly disrupts generation and acceptance recovers
      between ticks. Fix: make swaps gentler or rarer.

They are distinguishable from logs already on disk, two ways:

  A. Time course over the run. Transient damage that recovers would leave the
     quartile means roughly flat and close to frozen, with the loss carried by
     a few bad steps; steady-state damage puts every quartile down together.

  B. Proximity to a tier visit. If disruption is transient, accept-length
     samples logged within a few seconds AFTER a visit should be measurably
     worse than samples far from any visit. If the loss is steady-state, the
     two populations are the same.

Run: .venv/bin/python experiments/expert_tiering_ssd/accept_timecourse.py
"""
import datetime
import glob
import os
import re
import statistics
import sys

DECODE = re.compile(
    r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?accept len: ([\d.]+)'
)
TIER = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?\[kt-tier\] layer=')


def parse(path):
    acc, visits = [], []
    for line in open(path, errors="ignore"):
        m = DECODE.search(line)
        if m:
            t = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            acc.append((t, float(m.group(2))))
            continue
        m = TIER.search(line)
        if m:
            visits.append(
                datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            )
    return acc, visits


def quartiles(acc):
    """Mean accept length in each quarter of the decode samples."""
    if len(acc) < 20:
        return None
    vals = [v for _, v in acc]
    k = len(vals) // 4
    return [statistics.mean(vals[i * k:(i + 1) * k]) for i in range(4)]


def near_far(acc, visits, window=3.0):
    """Split accept samples by whether a tier visit happened in the `window`
    seconds before them. Log timestamps are 1-second resolution, so this is
    coarse -- but a transient that does not show up within 3 s of a visit is
    not a transient that matters at this tick rate."""
    if not visits:
        return None
    vs = sorted(visits)
    near, far = [], []
    j = 0
    for t, v in acc:
        while j + 1 < len(vs) and vs[j + 1] <= t:
            j += 1
        prev = vs[j] if vs[j] <= t else None
        if prev is not None and (t - prev).total_seconds() <= window:
            near.append(v)
        else:
            far.append(v)
    return near, far


print(f"{'run':18s} {'n':>6s} {'mean':>6s} {'Q1':>6s} {'Q2':>6s} {'Q3':>6s} "
      f"{'Q4':>6s} {'near':>7s} {'far':>7s} {'delta':>7s}")
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "logs/dc_p1_*.log")):
    acc, visits = parse(path)
    if len(acc) < 40:
        continue
    # Drop the first fifth: it is the warm-up ramp, not the configuration.
    acc = acc[len(acc) // 5:]
    q = quartiles(acc)
    nf = near_far(acc, visits)
    tag = os.path.basename(path)[3:-4]
    qs = "".join(f"{x:6.3f}" for x in q) if q else " " * 24
    if nf and len(nf[0]) >= 5 and len(nf[1]) >= 5:
        n, f = statistics.mean(nf[0]), statistics.mean(nf[1])
        extra = f"{n:7.3f}{f:7.3f}{n - f:+7.3f}"
    else:
        extra = f"{'-':>7s}{'-':>7s}{'-':>7s}"
    print(f"{tag:18s} {len(acc):6d} "
          f"{statistics.mean(v for _, v in acc):6.3f} {qs} {extra}")

print("\nQ1..Q4 flat and low  => steady-state: the drifted set is a worse model.")
print("Q1..Q4 recovering    => transient: swaps disrupt, generation recovers.")
print("near < far           => transient disruption localised around visits.")
print("near == far          => the loss is a property of the residency, not")
print("                        of the act of swapping.")
