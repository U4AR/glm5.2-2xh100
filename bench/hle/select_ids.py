#!/usr/bin/env python3
"""Turn a run_hle.py results file into an id list for the next stage.

The two-stage border search:

  stage 1   run a wide pool at the FAST tier (top2)
  stage 2   re-run only what top2 missed at the STRONG tier (top8)

Questions where top8 passes and top2 fails are the border -- the ones the expert
substitution actually costs you. Questions both tiers miss are simply beyond the
model, and questions both pass are free; neither discriminates, so neither is
worth spending decode time on.

    # everything top2 got wrong -> stage 2 input
    python bench/hle/select_ids.py bench/hle/results/stage1_top2.json --failed > ids.txt

    # the fractional ones (0 < pass rate < 1) when a stage used --samples > 1
    python bench/hle/select_ids.py results.json --fractional > ids.txt

    # the final verdict: compare two stages question by question
    python bench/hle/select_ids.py stage1_top2.json --compare stage2_top8.json
"""
import argparse
import collections
import json
import sys


def pass_rates(path):
    d = json.load(open(path))
    by = collections.defaultdict(list)
    for r in d["records"]:
        if "err" in r:
            continue
        by[r["id"]].append(bool(r["correct"]))
    return {k: sum(v) / len(v) for k, v in by.items()}, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--failed", action="store_true", help="pass rate == 0")
    ap.add_argument("--passed", action="store_true", help="pass rate == 1")
    ap.add_argument("--fractional", action="store_true", help="0 < pass rate < 1")
    ap.add_argument("--compare", default=None, help="second results file (strong tier)")
    a = ap.parse_args()

    rates, d = pass_rates(a.results)

    if a.compare:
        strong, d2 = pass_rates(a.compare)
        meta = {q["id"]: q for q in d["questions"]}
        rows = []
        for qid in sorted(set(rates) & set(strong)):
            rows.append((qid, rates[qid], strong[qid],
                         meta.get(qid, {}).get("category", "?"),
                         meta.get(qid, {}).get("raw_subject", "?")))
        w_fast = d["config"]["model"]
        w_slow = d2["config"]["model"]
        print(f"{'id':26s} {w_fast:>12s} {w_slow:>12s}  verdict   subject")
        border = both_fail = both_pass = regress = 0
        for qid, f, s, cat, sub in rows:
            if f < 1 and s == 1:
                v, border = "BORDER", border + 1
            elif f == 0 and s == 0:
                v, both_fail = "too hard", both_fail + 1
            elif f == 1 and s == 1:
                v, both_pass = "too easy", both_pass + 1
            else:
                v, regress = "fast wins", regress + 1
            print(f"{qid:26s} {f*100:11.0f}% {s*100:11.0f}%  {v:9s} {sub[:28]}")
        print(f"\nborder={border}  too hard (both fail)={both_fail}  "
              f"too easy (both pass)={both_pass}  fast-tier wins={regress}")
        return

    want = [q for q, r in sorted(rates.items())
            if (a.failed and r == 0) or (a.passed and r == 1)
            or (a.fractional and 0 < r < 1)]
    if not (a.failed or a.passed or a.fractional):
        sys.exit("pick one of --failed / --passed / --fractional / --compare")
    for q in want:
        print(q)
    print(f"# {len(want)} of {len(rates)}", file=sys.stderr)


if __name__ == "__main__":
    main()
