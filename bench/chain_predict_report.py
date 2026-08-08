#!/usr/bin/env python3
"""Read bench/profile_out/chain_predict.rank0.json and print the comparison.

Two numbers per cell, and they answer different questions:

  recall    the user's "percentage of correct experts": of layer L+d's genuine
            top-K, how many did the prediction name. Diagnostic.
  layer     whole-layer coverage: the fraction of CPU-touching layer-calls where
            EVERY non-resident expert the layer needed was predicted. This is
            what a prefetcher is actually paid in -- a layer that misses one
            expert still submits, still syncs, and still pays the contention
            regime, so partial credit is worth very little.

`fetch` is the distinct non-resident experts named per layer-call, i.e. what the
prediction would cost to honour in bytes. A superset (P > K) buys coverage with
link duty, so the two must be read together.
"""

import json
import os
import sys

DEF = "bench/profile_out/chain_predict.rank0.json"

ORDER = ["persist", "prevlayer", "direct", "renorm", "post",
         "chain_shared", "chain", "chain_drop", "chain_drop_raw",
         "chain_post", "chain_exact"]
BLURB = {
    "persist": "this layer's own experts, last token   (free)",
    "prevlayer": "layer L's experts, carried forward  (free)",
    "direct": "h_L -> gate_{L+d}                (shipped)",
    "renorm": "post_ln_{L+d}(r_L)               (no propagation)",
    "post": "+ layer L's REAL MoE update      (free bound on step 1)",
    "chain_shared": "walk, shared expert only         (nearly free)",
    "chain": "walk, SEARCH for substitutes     (costs 6.55 ms)",
    "chain_drop": "walk, DROP + renormalise          (costs 1.55 ms)",
    "chain_drop_raw": "walk, DROP, do NOT renormalise   (same cost)",
    "chain_post": "walk, but step 1 is REAL         (= the shipped mechanism)",
    "chain_exact": "walk, genuine full routing       (unshippable ceiling)",
}


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEF
    if not os.path.exists(path):
        print(f"no such file: {path}")
        return 1
    d = json.load(open(path))
    rows = d["rows"]
    arms = [a for a in ORDER if any(r["arm"] == a for r in rows)]
    arms += sorted({r["arm"] for r in rows} - set(arms))
    depths = sorted({r["depth"] for r in rows})
    Ps = sorted({r["P"] for r in rows})

    err = d.get("resid_rel_err")
    print(f"steps {d['steps']}  stride {d['stride']}  true set = top-{d['K']}  "
          f"rank {d['rank']}")
    print(f"residual reconstruction rel-err: "
          f"{'n/a' if err is None else f'{err:.2e}'}")
    if d.get("failed"):
        print(f"FAILED ARMS: {d['failed']}")
    print()
    for a in arms:
        print(f"  {a:<13} {BLURB.get(a, '')}")
    print()

    idx = {(r["arm"], r["depth"], r["P"]): r for r in rows}
    dd = [x for x in depths if x > 0]

    def table(P0, title, cell, width=9):
        print(title)
        print(f"{'arm':<13}{'d=0':>{width}}  " +
              "".join(f"{'d=' + str(x):>{width}}" for x in dd))
        for a in arms:
            row = [f"{cell(idx.get((a, 0, P0))):>{width}}", "  "]
            row += [f"{cell(idx.get((a, x, P0))):>{width}}" for x in dd]
            print(f"{a:<13}" + "".join(row))
        print()

    for P0 in Ps:
        tag = "EXACT SET" if P0 == d["K"] else "superset"
        print(f"=== P={P0} ({tag}) " + "=" * 46)
        table(P0, "-- recall: % of the layer's genuine top-K that was named "
                  "(d=0 must be 100.0)",
              lambda r: "--" if not r else
              f"{100.0 * r['hit'] / max(r['tot'], 1):.1f}")
        table(P0, "-- whole-layer coverage: % of CPU-touching calls covered "
                  "COMPLETELY  <-- the payoff",
              lambda r: "--" if not r else
              f"{100.0 * r['full'] / max(r['active'], 1):.1f}")
        table(P0, "-- coverage BLENDED with this layer's previous-token demand "
                  "(free, costs bytes)",
              lambda r: "--" if not r else
              f"{100.0 * r.get('bfull', 0) / max(r['active'], 1):.1f}")
        table(P0, "-- fetch: distinct NON-RESIDENT experts named per "
                  "layer-call (the byte cost)",
              lambda r: "--" if not r else
              f"{r['set'] / max(r['calls'], 1):.2f}")

    ref = idx.get((arms[-1], dd[0], Ps[0]))
    if ref:
        print(f"for scale: the layer genuinely needs "
              f"{ref['need'] / max(ref['calls'], 1):.2f} non-resident experts "
              f"per CPU-touching call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
