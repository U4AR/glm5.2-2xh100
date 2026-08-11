"""Paired accuracy report over bench/accuracy_ladder.sh's per-row result files.

Absolute scores at n=60 are blunt: a 6-point standard error swallows anything
short of a collapse. The PAIRED statistics are the instrument -- the rows ran
the same questions in the same order at temperature 0, so most questions land
identically and only the disagreements carry information.

Reported per row, against the REF anchor (hybrid, every expert genuine):

  score        LiveBench score, 0-1, zebra with partial credit
  delta        score - REF score
  paired CI    95% bootstrap interval on the PER-QUESTION difference; if it
               straddles 0 the row is not distinguishable from the anchor at
               this sample size, and that is the honest statement
  W/L          questions where this row scored strictly better / worse
  sign p       two-sided sign test on those W/L, exact binomial
  agree        fraction of questions whose EXTRACTED ANSWER is identical to
               REF's -- the sharpest signal here, since a substitution that
               changes nothing about the output cannot have cost accuracy
  toks         mean generated tokens; a degraded row often rambles instead of
               answering, which the score alone does not show
  trunc        questions that produced NO extractable answer while spending
               >=95% of the token budget -- i.e. scored 0 for running out of
               room, not for reasoning wrongly. A prior zebra run lost a
               question exactly this way (7523 tok, pred=None). These zeros are
               real LiveBench zeros, but they move with verbosity rather than
               with expert substitution, so a row that differs ONLY here has
               not been shown to reason worse.

    .venv/bin/python bench/accuracy_report.py /path/to/acc_ladder
"""
import json
import math
import os
import random
import sys

ORDER = ["REF", "FLOOR", "CACHE", "S4", "PF1"]


def load(d):
    rows = {}
    if not os.path.isdir(d):
        return rows
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        row = name[:-5]
        recs = json.load(open(os.path.join(d, name)))
        rows[row] = {r["id"]: r for r in recs}
    return rows


def norm(p):
    return " ".join(str(p or "").strip().lower().split())


def boot_ci(diffs, n=10000, seed=0):
    """Bootstrap CI on the mean paired difference. Resamples QUESTIONS, which is
    the unit that was randomised; resampling anything else would understate it."""
    if not diffs:
        return 0.0, 0.0
    rnd = random.Random(seed)
    k = len(diffs)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rnd.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def sign_p(w, l):
    """Exact two-sided binomial test at p=0.5 on the win/loss questions."""
    n = w + l
    if n == 0:
        return 1.0
    lo = min(w, l)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "acc_ladder"
    rows = load(d)
    if not rows:
        print(f"no result files in {d}")
        return 1
    names = [r for r in ORDER if r in rows] + sorted(set(rows) - set(ORDER))
    ref = rows.get("REF")

    # Only questions every row actually answered -- a row that errored on a
    # question must not be scored on a different set than the others.
    common = set.intersection(*[set(rows[n]) for n in names])
    print(f"rows: {', '.join(names)}    questions in common: {len(common)}")
    if ref:
        tasks = sorted({ref[q]["task"] for q in common})
        print("per task: " + ", ".join(
            f"{t}={sum(1 for q in common if ref[q]['task'] == t)}" for t in tasks))
    print()

    # The token budget the rows were run under, not the largest answer seen --
    # inferring it from the data would flag a short unanswered question as a
    # truncation on a run where nothing actually hit the cap.
    maxtok = int(os.environ.get("MAXTOK", "8000"))

    def trunc(r, qs):
        """No answer extracted AND the budget was essentially spent."""
        return sum(1 for q in qs if r[q].get("pred") in (None, "")
                   and r[q].get("gen_toks", 0) >= 0.95 * maxtok)

    hdr = (f"{'row':7s} {'score':>7s} {'delta':>7s} {'paired 95% CI':>17s} "
           f"{'W/L':>7s} {'sign p':>7s} {'agree':>7s} {'toks':>6s} {'trunc':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        r = rows[n]
        qs = sorted(common)
        sc = sum(r[q]["score"] for q in qs) / len(qs)
        tk = sum(r[q].get("gen_toks", 0) for q in qs) / len(qs)
        tr = trunc(r, qs)
        if ref is None or n == "REF":
            print(f"{n:7s} {sc*100:6.1f}% {'-':>7s} {'-':>17s} {'-':>7s} "
                  f"{'-':>7s} {'-':>7s} {tk:6.0f} {tr:6d}")
            continue
        diffs = [r[q]["score"] - ref[q]["score"] for q in qs]
        lo, hi = boot_ci(diffs)
        w = sum(1 for x in diffs if x > 0)
        l = sum(1 for x in diffs if x < 0)
        agree = sum(1 for q in qs if norm(r[q].get("pred")) == norm(ref[q].get("pred")))
        print(f"{n:7s} {sc*100:6.1f}% {sum(diffs)/len(diffs)*100:+6.1f}% "
              f"[{lo*100:+6.1f},{hi*100:+6.1f}]  {w:3d}/{l:<3d} "
              f"{sign_p(w, l):7.3f} {agree/len(qs)*100:6.1f}% {tk:6.0f} {tr:6d}")

    # Per task, absolute only -- the paired stats above are the comparison, and
    # splitting 60 questions three ways leaves too few for a CI worth printing.
    if ref:
        tasks = sorted({ref[q]["task"] for q in common})
        print("\nper-task score")
        print(f"{'row':7s}" + "".join(f"{t:>18s}" for t in tasks))
        for n in names:
            line = f"{n:7s}"
            for t in tasks:
                qs = [q for q in common if ref[q]["task"] == t]
                line += f"{sum(rows[n][q]['score'] for q in qs)/len(qs)*100:17.1f}%"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
