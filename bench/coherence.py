#!/usr/bin/env python3
"""Score generated text for the failure modes expert substitution actually causes.

Substitution does not produce noise. It produces text that is locally fluent and
globally wrong: the model loops, repeats a phrase, drifts off the prompt, or
emits degenerate punctuation runs. `top0` reads as English at 300 characters and
falls apart at 2000, which is exactly why eyeballing a prefix is not a check.

Four signals, each chosen because it fires on a real observed failure:

  loop_score      longest run of an immediately repeated n-gram. The `top2`
                  tier's known defect is "rare loops"; `top0` loops hard.
  repeat_ratio    1 - distinct(tokens)/total. Degenerate output collapses its
                  own vocabulary long before it stops parsing.
  punct_run       longest run of non-alphanumeric tokens. The measured
                  long-context failure emitted `0,0,0,...`.
  type_token      distinct/total over a sliding window, which catches local
                  collapse that a whole-document ratio averages away.

No model, no tokenizer, no judgement -- just properties of the string, so the
same number means the same thing across boots and can be diffed.
"""
import argparse
import json
import os
import re
import sys
import zlib
from collections import Counter


def ngrams(xs, n):
    return [tuple(xs[i:i + n]) for i in range(len(xs) - n + 1)]


def longest_immediate_repeat(words, nmax=12):
    """Longest k such that some n-gram repeats back-to-back k times."""
    best = 1
    for n in range(1, min(nmax, len(words) // 2) + 1):
        i = 0
        while i + n <= len(words):
            k = 1
            while (i + (k + 1) * n <= len(words)
                   and words[i + k * n:i + (k + 1) * n] == words[i:i + n]):
                k += 1
            if k > best:
                best = k
            i += 1
    return best


def window_ttr(words, w=100):
    if len(words) < w:
        return len(set(words)) / max(len(words), 1)
    return min(len(set(words[i:i + w])) / w for i in range(0, len(words) - w, 25))


def char_signals(text):
    """Character-level signals, because the word-level ones can be EVADED.

    The first version of this scorer split on whitespace. The known-degenerate
    case turned out to be `</think>` repeated ~400 times -- 3286 characters
    containing exactly TWO whitespace characters -- so it collapsed to 3 "words"
    and scored repeat_ratio 0.000, ttr 1.000, verdict "clean". The single failure
    mode the check existed to catch was the one it could not see.

    Compressibility is the fix: it is language-agnostic, needs no tokenizer, and
    cannot be dodged by deleting spaces. Normal prose sits near 0.30-0.45; a
    repeated phrase compresses to a few percent whatever it is made of.
    """
    b = text.encode("utf-8", "replace")
    ratio = len(zlib.compress(b, 9)) / max(len(b), 1)
    runs = re.split(r"\s+", text)
    longest_nows = max((len(r) for r in runs), default=0)
    grams = ngrams(text, 8)
    distinct8 = len(set(grams)) / max(len(grams), 1)
    return {
        "zlib": round(ratio, 4),
        "longest_nows": longest_nows,
        "distinct8": round(distinct8, 3),
    }


def score(text):
    words = re.findall(r"\S+", text)
    ch = char_signals(text)
    if not words:
        return {"words": 0, "verdict": "EMPTY", **ch}
    punct_run = 0
    cur = 0
    for tok in words:
        cur = cur + 1 if not re.search(r"[A-Za-z0-9]", tok) else 0
        punct_run = max(punct_run, cur)
    rep = 1.0 - len(set(words)) / len(words)
    loop = longest_immediate_repeat(words)
    ttr = window_ttr(words)
    # Thresholds are deliberately loose: this flags "look at this", it does not
    # adjudicate. A clean sample here reads loop<=2, repeat<0.55, ttr>0.45.
    bad = []
    # CHARACTER-LEVEL FIRST -- these are the ones that cannot be evaded.
    # Thresholds are set well clear of BOTH observed cases rather than tuned to
    # split them: measured `</think>`x400 compresses to 0.006 and reads 3286
    # chars with no whitespace, while real 1200-token answers sit at 0.30-0.40
    # with runs under ~40. Anything between is worth a human look either way.
    if ch["zlib"] < 0.15:
        bad.append(f"compresses to {ch['zlib']:.3f}")
    if ch["longest_nows"] >= 400:
        bad.append(f"{ch['longest_nows']} chars w/o whitespace")
    if ch["distinct8"] < 0.10:
        bad.append(f"distinct 8-grams {ch['distinct8']:.3f}")
    # Word-level, kept as secondary: they catch looping that still has spaces.
    if loop >= 4:
        bad.append(f"loop x{loop}")
    if rep > 0.62:
        bad.append(f"repeat {rep:.2f}")
    if punct_run >= 8:
        bad.append(f"punct run {punct_run}")
    if ttr < 0.38 and len(words) >= 100:
        bad.append(f"local ttr {ttr:.2f}")
    return {
        "words": len(words),
        "loop": loop,
        "repeat_ratio": round(rep, 3),
        "punct_run": punct_run,
        "window_ttr": round(ttr, 3),
        **ch,
        "verdict": "DEGENERATE (" + ", ".join(bad) + ")" if bad else "clean",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="text files or directories")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many chars of each sample")
    a = ap.parse_args()

    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.endswith(".txt")]
        elif os.path.exists(p):
            files.append(p)
    if not files:
        print("no text files found")
        return 1

    rows = {}
    print(f"{'sample':22s} {'words':>6} {'zlib':>7} {'nows':>6} {'d8':>6} "
          f"{'loop':>5} {'rep':>6} {'ttr':>6}  verdict")
    for f in files:
        t = open(f, errors="replace").read()
        s = score(t)
        rows[os.path.basename(f)] = s
        print(f"{os.path.basename(f)[:22]:22s} {s.get('words',0):6d} "
              f"{s.get('zlib',0):7.4f} {s.get('longest_nows',0):6d} "
              f"{s.get('distinct8',0):6.3f} {s.get('loop',0):5d} "
              f"{s.get('repeat_ratio',0):6.3f} {s.get('window_ttr',0):6.3f}  "
              f"{s['verdict']}")
        if a.show:
            print("   " + t[:a.show].replace("\n", "\n   "))
            print()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
