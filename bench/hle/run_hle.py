#!/usr/bin/env python3
"""Score the running GLM-5.2 server on Humanity's Last Exam (text-only).

Why HLE: it is the only benchmark on Zhipu's own GLM-5.2 eval table that is not
saturated. Official GLM-5.2 numbers are AIME 99.2 / HMMT 94.4 / GPQA-D 91.2 /
Terminal-Bench 81.0 / SWE-bench Pro 62.1 / **HLE 40.5** (no tools). Only HLE sits
in the band where the model answers some and misses some, which is what a
top8-vs-top2 degradation A/B needs -- on a 91%-saturated benchmark the tiers are
separated by noise.

Dataset: `macabdul9/hle_text_only` (ungated mirror of the text-only split of
cais/hle, 2370 rows, canary GUID intact). cais/hle itself is gated; this avoids
the access click and contains no images, which the server cannot take anyway.

Two modes:

  probe   -- run a handful of questions with a huge token cap and report the
             generation-length distribution, so the real cap and sample count
             can be chosen from data instead of guessed.
  run     -- run a fixed question set at n samples per question per tier and
             report PER-QUESTION pass rates.

The per-question pass rate is the point. At the limit of the model's ability a
question is not pass/fail, it has a probability; one sample per tier cannot tell
a tier gap from a coin flip. Degradation shows up as pass-rate shift.

    # 1. how long do answers actually take?
    python bench/hle/run_hle.py probe --n-questions 6 --max-tokens 32000

    # 2. pick the at-the-limit questions from a pilot, then A/B them
    python bench/hle/run_hle.py run --ids-file bench/hle/selected.txt \
        --model GLM5.2-top8 --samples 3 --out bench/hle/results/top8.json

Results stream to the JSON file after every sample, so a killed run keeps its
work and `--resume` picks it up.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

DATASET = "macabdul9/hle_text_only"

# The official HLE system prompt. The rigid three-field reply is what makes
# exactMatch gradable without an LLM judge.
SYSTEM_EXACT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
SYSTEM_MC = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


# --------------------------------------------------------------------------
# answer extraction + grading
# --------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"(?:exact\s+answer|answer)\s*[::]\s*(.+)", re.I)


def extract_answer(text):
    """Last 'Exact Answer:'/'Answer:' field, else the last non-empty line."""
    hits = _ANSWER_RE.findall(text or "")
    if hits:
        return hits[-1].strip().split("\n")[0].strip()
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    return lines[-1] if lines else ""


_STRIP = r"\s*$\\\[\]{}()`*_\"'.,;:"


def _norm(s):
    """Normalise for string comparison: lowercase, drop LaTeX/markup wrappers,
    collapse whitespace. Deliberately conservative -- it is better to mark a
    right answer unmatched (and see it in the JSON) than to score a wrong one
    correct, because a lenient grader flatters whichever tier rambles more."""
    s = (s or "").strip().lower()
    s = re.sub(r"\\(?:text|mathrm|mathbf|boxed|left|right)\b", "", s)
    s = re.sub(r"\$+", "", s)
    s = s.strip(_STRIP)
    s = re.sub(r"\s+", " ", s)
    return s


def _as_number(s):
    s = _norm(s).replace(",", "").replace("%", "").rstrip(".")
    s = re.sub(r"^[+]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


_MC_RE = re.compile(r"^\(?([a-j])\)?\b", re.I)


def grade(row, text):
    """Return (correct: bool, extracted: str). MC is an exact letter match;
    exactMatch is normalised string equality with a numeric fallback."""
    pred = extract_answer(text)
    gt = row["answer"]
    if row["answer_type"] == "multipleChoice":
        m = _MC_RE.match(pred.strip())
        letter = m.group(1).upper() if m else None
        if letter is None:  # model wrote the option text, not the letter
            return (_norm(gt) != "" and _norm(gt) == _norm(pred)), pred
        return (letter == gt.strip().upper()[:1]), pred
    if _norm(pred) == _norm(gt):
        return True, pred
    a, b = _as_number(pred), _as_number(gt)
    if a is not None and b is not None:
        tol = max(1e-6, abs(b) * 1e-6)
        return (abs(a - b) <= tol), pred
    return False, pred


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def load_pool(args):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test")
    rows = []
    for r in ds:
        if r.get("image"):                       # belt and braces: text only
            continue
        if args.answer_type and r["answer_type"] != args.answer_type:
            continue
        if args.category and args.category.lower() not in r["category"].lower():
            continue
        if len(r["question"]) > args.max_question_chars:
            continue
        if args.max_answer_chars and len(r["answer"]) > args.max_answer_chars:
            continue
        rows.append({k: r[k] for k in
                     ("id", "question", "answer", "answer_type", "category", "raw_subject")})
    rows.sort(key=lambda r: r["id"])             # stable order before sampling
    if args.ids_file:
        want = [l.strip() for l in open(args.ids_file) if l.strip()
                and not l.startswith("#")]
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in want if i not in by_id]
        if missing:
            sys.exit(f"ids not in pool (check filters): {missing}")
        return [by_id[i] for i in want]
    random.Random(args.seed).shuffle(rows)
    return rows[: args.n_questions] if args.n_questions else rows


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------
def ask(base, model, row, max_tokens, temperature, timeout):
    system = SYSTEM_MC if row["answer_type"] == "multipleChoice" else SYSTEM_EXACT
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": row["question"]}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = j["choices"][0]
    msg = ch["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    usage = j.get("usage", {})
    return {
        "reasoning": reasoning,
        "content": content,
        "finish_reason": ch.get("finish_reason"),
        "gen_toks": usage.get("completion_tokens", 0),
        "prompt_toks": usage.get("prompt_tokens", 0),
        "sec": dt,
    }


def wait_for_server(base, timeout=1200):
    """Block until /models answers. Weight load is minutes; a bare connection
    refusal here otherwise reads as a benchmark failure."""
    t0 = time.time()
    url = base.rsplit("/v1", 1)[0] + "/v1/models"
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=10).read()
            return True
        except Exception:
            time.sleep(5)
    return False


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def summarise(recs, questions):
    by_q = {}
    for r in recs:
        by_q.setdefault(r["id"], []).append(r)
    print(f"\n{'id':26s} {'cat':22s} {'pass':>7s} {'n':>3s} {'trunc':>6s} "
          f"{'med tok':>8s} {'med s':>7s}")
    fractional = []
    for q in questions:
        rs = by_q.get(q["id"], [])
        if not rs:
            continue
        ok = [r for r in rs if r.get("correct")]
        tr = [r for r in rs if r.get("finish_reason") == "length"]
        toks = sorted(r["gen_toks"] for r in rs)
        secs = sorted(r["sec"] for r in rs)
        rate = len(ok) / len(rs)
        if 0 < rate < 1:
            fractional.append(q["id"])
        print(f"{q['id']:26s} {q['category'][:22]:22s} {rate*100:6.0f}% {len(rs):3d} "
              f"{len(tr):6d} {toks[len(toks)//2]:8d} {secs[len(secs)//2]:7.0f}")
    n = len(recs)
    if not n:
        return
    acc = sum(1 for r in recs if r.get("correct")) / n
    trunc = sum(1 for r in recs if r.get("finish_reason") == "length") / n
    toks = sorted(r["gen_toks"] for r in recs)
    print("-" * 84)
    print(f"samples={n}  accuracy={acc*100:.1f}%  truncated={trunc*100:.1f}%")
    print(f"gen tokens  p50={toks[len(toks)//2]}  p90={toks[int(len(toks)*0.9)]}  "
          f"max={toks[-1]}")
    tps = [r["gen_toks"] / r["sec"] for r in recs if r["sec"] > 0]
    if tps:
        print(f"throughput  mean={sum(tps)/len(tps):.1f} tok/s")
    if fractional:
        print(f"\nat-the-limit (0 < pass rate < 1): {len(fractional)}")
        for i in fractional:
            print(f"  {i}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "run"])
    ap.add_argument("--model", default="GLM5.2-top8")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--samples", type=int, default=1, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--ids-file", default=None, help="fixed question set, one id per line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer-type", default=None,
                    choices=[None, "exactMatch", "multipleChoice"])
    ap.add_argument("--category", default=None, help="substring of HLE category")
    ap.add_argument("--max-question-chars", type=int, default=2000)
    ap.add_argument("--max-answer-chars", type=int, default=60,
                    help="skip essay-length gold answers the string grader cannot judge")
    ap.add_argument("--out", default="bench/hle/results/hle.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-text", action="store_true",
                    help="store the full reply (large; needed to audit the grader)")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    if a.mode == "probe":
        a.samples = max(a.samples, 1)

    questions = load_pool(a)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    recs = []
    done = set()
    if a.resume and os.path.exists(a.out):
        recs = json.load(open(a.out))["records"]
        done = {(r["id"], r["sample"]) for r in recs}
        print(f"resuming: {len(recs)} samples already on disk")

    print(f"mode={a.mode}  model={a.model}  questions={len(questions)}  "
          f"samples={a.samples}  max_tokens={a.max_tokens}  temp={a.temperature}")
    if not wait_for_server(base):
        sys.exit("server never came up on " + base)

    def flush():
        json.dump({"config": vars(a), "questions": questions, "records": recs},
                  open(a.out, "w"), indent=1)

    t_start = time.time()
    total = len(questions) * a.samples
    k = 0
    for s in range(a.samples):
        for q in questions:
            k += 1
            if (q["id"], s) in done:
                continue
            try:
                r = ask(base, a.model, q, a.max_tokens, a.temperature, a.timeout)
            except Exception as e:
                print(f"[{k:3d}/{total}] {q['id'][:10]} ERROR {e}")
                recs.append({"id": q["id"], "sample": s, "err": str(e),
                             "correct": False, "gen_toks": 0, "sec": 0.0})
                flush()
                continue
            text = (r["reasoning"] + "\n" + r["content"]).strip()
            correct, pred = grade(q, r["content"] or text)
            rec = {"id": q["id"], "sample": s, "category": q["category"],
                   "answer_type": q["answer_type"], "gt": q["answer"], "pred": pred,
                   "correct": bool(correct), "finish_reason": r["finish_reason"],
                   "gen_toks": r["gen_toks"], "prompt_toks": r["prompt_toks"],
                   "sec": round(r["sec"], 1)}
            if a.save_text:
                rec["text"] = text
            recs.append(rec)
            flush()
            tps = r["gen_toks"] / r["sec"] if r["sec"] else 0
            print(f"[{k:3d}/{total}] {q['id'][:10]} s{s} "
                  f"{'OK ' if correct else '.  '} {r['gen_toks']:6d} tok "
                  f"{r['sec']:6.0f}s {tps:5.1f} tok/s "
                  f"{'TRUNC ' if r['finish_reason'] == 'length' else ''}"
                  f"gt={q['answer'][:24]!r} pred={pred[:24]!r}")

    summarise([r for r in recs if "err" not in r], questions)
    print(f"\nwall={(time.time()-t_start)/60:.1f} min   -> {a.out}")


if __name__ == "__main__":
    main()
