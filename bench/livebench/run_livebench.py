#!/usr/bin/env python3
"""Run the full LiveBench reasoning benchmark against the running GLM-5.2 server.

One question at a time (sequential), scored per LiveBench task rules, against
whatever server is listening on --base. The model name carries the expert tier:
the default `GLM5.2-top2` is the fast top-2 substitution recipe (see run_fast.sh).

Portable: pulls the dataset straight from the HuggingFace hub (livebench/reasoning),
so it works on any fresh box with network access — no local files to copy. Talks to
the server over plain OpenAI /v1/chat/completions with only the stdlib (urllib).

    python bench/livebench/run_livebench.py                 # top-2, all 200 reasoning Qs
    python bench/livebench/run_livebench.py --limit 20       # quick smoke (first 20)
    python bench/livebench/run_livebench.py --model GLM5.2-top8   # baseline tier
    python bench/livebench/run_livebench.py --task zebra_puzzle   # one task only

The dataset (livebench/reasoning) is 200 questions: 100 zebra_puzzle, 50 spatial,
50 web_of_lies_v2. Each task has its own answer format and grader below.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# --- number words for the spatial task's integer answers --------------------
_WORD2NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _norm(s):
    """Lowercase, strip surrounding markup/space, collapse internal whitespace."""
    s = s.strip().lower().strip(".*`_ \n\t")
    return re.sub(r"\s+", " ", s)


def _last_bold(text):
    """Return the last **bold** span (LiveBench spatial / web_of_lies answer)."""
    m = re.findall(r"\*\*(.+?)\*\*", text, re.S)
    return m[-1].strip() if m else None


def _last_solution(text):
    """Return the last <solution>..</solution> or ***..*** span (zebra answer)."""
    m = re.findall(r"<solution>(.*?)</solution>", text, re.S)
    if m:
        return m[-1].strip()
    m = re.findall(r"\*\*\*(.+?)\*\*\*", text, re.S)
    return m[-1].strip() if m else None


def grade(task, ground_truth, text):
    """LiveBench-style score in [0,1]. Zebra gets partial credit (fraction of
    slots correct); spatial and web_of_lies are all-or-nothing exact match."""
    if task == "zebra_puzzle":
        pred = _last_solution(text)
        if pred is None:
            return 0.0, None
        gts = [_norm(x) for x in ground_truth.split(",")]
        pas = [_norm(x) for x in pred.split(",")]
        hits = sum(1 for i, g in enumerate(gts) if i < len(pas) and pas[i] == g)
        return hits / len(gts), pred

    if task == "web_of_lies_v2":
        pred = _last_bold(text)
        if pred is None:
            return 0.0, None
        gts = [_norm(x) for x in ground_truth.split(",")]
        pas = [_norm(x) for x in pred.split(",")]
        return (1.0 if pas == gts else 0.0), pred

    if task == "spatial":
        pred = _last_bold(text) or ""
        g = _norm(ground_truth)
        p = _norm(pred)
        # map number words -> digits both ways so "three" matches "3"
        gd = _WORD2NUM.get(g, g)
        pd = _WORD2NUM.get(p, p)
        return (1.0 if str(gd) == str(pd) else 0.0), pred

    # unknown task: exact normalized match
    p = _norm(text)
    return (1.0 if _norm(ground_truth) in p else 0.0), None


def ask(base, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=timeout)
    j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0]["message"]
    txt = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    ctoks = j.get("usage", {}).get("completion_tokens", 0)
    return txt, ctoks, dt


def load_questions(category, task, limit, per_task_limit=0):
    """Pull the LiveBench split from HF (downloads on first run) -> list of dicts.

    --limit takes the first N of the file, and the file is ORDERED BY TASK, so
    `--limit 3` is three zebra puzzles and nothing else. For comparing two
    server configurations that is the wrong sample: it measures one task and
    calls it the benchmark. --per-task-limit takes the first N of EACH task, so
    every configuration sees the same stratified subset in the same order.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Need the 'datasets' package:  pip install datasets")
    ds = load_dataset(f"livebench/{category}", split="test")
    qs = []
    seen = {}
    for r in ds:
        if task and r["task"] != task:
            continue
        if per_task_limit:
            n = seen.get(r["task"], 0)
            if n >= per_task_limit:
                continue
            seen[r["task"]] = n + 1
        turns = r["turns"]
        if isinstance(turns, str):
            turns = json.loads(turns)
        qs.append({"id": r["question_id"][:12], "task": r["task"],
                   "prompt": turns[0], "gt": r["ground_truth"]})
    if limit:
        qs = qs[:limit]
    return qs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="GLM5.2-top2",
                    help="served model + tier suffix (default: GLM5.2-top2)")
    ap.add_argument("--base", default="http://localhost:8000/v1",
                    help="OpenAI base URL (default: http://localhost:8000/v1)")
    ap.add_argument("--category", default="reasoning", help="LiveBench category")
    ap.add_argument("--task", default=None,
                    help="restrict to one task (zebra_puzzle|spatial|web_of_lies_v2)")
    ap.add_argument("--limit", type=int, default=0, help="only first N questions")
    ap.add_argument("--per-task-limit", type=int, default=0,
                    help="first N of EACH task (stratified; use this to compare configs)")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="livebench_results.json")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    qs = load_questions(a.category, a.task, a.limit, a.per_task_limit)
    print(f"model={a.model}  base={base}  category={a.category}"
          f"{'  task=' + a.task if a.task else ''}  n={len(qs)}\n")

    results = []
    t_start = time.time()
    for i, q in enumerate(qs, 1):
        try:
            txt, ctoks, dt = ask(base, a.model, q["prompt"], a.max_tokens, a.timeout)
            score, pred = grade(q["task"], q["gt"], txt)
            toks = ctoks / dt if dt else 0
            results.append({"id": q["id"], "task": q["task"], "gt": q["gt"],
                            "pred": pred, "score": score, "sec": round(dt, 1),
                            "gen_toks": ctoks, "tok_s": round(toks, 1)})
            print(f"[{i:3d}/{len(qs)}] {q['task']:15s} score={score:4.2f} "
                  f"{toks:5.1f} tok/s {ctoks:5d} tok  gt={q['gt'][:32]!r} pred={str(pred)[:32]!r}")
        except Exception as e:
            print(f"[{i:3d}/{len(qs)}] {q['task']:15s} ERROR {e}")
            results.append({"id": q["id"], "task": q["task"], "score": 0.0, "err": str(e)})

    wall = time.time() - t_start
    json.dump(results, open(a.out, "w"), indent=1)

    # --- summary: overall + per task ---
    def agg(rs):
        rs = [r for r in rs if "err" not in r]
        if not rs:
            return 0.0, 0.0, 0
        acc = sum(r["score"] for r in rs) / len(rs)
        ts = sum(r["tok_s"] for r in rs) / len(rs)
        return acc, ts, len(rs)

    tasks = sorted({r["task"] for r in results})
    print(f"\n{'task':16s} {'score':>7s} {'avg tok/s':>10s} {'n':>4s}")
    for t in tasks:
        acc, ts, n = agg([r for r in results if r["task"] == t])
        print(f"{t:16s} {acc*100:6.1f}% {ts:10.1f} {n:4d}")
    acc, ts, n = agg(results)
    print(f"{'-'*40}")
    print(f"{'OVERALL':16s} {acc*100:6.1f}% {ts:10.1f} {n:4d}")
    print(f"\nwall={wall/60:.1f} min   results -> {a.out}")


if __name__ == "__main__":
    main()
