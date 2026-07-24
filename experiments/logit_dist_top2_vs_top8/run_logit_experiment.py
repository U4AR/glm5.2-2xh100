#!/usr/bin/env python3
"""Compare first-output-token logprob distributions for GLM5.2 top2 vs top8.

The live server exposes expert tier selection via the OpenAI model name:
GLM5.2-top2 and GLM5.2-top8.  This harness sends paired prompts to both tiers,
records the sampled first visible token and logprob, then writes a statistical
summary over paired logprob differences.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_OUT = Path(__file__).resolve().parent / "runs"
SEED = 20260702


def build_prompts(n: int) -> list[str]:
    templates = [
        "Answer with only the number: {a} + {b}.",
        "Answer with only the number: {a} * {b}.",
        "Answer with one word: the opposite of {word}.",
        "Answer with one word: the color of {thing}.",
        "Complete the phrase with one word only: {phrase}",
        "Translate to English with one word only: {foreign}.",
        "Answer yes or no only: {claim}",
        "Return only the next item in the sequence: {seq}",
    ]
    words = [
        ("hot", "ice", "white", "bonjour", "the sky is", "2, 4, 6, 8,", "water is wet"),
        ("early", "grass", "green", "gracias", "peanut butter and", "3, 6, 9, 12,", "fire is cold"),
        ("up", "banana", "yellow", "gato", "salt and", "5, 10, 15, 20,", "Paris is in France"),
        ("empty", "snow", "white", "rojo", "lock and", "1, 1, 2, 3, 5,", "the moon is a star"),
        ("near", "coal", "black", "chien", "bread and", "10, 20, 30, 40,", "two plus two is four"),
    ]

    rng = random.Random(SEED)
    prompts: list[str] = []
    for i in range(n):
        t = templates[i % len(templates)]
        row = words[i % len(words)]
        a = rng.randint(2, 49)
        b = rng.randint(2, 49)
        prompts.append(t.format(
            a=a,
            b=b,
            word=row[0],
            thing=row[1],
            phrase=row[4],
            foreign=row[3],
            claim=row[6],
            seq=row[5],
        ))
    return prompts


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get_served_model(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
        data = json.load(resp)["data"]
    root = data[0]["root"] or data[0]["id"]
    return root.split("-top", 1)[0]


def extract_record(prompt_id: int, prompt: str, tier: int, elapsed: float, response: dict) -> dict:
    choice = response["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content")
    rows = ((choice.get("logprobs") or {}).get("content") or [])
    first = rows[0] if rows else {}
    top = first.get("top_logprobs") or []
    logprob = first.get("logprob")
    prob = math.exp(logprob) if isinstance(logprob, (int, float)) else None
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "tier": tier,
        "model": response.get("model"),
        "token": first.get("token"),
        "content": content,
        "logprob": logprob,
        "prob": prob,
        "top_logprobs_count": len(top),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_s": elapsed,
        "raw_id": response.get("id"),
    }


def run_one(base: str, model_root: str, prompt_id: int, prompt: str, tier: int) -> dict:
    payload = {
        "model": f"{model_root}-top{tier}",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.time()
    response = post_json(base + "/v1/chat/completions", payload)
    return extract_record(prompt_id, prompt, tier, time.time() - start, response)


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_stats(records: list[dict]) -> dict:
    by_prompt: dict[int, dict[int, dict]] = {}
    for rec in records:
        by_prompt.setdefault(rec["prompt_id"], {})[rec["tier"]] = rec

    pairs = [tiers for _, tiers in sorted(by_prompt.items()) if 2 in tiers and 8 in tiers]
    diffs = [p[2]["logprob"] - p[8]["logprob"] for p in pairs
             if isinstance(p[2].get("logprob"), (int, float))
             and isinstance(p[8].get("logprob"), (int, float))]
    abs_diffs = [abs(x) for x in diffs]
    signs = [x for x in diffs if abs(x) > 1e-12]
    same_token = sum(1 for p in pairs if p[2].get("token") == p[8].get("token"))
    top_counts = [rec.get("top_logprobs_count", 0) for rec in records]

    rng = random.Random(SEED)
    boots = []
    if diffs:
        for _ in range(5000):
            boots.append(mean([rng.choice(diffs) for _ in diffs]))
    sd = stdev(diffs)
    se = sd / math.sqrt(len(diffs)) if diffs and math.isfinite(sd) else float("nan")
    t_stat = mean(diffs) / se if se and math.isfinite(se) and se > 0 else float("nan")
    p_norm = 2 * (1 - normal_cdf(abs(t_stat))) if math.isfinite(t_stat) else float("nan")

    positives = sum(1 for x in signs if x > 0)
    n_sign = len(signs)
    # Two-sided exact sign test against p=0.5.
    if n_sign:
        k = min(positives, n_sign - positives)
        tail = sum(math.comb(n_sign, i) for i in range(k + 1)) / (2 ** n_sign)
        sign_p = min(1.0, 2 * tail)
    else:
        sign_p = float("nan")

    return {
        "n_pairs": len(diffs),
        "top2_minus_top8_logprob_mean": mean(diffs),
        "top2_minus_top8_logprob_median": quantile(diffs, 0.5),
        "top2_minus_top8_logprob_sd": sd,
        "mean_ci95_bootstrap": [quantile(boots, 0.025), quantile(boots, 0.975)] if boots else [float("nan"), float("nan")],
        "paired_t_normal_approx_p": p_norm,
        "sign_test_nonzero_n": n_sign,
        "sign_test_top2_higher_n": positives,
        "sign_test_two_sided_p": sign_p,
        "abs_logprob_diff_median": quantile(abs_diffs, 0.5),
        "abs_logprob_diff_p90": quantile(abs_diffs, 0.9),
        "same_first_token_rate": same_token / len(pairs) if pairs else float("nan"),
        "top_logprobs_count_min": min(top_counts) if top_counts else None,
        "top_logprobs_count_median": quantile(top_counts, 0.5) if top_counts else None,
        "top_logprobs_count_max": max(top_counts) if top_counts else None,
    }


def write_summary(path: Path, run_meta: dict, stats: dict, records: list[dict]) -> None:
    top2 = [r for r in records if r["tier"] == 2]
    top8 = [r for r in records if r["tier"] == 8]
    examples = []
    for pid in sorted({r["prompt_id"] for r in records})[:8]:
        a = next(r for r in records if r["prompt_id"] == pid and r["tier"] == 2)
        b = next(r for r in records if r["prompt_id"] == pid and r["tier"] == 8)
        examples.append((pid, a["prompt"], a["token"], a["logprob"], b["token"], b["logprob"]))

    lines = [
        "# Top2 vs Top8 First-Token Logprob Experiment",
        "",
        f"- Date: {run_meta['date_utc']}",
        f"- Base URL: `{run_meta['base']}`",
        f"- Model root: `{run_meta['model_root']}`",
        f"- Prompts: {run_meta['n_prompts']}",
        f"- Paired valid observations: {stats['n_pairs']}",
        f"- Request shape: chat completions, `temperature=0`, `max_tokens=1`, `logprobs=true`, `top_logprobs=20`, thinking disabled.",
        "",
        "## Main Result",
        "",
        f"- Mean paired logprob difference, top2 - top8: {stats['top2_minus_top8_logprob_mean']:.6f}",
        f"- 95% bootstrap CI for the mean: [{stats['mean_ci95_bootstrap'][0]:.6f}, {stats['mean_ci95_bootstrap'][1]:.6f}]",
        f"- Median paired difference: {stats['top2_minus_top8_logprob_median']:.6f}",
        f"- Paired t normal-approx p-value: {stats['paired_t_normal_approx_p']:.4g}",
        f"- Sign test: top2 higher on {stats['sign_test_top2_higher_n']}/{stats['sign_test_nonzero_n']} nonzero pairs, p={stats['sign_test_two_sided_p']:.4g}",
        f"- Same first visible token rate: {100 * stats['same_first_token_rate']:.1f}%",
        f"- Median absolute logprob shift: {stats['abs_logprob_diff_median']:.6f}",
        f"- 90th percentile absolute logprob shift: {stats['abs_logprob_diff_p90']:.6f}",
        "",
        "## Tier Marginals",
        "",
        f"- Top2 mean logprob: {mean([r['logprob'] for r in top2]):.6f}",
        f"- Top8 mean logprob: {mean([r['logprob'] for r in top8]):.6f}",
        f"- Top2 mean probability of sampled token: {mean([r['prob'] for r in top2]):.6f}",
        f"- Top8 mean probability of sampled token: {mean([r['prob'] for r in top8]):.6f}",
        "",
        "## Logprobs Payload Note",
        "",
        f"`top_logprobs` candidate counts were min/median/max = {stats['top_logprobs_count_min']}/"
        f"{stats['top_logprobs_count_median']}/{stats['top_logprobs_count_max']}. "
        "On this running server the OpenAI-compatible response returns the sampled token logprob reliably, "
        "but the requested top-logprobs list is collapsed to one packed candidate. The analysis therefore "
        "uses sampled-token logprob distributions, not full vocabulary entropy.",
        "",
        "## Example Pairs",
        "",
        "| id | prompt | top2 token | top2 logprob | top8 token | top8 logprob |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for pid, prompt, t2, lp2, t8, lp8 in examples:
        prompt = prompt.replace("|", "\\|")
        lines.append(f"| {pid} | {prompt} | `{t2}` | {lp2:.6f} | `{t8}` | {lp8:.6f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--prompts", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / stamp
    run_dir.mkdir()

    model_root = get_served_model(args.base)
    prompts = build_prompts(args.prompts)
    jobs = [(pid, prompt, tier) for pid, prompt in enumerate(prompts) for tier in (2, 8)]
    records: list[dict] = []

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, args.base, model_root, pid, prompt, tier)
                for pid, prompt, tier in jobs]
        for i, fut in enumerate(futures.as_completed(futs), 1):
            try:
                records.append(fut.result())
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                records.append({"error": repr(exc)})
            if i % 24 == 0:
                print(f"completed {i}/{len(jobs)} requests", flush=True)

    records = [r for r in records if "error" not in r]
    records.sort(key=lambda r: (r["prompt_id"], r["tier"]))
    raw_path = run_dir / "records.jsonl"
    raw_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    stats = paired_stats(records)
    meta = {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(),
        "base": args.base,
        "model_root": model_root,
        "n_prompts": args.prompts,
        "workers": args.workers,
        "seed": SEED,
        "records": len(records),
    }
    (run_dir / "stats.json").write_text(json.dumps({"meta": meta, "stats": stats}, indent=2), encoding="utf-8")
    write_summary(run_dir / "SUMMARY.md", meta, stats, records)

    latest = args.out / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    print(f"wrote {run_dir}")
    print(f"mean top2-top8 logprob diff: {stats['top2_minus_top8_logprob_mean']:.6f}")
    print(f"same first token rate: {100 * stats['same_first_token_rate']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
