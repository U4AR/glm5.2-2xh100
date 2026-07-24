#!/usr/bin/env python3
"""Benchmark the Terminal-Bench prompts through the live OpenAI API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from capture import DEFAULT_BASE, DEFAULT_TASKS, discover_tb_root, read_task_instruction


def served_model(base: str) -> str:
    with urllib.request.urlopen(f"{base.rstrip('/')}/v1/models", timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["id"]


def post_chat(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.perf_counter() - start


def summarize(rows: list[dict]) -> dict:
    elapsed = [row["elapsed_s"] for row in rows]
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    completion_tokens = sum(row["completion_tokens"] for row in rows)
    total_tokens = sum(row["total_tokens"] for row in rows)
    total_elapsed = sum(elapsed)
    return {
        "tasks": len(rows),
        "total_elapsed_s": round(total_elapsed, 3),
        "mean_elapsed_s": round(statistics.mean(elapsed), 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tok_s": round(completion_tokens / total_elapsed, 3),
        "total_tok_s": round(total_tokens / total_elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "live_bench")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tb_root = discover_tb_root()
    model = served_model(args.base)
    rows = []

    for task in args.tasks:
        prompt = read_task_instruction(tb_root, task)
        data, elapsed = post_chat(args.base, model, prompt, args.max_tokens, args.timeout)
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        text = content if content is not None else reasoning
        row = {
            "label": args.label,
            "task": task,
            "elapsed_s": round(elapsed, 3),
            "prompt_chars": len(prompt),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "finish_reason": choice.get("finish_reason"),
            "content_is_none": content is None,
            "output_preview": (text or "")[:240],
        }
        row["completion_tok_s"] = round(row["completion_tokens"] / elapsed, 3) if elapsed else 0
        row["total_tok_s"] = round(row["total_tokens"] / elapsed, 3) if elapsed else 0
        rows.append(row)
        print(
            f"{args.label} {task}: {row['elapsed_s']}s "
            f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
            f"completion_tok_s={row['completion_tok_s']}"
        )

    result = {
        "label": args.label,
        "model": model,
        "max_tokens": args.max_tokens,
        "summary": summarize(rows),
        "rows": rows,
    }
    out_path = args.out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
