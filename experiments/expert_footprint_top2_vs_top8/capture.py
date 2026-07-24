#!/usr/bin/env python3
"""Capture top-8 router traces for the expert footprint experiment.

Run this against a server started with KT_DUMP_TOPK=1. Requests are intentionally
sequential because the trace hook writes one process-global buffer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_TAG_FILE = Path("/tmp/kt_topk_tag")
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_TASKS = [
    "llm-inference-batching-scheduler",
    "largest-eigenval",
    "fix-git",
    "compile-compcert",
    "git-multibranch",
]
EXP1_PROMPT = (
    "Write a short coherent paragraph about why the sky is blue. "
    "Then list the first 8 prime numbers."
)


def url_json(url: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def served_model(base: str) -> str:
    data = url_json(f"{base.rstrip('/')}/v1/models")
    return data["data"][0]["id"]


def post_chat(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_tb_root() -> Path:
    candidates = []
    env = os.environ.get("TB_DIR")
    if env:
        candidates.extend([Path(env) / "terminal-bench-2", Path(env)])
    candidates.extend(
        [
            REPO / ".terminalbench" / "terminal-bench-2",
            Path("/data/projects/isolated_bench/terminal-bench-2"),
            Path("/data/tmp/tb2"),
        ]
    )
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path

    target = Path("/data/tmp/tb2")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/laude-institute/terminal-bench-2.git",
            str(target),
        ],
        check=True,
    )
    return target


def task_dir(tb_root: Path, name: str) -> Path:
    for candidate in (tb_root / name, tb_root / "tasks" / name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find Terminal-Bench task {name!r} under {tb_root}")


def read_task_instruction(tb_root: Path, name: str) -> str:
    tdir = task_dir(tb_root, name)
    for filename in ("instruction.md", "instruction.txt"):
        path = tdir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text

    for filename in ("task.yaml", "task.yml", "task.toml"):
        path = tdir / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        prompt = extract_instruction_field(text)
        if prompt:
            return prompt
    raise FileNotFoundError(f"no instruction file found for {name} in {tdir}")


def extract_instruction_field(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("instruction"):
            _, sep, rest = stripped.partition(":")
            if not sep:
                _, sep, rest = stripped.partition("=")
            if not sep:
                continue
            rest = rest.strip()
            if rest in {"|", ">"}:
                block = []
                for follow in lines[i + 1 :]:
                    if follow.startswith((" ", "\t")):
                        block.append(follow.strip())
                    elif not follow.strip():
                        block.append("")
                    else:
                        break
                return "\n".join(block).strip()
            return rest.strip().strip("'\"")
    return ""


def wait_for_trace(path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.25)
    raise TimeoutError(f"no trace written at {path}")


def capture_one(
    *,
    base: str,
    model: str,
    runs: Path,
    tag_file: Path,
    tag: str,
    prompt: str,
    max_tokens: int,
    request_timeout: float,
    trace_timeout: float,
) -> dict:
    runs.mkdir(parents=True, exist_ok=True)
    trace = runs / f"{tag}.pt"
    if trace.exists():
        trace.unlink()
    tag_file.write_text(tag, encoding="utf-8")
    started = time.time()
    response = post_chat(base, model, prompt, max_tokens, request_timeout)
    wait_for_trace(trace, trace_timeout)
    tag_file.write_text(f"{tag}_FLUSH", encoding="utf-8")
    return {
        "tag": tag,
        "trace": str(trace),
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "elapsed_s": round(time.time() - started, 3),
        "response_id": response.get("id"),
    }


def build_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    items = [("exp1_single", EXP1_PROMPT)]
    if args.exp == "exp1":
        return items

    tb_root = discover_tb_root()
    tasks = args.tasks or DEFAULT_TASKS
    prompts = [(f"exp3_{name}", read_task_instruction(tb_root, name)) for name in tasks]
    if args.exp == "exp2":
        return [(f"exp2_{tasks[0]}", prompts[0][1])]
    if args.exp == "exp3":
        return prompts
    return items + prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--tag-file", type=Path, default=DEFAULT_TAG_FILE)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--with-decode", action="store_true", help="use --decode-tokens")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--trace-timeout", type=float, default=30.0)
    parser.add_argument("--exp", choices=("exp1", "exp2", "exp3", "all"), default="all")
    parser.add_argument("--tasks", nargs="*", help="Terminal-Bench task names for exp2/exp3/all")
    args = parser.parse_args()

    model = served_model(args.base)
    max_tokens = args.decode_tokens if args.with_decode else args.max_tokens
    plan = build_plan(args)

    manifest = {
        "base": args.base,
        "model": model,
        "max_tokens": max_tokens,
        "items": [],
    }
    for tag, prompt in plan:
        print(f"capturing {tag} ({len(prompt)} chars)")
        try:
            rec = capture_one(
                base=args.base,
                model=model,
                runs=args.runs,
                tag_file=args.tag_file,
                tag=tag,
                prompt=prompt,
                max_tokens=max_tokens,
                request_timeout=args.request_timeout,
                trace_timeout=args.trace_timeout,
            )
        except urllib.error.URLError as exc:
            raise SystemExit(f"request failed for {tag}: {exc}") from exc
        manifest["items"].append(rec)
        print(f"  wrote {rec['trace']} in {rec['elapsed_s']}s")

    manifest_path = args.runs / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
