#!/usr/bin/env python3
"""Generate SWE-bench Pro patches with the local GLM-5.2 server, one call per instance.

Setting: ORACLE FILE LOCALIZATION, not the agentic setting Zhipu's 62.1 comes
from. The model is given the issue, the stated requirements/interface, and the
full current text of exactly the files the gold patch touches -- it does not
have to search the repo, run tests, or iterate. That is a different (easier on
localization, harder on one-shot correctness) task than SWE-agent's 62.1, so the
number here is NOT comparable to the published one. It is comparable across
tiers, which is the point: both tiers get the identical prompt.

The reply format is SEARCH/REPLACE blocks rather than a unified diff, because
models reliably miscount diff hunk line numbers, and a malformed diff would
score as a reasoning failure when it is a formatting failure. Blocks are applied
here with exact string matching and the patch is produced by `git diff`, so an
unapplied block is reported as `apply_failed` and kept separate from a test
failure.

    # repo checkout used to read base-commit file contents
    python bench/swebenchpro/gen_patches.py --clone

    python bench/swebenchpro/gen_patches.py --model GLM5.2-top2 \
        --out bench/swebenchpro/patches_top2.json

Then grade with the upstream evaluator (see run_eval.sh).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CACHE = os.environ.get("SWEBP_REPO_CACHE", "/data/swebp_repos")

SYSTEM = """You are an expert software engineer fixing a real issue in a large codebase.

You will be given an issue, the requirements it must satisfy, and the full current
contents of the files you may modify. Reply with edits ONLY, in this exact format:

<<<<<<< SEARCH path/to/file.py
(the exact existing lines to replace, copied character for character)
=======
(the replacement lines)
>>>>>>> REPLACE

Rules:
- The SEARCH text must appear EXACTLY ONCE in the named file, copied verbatim
  including indentation. Include enough surrounding lines to be unique.
- Emit as many blocks as you need, for any of the files shown.
- Do not explain, do not output a diff, do not output the whole file.
- Your edit must make the described behaviour correct in general, not special-case
  the tests."""

BLOCK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s+(?P<path>\S+)\s*\n(?P<search>.*?)\n?=======\s*\n(?P<replace>.*?)\n?>>>>>>>\s*REPLACE",
    re.S)


def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def ensure_clone(repo):
    """A bare-ish checkout per repo so file contents at base_commit can be read
    without booting a container for every instance."""
    path = os.path.join(REPO_CACHE, repo.replace("/", "__"))
    if not os.path.isdir(path):
        os.makedirs(REPO_CACHE, exist_ok=True)
        print(f"cloning {repo} -> {path}")
        sh(["git", "clone", "-q", f"https://github.com/{repo}.git", path])
    return path


# Documentation the gold patch happens to touch. Showing it is pure cost: no
# FAIL_TO_PASS test reads a changelog, and one instance's changelog + settings
# docs alone are 334 KB, which blows the 82k context on their own.
DOC_RE = re.compile(r"(^|/)(doc/|docs/|CHANGELOG|changelog)", re.I)


def files_in_patch(patch, drop_docs=True):
    fs = [l.split(" b/", 1)[1].strip() for l in patch.split("\n")
          if l.startswith("diff --git")]
    return [f for f in fs if not (drop_docs and DOC_RE.search(f))]


def build_prompt(inst, path):
    """Issue + requirements + interface + oracle files at base_commit."""
    parts = [f"# Issue\n\n{inst['problem_statement'].strip()}"]
    if inst.get("requirements"):
        parts.append(f"# Requirements\n\n{inst['requirements'].strip()}")
    if inst.get("interface"):
        parts.append(f"# New interfaces to introduce\n\n{inst['interface'].strip()}")
    parts.append("# Files you may modify")
    for f in files_in_patch(inst["patch"]):
        try:
            body = sh(["git", "show", f"{inst['base_commit']}:{f}"], cwd=path)
        except RuntimeError:
            continue                       # file created by the patch
        parts.append(f"## {f}\n```\n{body}\n```")
    parts.append("Now output the SEARCH/REPLACE blocks that fix the issue.")
    return "\n\n".join(parts)


def apply_blocks(text, path, base_commit):
    """Apply blocks to a clean checkout at base_commit; return (patch, report)."""
    sh(["git", "checkout", "-q", "--detach", base_commit], cwd=path)
    sh(["git", "reset", "-q", "--hard"], cwd=path)
    sh(["git", "clean", "-qfd"], cwd=path)

    blocks = list(BLOCK_RE.finditer(text))
    ok = bad = 0
    problems = []
    for m in blocks:
        f = os.path.join(path, m.group("path"))
        if not os.path.isfile(f):
            bad += 1
            problems.append(f"no such file: {m.group('path')}")
            continue
        src = open(f, encoding="utf-8", errors="surrogateescape").read()
        s, r = m.group("search"), m.group("replace")
        n = src.count(s)
        if n != 1:
            bad += 1
            problems.append(f"{m.group('path')}: SEARCH matched {n} times")
            continue
        open(f, "w", encoding="utf-8", errors="surrogateescape").write(src.replace(s, r, 1))
        ok += 1
    patch = sh(["git", "diff"], cwd=path)
    return patch, {"blocks": len(blocks), "applied": ok, "failed": bad,
                   "problems": problems}


def ask(base, model, prompt, max_tokens, temperature, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    j = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = j["choices"][0]
    return {"content": ch["message"].get("content") or "",
            "reasoning": ch["message"].get("reasoning_content") or "",
            "finish_reason": ch.get("finish_reason"),
            "gen_toks": j.get("usage", {}).get("completion_tokens", 0),
            "prompt_toks": j.get("usage", {}).get("prompt_tokens", 0),
            "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=os.path.join(HERE, "instances.json"))
    ap.add_argument("--model", default="GLM5.2-top2")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--out", default=os.path.join(HERE, "patches.json"))
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--clone", action="store_true", help="only clone repos and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and print their token-ish size; no server")
    a = ap.parse_args()

    insts = json.load(open(a.instances))
    paths = {r["repo"]: ensure_clone(r["repo"]) for r in insts}
    if a.clone:
        return

    if a.dry_run:
        for r in insts:
            p = build_prompt(r, paths[r["repo"]])
            print(f"{r['instance_id'][:52]}  prompt~{len(p)//4} tok  "
                  f"files={files_in_patch(r['patch'])}")
        return

    out, log = [], []
    for s in range(a.samples):
        for i, r in enumerate(insts, 1):
            path = paths[r["repo"]]
            prompt = build_prompt(r, path)
            tag = f"{r['instance_id'][:44]} s{s}"
            try:
                resp = ask(a.base.rstrip("/"), a.model, prompt, a.max_tokens,
                           a.temperature, a.timeout)
            except Exception as e:
                print(f"[{i}/{len(insts)}] {tag} ERROR {e}")
                log.append({"instance_id": r["instance_id"], "sample": s, "err": str(e)})
                continue
            patch, rep = apply_blocks(resp["content"], path, r["base_commit"])
            prefix = f"{a.model}_s{s}"
            out.append({"instance_id": r["instance_id"], "patch": patch,
                        "prefix": prefix})
            log.append({"instance_id": r["instance_id"], "sample": s, "prefix": prefix,
                        "gen_toks": resp["gen_toks"], "prompt_toks": resp["prompt_toks"],
                        "sec": round(resp["sec"], 1),
                        "finish_reason": resp["finish_reason"],
                        "patch_bytes": len(patch), **rep})
            print(f"[{i}/{len(insts)}] {tag} {resp['gen_toks']:6d} tok "
                  f"{resp['sec']:6.0f}s  blocks={rep['blocks']} applied={rep['applied']} "
                  f"failed={rep['failed']} patch={len(patch)}B"
                  f"{'  TRUNC' if resp['finish_reason'] == 'length' else ''}")
            for p in rep["problems"]:
                print(f"      ! {p}")
            json.dump(out, open(a.out, "w"), indent=1)
            json.dump(log, open(a.out.replace(".json", "_log.json"), "w"), indent=1)

    empty = sum(1 for o in out if not o["patch"].strip())
    print(f"\n{len(out)} patches -> {a.out}   ({empty} empty)")


if __name__ == "__main__":
    main()
