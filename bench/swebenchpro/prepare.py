#!/usr/bin/env python3
"""Pick N SWE-bench Pro instances, emit what the official evaluator needs, pull images.

SWE-bench Pro is the one benchmark on Zhipu's GLM-5.2 table that is both hard
(GLM-5.2 scores 62.1) and pass/fail per item -- the tests decide, so there is no
judge and no string matching to argue with.

Grading is done by the UPSTREAM script (bench/swebenchpro_upstream/swe_bench_pro_eval.py
with --use_local_docker), not by anything written here. An instance counts as
resolved only when (FAIL_TO_PASS | PASS_TO_PASS) are all PASSED, which is the
official criterion. We only prepare its inputs:

  instances.csv     the raw-sample table (instance_id, before_repo_set_cmd,
                    selected_test_files_to_run, base_commit, fail_to_pass,
                    pass_to_pass, repo)
  gold_patches.json the reference patches -- run these FIRST. If gold does not
                    score 5/5 the harness is broken, not the model, and any
                    tier comparison built on it is meaningless.
  instances.json    full records (problem statement, requirements, interface,
                    gold patch) for the patch generator.

    python bench/swebenchpro/prepare.py --n 5 --repo qutebrowser/qutebrowser --pull

The list fields in the HF dataset are PYTHON reprs, not JSON (mixed quote styles),
which is why they are read with ast.literal_eval -- json.loads throws on them.
"""
import argparse
import ast
import json
import os
import random
import subprocess
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(os.path.dirname(OUT_DIR), "swebenchpro_upstream")
sys.path.insert(0, UPSTREAM)


def as_list(v):
    return v if isinstance(v, list) else ast.literal_eval(v)


def image_uri(uid, repo, username="jefzda"):
    from helper_code.image_uri import get_dockerhub_image_uri
    return get_dockerhub_image_uri(uid, username, repo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--repo", default="qutebrowser/qutebrowser",
                    help="one repo keeps the images similar in size and the test "
                         "runner uniform; '' = any repo")
    ap.add_argument("--max-test-files", type=int, default=2,
                    help="cost control: each selected test file is a full suite run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--pull", action="store_true", help="docker pull the images now")
    ap.add_argument("--username", default="jefzda")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    rows = []
    for r in ds:
        if a.repo and r["repo"] != a.repo:
            continue
        if len(as_list(r["selected_test_files_to_run"])) > a.max_test_files:
            continue
        # the upstream run_scripts/dockerfiles must exist for the evaluator
        if not os.path.isdir(os.path.join(UPSTREAM, "run_scripts", r["instance_id"])):
            continue
        if not os.path.isdir(os.path.join(UPSTREAM, "dockerfiles", "base_dockerfile",
                                          r["instance_id"])):
            continue
        rows.append(r)

    if len(rows) < a.n:
        sys.exit(f"only {len(rows)} instances match the filters; loosen --max-test-files")

    # Random within the cost filter, NOT sorted by patch size: ranking by "smallest
    # gold patch" would quietly select the easy tail and inflate every score.
    random.Random(a.seed).shuffle(rows)
    picked = rows[: a.n]

    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    cols = ["instance_id", "repo", "base_commit", "before_repo_set_cmd",
            "selected_test_files_to_run", "fail_to_pass", "pass_to_pass"]
    pd.DataFrame([{c: r[c] for c in cols} for r in picked]).to_csv(
        os.path.join(a.out, "instances.csv"), index=False)

    json.dump([{"instance_id": r["instance_id"], "patch": r["patch"], "prefix": "gold"}
               for r in picked],
              open(os.path.join(a.out, "gold_patches.json"), "w"), indent=1)

    keep = cols + ["patch", "test_patch", "problem_statement", "requirements",
                   "interface", "repo_language", "dockerhub_tag"]
    json.dump([{k: r[k] for k in keep} for r in picked],
              open(os.path.join(a.out, "instances.json"), "w"), indent=1)

    print(f"{len(picked)} instances -> {a.out}/instances.csv, gold_patches.json, instances.json\n")
    for r in picked:
        f2p, p2p = as_list(r["fail_to_pass"]), as_list(r["pass_to_pass"])
        print(f"  {r['instance_id'][:70]}")
        print(f"    tests={as_list(r['selected_test_files_to_run'])} F2P={len(f2p)} "
              f"P2P={len(p2p)} patch={len(r['patch'])}B")
        print(f"    image={image_uri(r['instance_id'], r['repo'], a.username)}")

    if a.pull:
        for r in picked:
            uri = image_uri(r["instance_id"], r["repo"], a.username)
            print(f"\npulling {uri}")
            subprocess.run(["docker", "pull", uri], check=False)
        subprocess.run(["df", "-h", "/"], check=False)


if __name__ == "__main__":
    main()
