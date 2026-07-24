#!/usr/bin/env python3
"""Adaptive-cache phase test:
  A1: prompt A (cold)     -> tok/s + resident snapshot
  A2: prompt A again      -> tok/s + snapshot; similarity A1->A2
  B : similar-domain      -> tok/s + snapshot; similarity A2->B
  C : different-domain    -> tok/s + snapshot; similarity A2->C, B->C
Similarity = mean over layers of |X∩Y|/num_gpu_experts. Random-independent
baseline at 96/256 residents = 0.375. tok/s from usage.completion_tokens.
Each phase runs RUNS generations so the cache sees enough steps to sweep."""
import json, shutil, sys, time, urllib.request

import torch

BASE = "http://127.0.0.1:8000"
SNAP_DIR = "/data/tmp/claude-1002/-data-models-RunGLM/e275e7cc-6077-447f-b05c-f8d8da259f76/scratchpad"
MASKS = "/tmp/kt_adaptive_masks.pt"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 900

PROMPT_A = ("Write a thorough, coherent essay (do NOT stop early, keep going) explaining "
            "how modern LLM inference engines work: tokenization, KV cache, paged attention, "
            "continuous batching, MoE routing, speculative decoding, and quantization. "
            "Cover each topic in depth with a single consistent narrative voice.")
PROMPT_B = ("Write a thorough, coherent essay (do NOT stop early, keep going) explaining how "
            "GPUs execute programs: SIMT execution, warps, memory hierarchy, shared memory, "
            "tensor cores, kernel launches, and CUDA streams. Cover each topic in depth.")
PROMPT_C = ("Write a thorough, coherent essay (do NOT stop early, keep going) explaining how "
            "the human immune system works: innate immunity, adaptive immunity, antibodies, "
            "T cells, vaccines, and autoimmune disease. Cover each topic in depth.")


def gen_once(prompt, gen=GEN):
    body = json.dumps({"model": "GLM5.2-top2",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "max_tokens": gen, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    tf = tl = None; chunks = 0; comp = None
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        try:
            o = json.loads(p)
        except Exception:
            continue
        u = o.get("usage")
        if u:
            comp = u.get("completion_tokens")
        ch = o.get("choices", [])
        if ch:
            d = ch[0].get("delta", {})
            if d.get("reasoning_content") or d.get("content"):
                now = time.time()
                if tf is None:
                    tf = now
                tl = now; chunks += 1
    dt = (tl - tf) if (tf and tl) else 0.0
    return comp, dt, (comp / dt if comp and dt > 0 else 0.0), (comp / chunks if comp and chunks else 0.0)


def phase(name, prompt):
    speeds = []
    for r in range(RUNS):
        comp, dt, tps, al = gen_once(prompt)
        speeds.append(tps)
        print(f"  [{name}] run{r}: tokens={comp} decode_s={dt:.1f} TOK/S={tps:.1f} accept~{al:.2f}", flush=True)
    snap = f"{SNAP_DIR}/snap_{name}.pt"
    try:
        shutil.copy(MASKS, snap)
    except FileNotFoundError:
        snap = None
        print(f"  [{name}] WARNING: no mask dump yet")
    return speeds, snap


def sim(f1, f2):
    if not (f1 and f2):
        return float("nan")
    a, b = torch.load(f1), torch.load(f2)
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan")
    vals = [ (a[l] & b[l]).sum().item() / max(1, a[l].sum().item()) for l in common ]
    return sum(vals) / len(vals)


print(f"runs/phase={RUNS} gen={GEN}")
sA1, fA1 = phase("A1_llm_cold", PROMPT_A)
sA2, fA2 = phase("A2_llm_repeat", PROMPT_A)
sB, fB = phase("B_gpu_similar", PROMPT_B)
sC, fC = phase("C_bio_different", PROMPT_C)

med = lambda v: sorted(v)[len(v) // 2]
print("\n=== speeds (median tok/s per phase) ===")
print(f"A1 cold:      {med(sA1):.1f}  (last run {sA1[-1]:.1f})")
print(f"A2 repeat:    {med(sA2):.1f}  (last run {sA2[-1]:.1f})")
print(f"B similar:    {med(sB):.1f}  (last run {sB[-1]:.1f})")
print(f"C different:  {med(sC):.1f}  (last run {sC[-1]:.1f})")
print("\n=== resident-set similarity (mean |X∩Y|/96; random-baseline 0.375) ===")
print(f"A1 -> A2 (same prompt):      {sim(fA1, fA2):.3f}")
print(f"A2 -> B  (similar domain):   {sim(fA2, fB):.3f}")
print(f"A2 -> C  (different domain): {sim(fA2, fC):.3f}")
print(f"B  -> C  (different domain): {sim(fB, fC):.3f}")
