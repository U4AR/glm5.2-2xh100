#!/usr/bin/env python3
"""Measure one point of the three-tier ladder.

Reports, for the running server:
  - decode tok/s  (usage.completion_tokens / streaming wall time, NOT chunk
    counts -- under MTP one chunk carries accept_len tokens)
  - host RSS of the TP0 scheduler, which is what the RAM tier actually costs
  - genuine top-2 tier coverage, from the per-layer demand counters the server
    dumps (KT_ADAPTIVE_COUNTS_DUMP_PT) intersected with the tier masks

Usage:
  tier_bench.py <label> [passes] [gen_tokens] [topic]

Bench discipline: run at least two passes and read the LAST one. The first pass
warms the page cache and, once Phase 1 lands, lets the cache adapt to the bench
prompt.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.environ.get("TIER_BENCH_BASE", "http://127.0.0.1:8000")

PROMPTS = {
    "llm": (
        "Write a thorough, coherent essay (do NOT stop early, keep going) explaining "
        "how modern LLM inference engines work: tokenization, KV cache, paged attention, "
        "continuous batching, MoE routing, speculative decoding, and quantization. "
        "Cover each topic in depth with a single consistent narrative voice."
    ),
    "bio": (
        "Write a thorough, coherent essay (do NOT stop early, keep going) explaining how "
        "the human immune system works: innate immunity, adaptive immunity, antibodies, "
        "T cells, vaccines, and autoimmune disease. Cover each in depth."
    ),
    "hist": (
        "Write a thorough, coherent essay (do NOT stop early, keep going) on the causes "
        "and consequences of the Industrial Revolution in Britain: enclosure, coal, steam, "
        "textiles, urbanisation, labour, and empire. Cover each in depth."
    ),
}


def run_pass(prompt, gen, model="GLM5.2-top2"):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": gen,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    tf = tl = None
    chunks = 0
    comp = None
    text = []
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
            piece = d.get("content") or d.get("reasoning_content")
            if piece:
                now = time.time()
                if tf is None:
                    tf = now
                tl = now
                chunks += 1
                text.append(piece)
    dt = (tl - tf) if (tf and tl) else 0.0
    return {
        "completion_tokens": comp or 0,
        "decode_s": dt,
        "tok_s": (comp / dt) if (comp and dt > 0) else 0.0,
        "accept_len": (comp / chunks) if (comp and chunks) else 0.0,
        "text": "".join(text),
    }


def tp0_rss_gb():
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "rss,args"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    for line in out.splitlines():
        if "scheduler_TP0" in line:
            return int(line.split()[0]) / (1024 * 1024)
    return None


def tier_coverage():
    """Genuine-demand coverage per tier, from the server's counter dump and the
    resident-mask dump. Returns None when the dumps are not available."""
    counts_pt = os.environ.get("KT_ADAPTIVE_COUNTS_DUMP_PT", "/tmp/kt_tier_counts.pt")
    masks_pt = os.environ.get("KT_ADAPTIVE_DUMP_PT", "/tmp/kt_adaptive_masks.pt")
    if not (os.path.exists(counts_pt) and os.path.exists(masks_pt)):
        return None
    try:
        import torch

        counts = torch.load(counts_pt, map_location="cpu", weights_only=True)["counts"]
        gpu_masks = torch.load(masks_pt, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {"error": str(exc)}
    n_ram = int(os.environ.get("KT_RAM_EXPERTS", "-1"))
    rank_pt = os.environ.get("KT_HOTCORE_RANKING_PT", "")
    ranking = None
    if n_ram >= 0 and rank_pt and os.path.exists(rank_pt):
        import torch

        ranking = torch.load(rank_pt, map_location="cpu", weights_only=True).long()
    gpu_tot = ram_tot = ssd_tot = 0.0
    for li, c in counts.items():
        c = c.float()
        total = float(c.sum())
        if total <= 0 or li not in gpu_masks:
            continue
        gpu = gpu_masks[li].bool()
        if ranking is not None and li < ranking.shape[0]:
            import torch

            ram = torch.zeros_like(gpu)
            order = ranking[li]
            eligible = order[~gpu[order]]
            ram[eligible[:n_ram]] = True
        else:
            ram = ~gpu
        gpu_tot += float(c[gpu].sum())
        ram_tot += float(c[ram].sum())
        ssd_tot += float(c[~(gpu | ram)].sum())
    tot = gpu_tot + ram_tot + ssd_tot
    if tot <= 0:
        return None
    return {"gpu": gpu_tot / tot, "ram": ram_tot / tot, "ssd": ssd_tot / tot}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    gen = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    topic = sys.argv[4] if len(sys.argv) > 4 else "llm"

    print(f"== {label} == passes={passes} gen={gen} topic={topic}")
    results = []
    for p in range(passes):
        r = run_pass(PROMPTS[topic], gen)
        results.append(r)
        print(
            f"  pass {p}: {r['tok_s']:6.2f} tok/s  "
            f"({r['completion_tokens']} tok in {r['decode_s']:.1f}s, "
            f"accept_len {r['accept_len']:.2f})"
        )
    last = results[-1]
    rss = tp0_rss_gb()
    cov = tier_coverage()
    summary = {
        "label": label,
        "gpu_experts": int(os.environ.get("GPU_EXPERTS", "-1")),
        "ram_experts": int(os.environ.get("KT_RAM_EXPERTS", "-1")),
        "fill_pool": os.environ.get("KT_TIER_FILL_POOL", "resident"),
        "count_mode": os.environ.get("KT_TIER_COUNT_MODE", "top2"),
        "tok_s": round(last["tok_s"], 2),
        "tok_s_all": [round(r["tok_s"], 2) for r in results],
        "accept_len": round(last["accept_len"], 3),
        "tp0_rss_gb": round(rss, 1) if rss else None,
        "coverage": {k: round(v, 4) for k, v in cov.items()} if cov and "error" not in cov else cov,
        "sample": last["text"][:400],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, indent=2))
    out = os.environ.get("TIER_BENCH_OUT", "experiments/expert_tiering_ssd/results.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\n--- first 400 chars of the last completion (coherence eyeball) ---\n{last['text'][:400]}")


if __name__ == "__main__":
    main()
