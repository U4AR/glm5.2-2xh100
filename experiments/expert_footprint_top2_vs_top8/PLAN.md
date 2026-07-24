# Experiment plan: expert working-set "size" — top-2 vs top-8

**Author:** handoff plan for the next agent. **Date:** 2026-07-03. **Repo root:** `/data/models/RunGLM`.
**Model:** GLM-5.2-W4AFP8 (KTransformers hybrid CPU/GPU MoE on 2×H100), served via SGLang OpenAI API on `http://127.0.0.1:8000`.

Read this whole file before writing code. Every file path, line number, formula, and env var you need is here.

---

## 0. What we are measuring (READ FIRST — resolves the ambiguity)

The user's request: "run a prompt through the model and see (a) the **total size of all the experts used** and (b) the **90th-percentile size** of the experts used, in two cases: **top-2 experts only** vs **all 8 experts**." Then repeat for one Terminal-Bench task, then for 5 tasks, then draw usage-distribution diagrams.

GLM-5.2 is a **top-8 / 256-expert** MoE (see §1). "top-2 vs top-8" = for each token, look only at that token's 2 highest-weight experts vs all 8 selected experts.

**"Size" is interpreted as expert-weight memory footprint**, because that is the project's real question (VRAM budget / which experts to keep resident — see the memory notes on `-topN` tiers and `top2_transfer`). We report it BOTH ways so nothing is lost:

- **Metric A — footprint (bytes):** `distinct_experts_activated × per_expert_bytes`. This is the headline "size."
- **Metric B — count:** number of distinct experts activated (the "working set"). Footprint is just count × a constant (all routed experts are identical size, §1), so A and B carry the same information; report both because the user said "size."

We compute two aggregates:

- **"Total size of all experts used"** = sum over all 75 routed-MoE layers of (distinct experts used in that layer) × per-expert-bytes. (A token in layer L can route to a different expert than in layer L+1, so the working sets are per-layer; summing gives the total distinct expert-weight volume the prompt touches.)
- **"90th-percentile size"** = the 90th percentile, across the 75 per-layer values, of (distinct experts in that layer × per-expert-bytes). This is the "a heavy layer needs about this much" number. (p90 only makes sense over a distribution of many values → the 75 layers are that distribution.)

> If, when the takeover agent reads this, the user actually meant "p90 of per-expert **load** (token counts)" instead, that is a trivial change in the analysis script (§6) — the raw trace we capture contains everything to compute either. Do NOT re-capture; just re-aggregate.

### The one-run trick (important — do NOT run the model twice per prompt)
Do **not** launch a `top2` server and a `top8` server. Capture the **full top-8 routing trace once** per prompt (every MoE layer, every token: the 8 selected expert IDs **and** their router weights). Then in analysis derive:
- **top-8 set** = distinct IDs over all 8 slots.
- **top-2 set** = distinct IDs over the 2 highest-weight slots per token (sort each row by weight, take the first 2).

This is exactly what the existing `experiments/top2_transfer/analyze_trace.py` already does (`ids[:2]` for the top-2 view) — reuse that idea. It is deterministic, cheaper, and guarantees the two cases are the same tokens.

---

## 1. Model facts (from `weights/GLM-5.2-W4AFP8/config.json` — verified 2026-07-03)

| field | value | meaning |
|---|---|---|
| `n_routed_experts` | **256** | routed experts per MoE layer |
| `num_experts_per_tok` | **8** | top-k (each token picks 8) |
| `hidden_size` | 6144 | |
| `moe_intermediate_size` | 2048 | per-expert FFN width |
| `num_hidden_layers` | 78 | |
| `first_k_dense_replace` | **3** | layers 0,1,2 are DENSE (no routed experts) |
| `n_shared_experts` | 1 | always-on shared expert (not one of the 256; constant, ignore for top-2/8) |
| `num_nextn_predict_layers` | 1 | MTP/NextN draft layer (layer ~78) — exclude from the main count |

**Routed-MoE layers = 78 − 3 = 75** (layer indices 3..77). This "75" is the population for the p90 percentile.

### Per-expert byte size (Metric A constant)
Each routed expert = gate_proj + up_proj + down_proj = `3 × (6144 × 2048)` = **37,748,736 weights**.
- **W4AFP8 int4 weights (actual on disk/GPU):** 0.5 byte/weight → **18,874,368 B = 18.0 MiB/expert.** ← use this as the primary unit.
- (fp16 reference, if asked: 2 B/weight → 72.0 MiB/expert.)
- Ignore the fp8 group scales in the headline number (they add <1%); mention them as a footnote if you want exactness.

Sanity check: full model routed-expert weight = `75 layers × 256 × 18.0 MiB ≈ 337.5 GiB`, consistent with the ~373 GB download. Good.

Put these constants in a shared `_consts.py` in this dir so the capture and analysis scripts agree.

---

## 2. Where the routing hook goes (the ONE code change)

File: `.venv/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py` (this is the live SGLang copy the server imports; it is already patched with the project's kt hooks).

Relevant existing anchors (verified line numbers 2026-07-03; re-grep if drifted):
- `430  def _kt_dump_route_mass(topk_output):` — an **existing** eager-only dump hook. **Copy its pattern** (esp. the `torch.cuda.is_current_stream_capturing()` early-return guard).
- Env/flag setup around lines **424–426**: `_KT_ROUTE_MASS_FLAG`, `_KT_DUMP_ROUTE_MASS`, `_KT_ROUTE_MASS_FILE`.
- The MoE forward call site, lines **990–998** inside `DeepseekV2MoE.forward_normal`:
  ```python
  topk_output = self.topk(hidden_states, router_logits, **topk_kwargs)
  if _KT_TOPK_MODE is not None:
      _kt_topk_experiment(topk_output, router_logits, ..., keep_k=getattr(forward_batch, "kt_keep_k", None))
  elif _KT_DUMP_ROUTE_MASS:
      _kt_dump_route_mass(topk_output)
  ```
  `topk_output.topk_ids` is `(T, 8)` int, `topk_output.topk_weights` is `(T, 8)` float. `T` = tokens in this forward (prefill: whole prompt; decode: 1).

### The hook to add (`_kt_dump_topk`)
Add near `_kt_dump_route_mass`, gated by env `KT_DUMP_TOPK=1`:

```python
import os, torch
_KT_DUMP_TOPK       = os.environ.get("KT_DUMP_TOPK", "0") == "1"
_KT_DUMP_TOPK_DIR   = os.environ.get("KT_DUMP_TOPK_DIR", "/data/models/RunGLM/experiments/expert_footprint_top2_vs_top8/runs")
_KT_DUMP_TOPK_TAGF  = os.environ.get("KT_DUMP_TOPK_TAG", "/tmp/kt_topk_tag")  # driver writes the current task id here
_kt_topk_state = {"tag": None, "buf": []}   # buf: list of (layer_idx, ids_cpu, weights_cpu)

def _kt_dump_topk(topk_output, layer_idx):
    try:
        if torch.cuda.is_current_stream_capturing():   # never touch python inside a CUDA graph
            return
        ids = getattr(topk_output, "topk_ids", None)
        w   = getattr(topk_output, "topk_weights", None)
        if ids is None or ids.numel() == 0:
            return
        # rotate output file when the driver changes the tag (one .pt per task/prompt)
        try:
            tag = open(_KT_DUMP_TOPK_TAGF).read().strip()
        except Exception:
            tag = "default"
        st = _kt_topk_state
        if tag != st["tag"]:
            _kt_flush_topk()          # save previous task's buffer
            st["tag"] = tag
            st["buf"] = []
        st["buf"].append((int(layer_idx), ids.detach().to("cpu", torch.int32).clone(),
                          w.detach().float().cpu().clone()))
        _kt_flush_topk()              # save-after-each so a killed run still has data
    except Exception:
        pass

def _kt_flush_topk():
    st = _kt_topk_state
    if st["tag"] and st["buf"]:
        os.makedirs(_KT_DUMP_TOPK_DIR, exist_ok=True)
        torch.save(st["buf"], os.path.join(_KT_DUMP_TOPK_DIR, f"{st['tag']}.pt"))
```

Wire it at the call site (line ~997), and pass the layer index. The layer id is available on the module — `DeepseekV2MoE` is built with `layer_id`; store it as `self.layer_id` in `__init__` if not already, or read `prefix`. Add:
```python
elif _KT_DUMP_ROUTE_MASS:
    _kt_dump_route_mass(topk_output)
elif _KT_DUMP_TOPK:
    _kt_dump_topk(topk_output, getattr(self, "layer_id", -1))
```
Check whether `self.layer_id` exists (grep `layer_id` in the class ~line 660–740); if only `layer_idx`/`prefix` exists, adapt. Saving a wrong-but-consistent id is fine as long as it's unique per layer — worst case parse it from `prefix` (e.g. `"model.layers.7.mlp"`).

**Format saved:** `list[ (layer_idx:int, ids:int32[T,8], weights:float32[T,8]) ]` per forward call, appended across the whole request. This matches `top2_transfer/analyze_trace.py`'s loader (`list of (layer_idx, tensor[T,top_k])`), extended with a weights tensor — reuse its grouping code.

### CRITICAL gotcha — CUDA graphs (decode)
Decode steps replay a captured CUDA graph; a Python hook will **not** fire during replay, and it early-returns during capture. So to capture **decode** routing you MUST launch the server with **`DISABLE_CUDA_GRAPH=1`** (flag already supported: `run_server_int4.sh:74-84` → `--disable-cuda-graph`). Prefill runs eagerly regardless, so **prefill routing is captured even with graphs on**. See §3 on whether you need decode at all.

---

## 3. Do we need generation, or just the prompt? (scope decision)

"Run a prompt through the model" has two readings:
- **Prefill-only (recommended default):** one forward over the prompt tokens → captures which experts *the prompt itself* activates. One eager pass, `max_tokens=1`, no CUDA-graph worry, fast. This is the cleanest, most reproducible "pass a task through the model."
- **Prefill + decode:** also capture the experts used while *generating the answer* (`max_tokens=128`). Requires `DISABLE_CUDA_GRAPH=1`. Richer but slower and answer-length-dependent.

**Recommendation:** do **prefill-only** for the headline results (deterministic, prompt-defined), and optionally add a `--with-decode` variant. The analysis script handles both (it just gets more `T=1` rows). Terminal-Bench task prompts are long (hundreds–thousands of tokens), so prefill alone already gives a substantial, meaningful working set.

---

## 4. Server launch for capture

Start (or restart) the fast server with the dump hook on. Keep everything else at production defaults.

```bash
cd /data/models/RunGLM
# make sure no -topN routing rewrite is active: the dump hook path only runs
# when _KT_TOPK_MODE is None. Remove the sentinel if present:
rm -f /tmp/kt_topk_mode
echo default > /tmp/kt_topk_tag          # initial tag

KT_DUMP_TOPK=1 \
KT_DUMP_TOPK_DIR=/data/models/RunGLM/experiments/expert_footprint_top2_vs_top8/runs \
KT_DUMP_TOPK_TAG=/tmp/kt_topk_tag \
DISABLE_CUDA_GRAPH=1 \
KT_GPU_PREFILL_THRESHOLD=0 \
TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
./run_fast.sh > experiments/expert_footprint_top2_vs_top8/server.log 2>&1 &
```
Notes:
- `_KT_TOPK_MODE`/`_KT_DUMP_ROUTE_MASS` are read at import from sentinel files (`deepseek_v2.py:424-425`, `:466`). The dump hook is the `elif` branch, so it only runs when **neither** `KT_TOPK_MODE_FILE` (`/tmp/kt_topk_mode`) nor route-mass dumping is active. Delete `/tmp/kt_topk_mode` first (see above).
- `DISABLE_CUDA_GRAPH=1` only needed if capturing decode. For prefill-only you may leave graphs ON (faster boot/serve) — but simplest is to keep it off for the whole experiment.
- `KT_GPU_PREFILL_THRESHOLD=0` forces CPU prefill so long prompts don't OOM VRAM (memory note: long-ctx prefill VRAM OOM). Safe here; we don't care about prefill speed.
- Wait for `Uvicorn running on ... :8000` in `server.log` before driving. Boot ~3 min (int4 weights already persistent at `/data/models/glm52-w4afp8`, symlinked `weights/`).

---

## 5. The four experiments

All driven by a single script `capture.py` (write it in this dir). It sets the tag file, sends ONE non-streaming request, waits, confirms the `.pt` appeared. Sequential only (never concurrent — concurrency interleaves tokens from different requests into the same trace).

Driver skeleton:
```python
import json, urllib.request, time, pathlib
BASE="http://127.0.0.1:8000"; TAGF="/tmp/kt_topk_tag"
RUNS=pathlib.Path("/data/models/RunGLM/experiments/expert_footprint_top2_vs_top8/runs")
def served():
    return json.load(urllib.request.urlopen(BASE+"/v1/models"))["data"][0]["id"]
def run(tag, prompt, max_tokens=1):
    pathlib.Path(TAGF).write_text(tag)                 # rotate trace file
    body=json.dumps({"model":served(),"messages":[{"role":"user","content":prompt}],
                     "temperature":0,"max_tokens":max_tokens,"stream":False}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=body,
                               headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req,timeout=600).read()
    pathlib.Path(TAGF).write_text(tag+"_FLUSH")        # force final flush of this tag
    time.sleep(1.0)
    assert (RUNS/f"{tag}.pt").exists(), f"no trace for {tag}"
```
(The `_FLUSH` tag change makes the hook flush the completed buffer before the next prompt; harmless empty `{tag}_FLUSH.pt` files can be ignored/cleaned.)

### Experiment 1 — single hand-written prompt
Use one clear, non-trivial prompt (reuse the project's canonical one for continuity):
```
"Write a short coherent paragraph about why the sky is blue. Then list the first 8 prime numbers."
```
`run("exp1_single", PROMPT)`. → `runs/exp1_single.pt`.

### Experiment 2 — one Terminal-Bench task
Terminal-Bench task prompts come from the `terminal-bench-2` repo. It may not be cloned yet.
- Cloned location (if present): `$TB_DIR/terminal-bench-2/tasks/<task-name>/` where `TB_DIR` defaults to `$REPO/.terminalbench` (see `bench/run_terminalbench.sh:38-45`). Legacy: `/data/projects/isolated_bench/terminal-bench-2/`.
- If NOT cloned, get just the prompts cheaply (no Docker/Harbor needed):
  ```bash
  git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git \
      /data/tmp/tb2   # or run bench/terminalbench/setup_terminalbench.sh to pre-warm
  ls /data/tmp/tb2/tasks | head
  ```
  Each task dir has an instruction file (commonly `task.yaml` with an `instruction:` field, or `instruction.md` / a `task/` prompt). **Inspect one task dir** and extract the natural-language instruction string that the agent receives — that string is the "task" you feed as the prompt.
- Pick a **passing** task from `bench/terminalbench/task_labels.txt` so it's a well-formed, model-solvable prompt. Good candidates (present in labels): `llm-inference-batching-scheduler`, `largest-eigenval`, `fix-git`, `compile-compcert`, `count-dataset-tokens`.
- `run("exp2_<taskname>", task_instruction_text)`.

### Experiment 3 — 5 Terminal-Bench tasks
Pick 5 varied passing tasks, e.g.:
`llm-inference-batching-scheduler`, `largest-eigenval`, `fix-git`, `compile-compcert`, `git-multibranch`.
Loop: `for name in TASKS: run(f"exp3_{name}", instruction_text[name])`. → five `.pt` files.
(Experiment 2 is just the first of these — you can fold 2 into 3 and label the first task as the "single task" case.)

### Experiment 4 — diagrams
See §7.

---

## 6. Analysis script (`analyze.py`)

Reuse `experiments/top2_transfer/analyze_trace.py` as the structural template (it already groups records by layer and takes `ids[:2]`). For **each** `.pt` file compute, per interpretation:

For each trace file:
1. Load `list[(layer_idx, ids[T,8], weights[T,8])]`. Group by `layer_idx`.
2. For each layer, over all its token rows:
   - Sort each row's 8 slots by weight descending (so column 0,1 = the true top-2). *Note:* router `topk_ids` are usually already weight-descending, but sort explicitly using the captured weights to be safe.
   - **top-8 distinct set** = `set()` of all 8 ids across all rows in the layer.
   - **top-2 distinct set** = `set()` of the first 2 ids (by weight) across all rows.
   - Also accumulate a **per-expert token-count histogram** (Counter over ids) for both top-2 and top-8 (needed for diagrams and the alt p90-of-load interpretation).
3. Per-layer values: `distinct_count` and `footprint = distinct_count × 18.0 MiB` for each of top-2 and top-8.
4. Report per trace:
   - **Total size** (Metric A) top-2 and top-8 = `Σ_layers footprint`. Also **Total distinct-count** (Metric B) = `Σ_layers distinct_count`.
   - **p90 size** = `numpy.percentile([per-layer footprint], 90)` over the 75 layers; likewise p90 count. (Use nearest-rank or linear — state which; `numpy.percentile(...,90)` default linear is fine, note it.)
   - Also print min/median/max per-layer for context, and the top-8÷top-2 ratio.
5. Write a machine-readable `runs/<tag>.summary.json` and a human table `results.md`.

Put the constants (256 experts, 8 top-k, 75 layers, 18.0 MiB/expert, 6144/2048 dims) in `_consts.py`. numpy is available; matplotlib is **NOT** (see §7).

Expected shape of results (hypothesis, verify): for a short prompt, top-2 touches far fewer distinct experts/layer than top-8; as tokens grow, top-8 per-layer sets saturate toward 256 faster than top-2. The p90 layer is the "hottest" layer's footprint.

---

## 7. Diagrams (Experiment 4)

**matplotlib is NOT installed** in `.venv` (verified). Two options:
1. **Self-contained SVG** (recommended, matches repo style): copy the generator pattern in
   `experiments/logit_dist_top2_vs_top8/make_distribution_chart.py` — it hand-writes an SVG histogram (fonts, grid, overlaid bars, `<title>` tooltips) with zero deps. Adapt it for expert-usage.
2. `pip install matplotlib` into `.venv` if you prefer — allowed, but keep the SVG path as the portable default.

**Before writing any chart code, load the `dataviz` skill** (per its trigger) to get the palette/mark/legend rules, then follow the existing SVG file's structure.

Diagrams to produce (save under `runs/figs/`), one set per task plus a combined view:
- **A. Expert-usage distribution per task** (the user's explicit ask): for each task, a bar/line chart of **per-expert token counts sorted descending** (a "load curve" over the 256 experts, aggregated across layers OR for a representative layer), with top-2 and top-8 overlaid (like the top2/top8 overlay in the reference SVG). Shows how concentrated routing is.
- **B. Per-layer distinct-expert count** across the 75 layers (x=layer, y=distinct count), top-2 vs top-8 lines, with the p90 line marked. One per task or small-multiples for the 5 tasks.
- **C. Summary bar chart:** total footprint (MiB) top-2 vs top-8 for each of the (1 + 5) tasks side by side.
- Optionally a **Lorenz curve / cumulative load** to visualize concentration.

Label axes, units (MiB, expert count), and add a caption stating model/config and that top-2 = the 2 highest-weight of the 8 selected. Save each as `.svg` (and optionally render to `.png` if a converter is available — `rsvg-convert`/`cairosvg`; check first).

---

## 8. Output layout
```
experiments/expert_footprint_top2_vs_top8/
  PLAN.md                <- this file
  _consts.py             <- model constants + per_expert_bytes
  capture.py             <- driver (§5)
  analyze.py             <- aggregation (§6)
  make_charts.py         <- SVG diagrams (§7)
  server.log
  runs/
    exp1_single.pt
    exp3_<task>.pt  (×5, one reused as exp2)
    *.summary.json
    results.md
    figs/*.svg
```

## 9. Gotchas / checklist
- [ ] `rm -f /tmp/kt_topk_mode` before launch, else `_KT_TOPK_MODE` != None and the dump `elif` never runs.
- [ ] Confirm the hook actually fires: after one request, `ls runs/*.pt` non-empty and `torch.load` shows ~75 layer records per forward.
- [ ] **Sequential requests only.** No concurrency during capture.
- [ ] Decode capture needs `DISABLE_CUDA_GRAPH=1`; prefill works either way. Default plan = prefill-only.
- [ ] Exclude the NextN/MTP layer and the 3 dense layers from the "75 layers" population (routed layers are indices 3..77; the dense layers won't emit routed topk anyway, but verify the layer_ids you capture).
- [ ] Per-expert size uses **int4 = 18.0 MiB** (W4AFP8). Don't accidentally use fp16.
- [ ] `served()` model id: use the bare id from `/v1/models` (no `-topN` suffix) so no tier rewrite happens.
- [ ] If server perf/behavior looks wrong, check for a runaway process (`ps --sort=-pcpu`) — a known landmine (memory: gpu-prefill note).
- [ ] Restore normal serving afterwards: relaunch without `KT_DUMP_TOPK`/`DISABLE_CUDA_GRAPH` (or `./run_fast.sh`), and remove the temporary `_kt_dump_topk` patch or leave it dormant (it's env-gated & CUDA-graph-safe, so leaving it in is harmless).

## 10. Key source references
- Routing hook site + patterns: `.venv/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py` — `_kt_dump_route_mass` (L430), flags (L424-426), MoE call site (L990-998), `_kt_topk_experiment` (L529).
- Trace analysis template: `experiments/top2_transfer/analyze_trace.py`.
- SVG chart template: `experiments/logit_dist_top2_vs_top8/make_distribution_chart.py`.
- Server launchers: `run_fast.sh`, `run_server_int4.sh` (cuda-graph flag L74-84).
- Terminal-Bench: `bench/run_terminalbench.sh`, `bench/terminalbench/setup_terminalbench.sh`, task labels `bench/terminalbench/task_labels.txt`. Tasks repo: `https://github.com/laude-institute/terminal-bench-2.git`.
- Model config: `weights/GLM-5.2-W4AFP8/config.json`.
