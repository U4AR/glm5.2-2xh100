#!/usr/bin/env python3
"""Zero-dependency control panel for the expert-streaming experiments.

Every knob that shapes the hybrid MoE path, in one page, with the measured
numbers next to the presets so a configuration can be compared against what it
scored rather than against a memory of it.

WHY THIS OWNS THE SERVER. Decode runs under a captured CUDA graph, so nearly
every knob here is read ONCE, at import or at capture, and cannot change in a
live process: the predictor's layer stride decides which layers get graph nodes,
the landing-slot count sizes a tensor, the cache decides how the kt store is
staged at load. Changing them means rebooting. The one genuine exception is the
expert TIER, which is per-request (the `<base>-topN` model field, backed by a
mode file), and it is marked as such below.

    python3 lab_ui.py            # serves on :8090
    python3 lab_ui.py 9100       # or a port of your choosing

Then open the forwarded port. The model server itself stays on :8000.
"""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
REPO = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE = "http://127.0.0.1:8000"
SCRATCH = os.environ.get("RUNGLM_SCRATCH") or os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "runglm-bench")
os.makedirs(SCRATCH, exist_ok=True)
BOOT_LOG = os.path.join(SCRATCH, "lab_boot.log")
MODE_FILE = os.path.join(SCRATCH, "lab_topk_mode")

# ---------------------------------------------------------------------------
# The knobs. `help` is the measured reason the knob exists, not a restatement
# of its name -- a panel that says "slots: number of slots" is a worse version
# of the env var it wraps.
# ---------------------------------------------------------------------------
KNOBS = [
    # group, name, default, kind, help
    ("Routing", "KT_GPU_ONLY", "1", "bool",
     "Every expert the layer computes must already be on the GPU: resident, or "
     "landed by the prefetcher. A genuine top-K expert that is neither gets "
     "substituted from the resident pool instead of taking a CPU round-trip, so "
     "the CPU expert path is zero BY CONSTRUCTION. This is what makes the floor "
     "53.4 ms instead of 146."),
    ("Routing", "RUNGLM_TOPK_MODE", "safe", "text",
     "safe = keep the genuine top-K always, substitute only the tail. The old "
     "residency-forcing 'sub' mode is removed; never reintroduce it."),
    ("Routing", "TIER", "8", "tier",
     "DIAGNOSTIC ONLY -- not a setting to tune. Under KT_GPU_ONLY the routing "
     "mask is `rank < K` AND `on_gpu`, and the model routes top-8, so K=8 makes "
     "the first term vacuous: every expert that is resident or landed is "
     "computed, everything else is substituted, nothing reaches the CPU. That is "
     "the whole design. LOWERING K MOVES NOTHING TO THE CPU -- it throws away "
     "genuine experts already sitting on the GPU for free, at identical speed "
     "(56.34/56.34/56.23/56.33 at 8/4/2/0). Leave it at 8. The lower tiers exist "
     "only as instruments: top0 keeps nothing and MUST produce garbage (the "
     "coherence detector's calibration case -- if it scores clean, no other row "
     "on that boot may be read as a pass), and top2 is a deliberate starvation "
     "probe for measuring how much quality margin a configuration has."),

    ("Prefetch", "KT_PREFETCH_SLOTS", "1", "int",
     "Landing slots per layer: trailing rows of the cutlass tensor the gather "
     "writes into. A SLOT IS STRICTLY WORSE THAN A RESIDENT EXPERT -- it costs a "
     "resident, costs ms per fetch, and only pays when the prediction is right. "
     "Measured at constant VRAM: 4 slots 61.6, 8 slots 85.1, 16 slots 132.3. "
     "Fewest wins; 1 is the operating point."),
    ("Prefetch", "KT_PREFETCH_REUSE", "2", "int",
     "A slot keeps whatever it last held, across steps and graph replays. Mode 2 "
     "re-routes a still-held expert instead of re-fetching it: 63-69% of coverage "
     "arrives free. This is where the prefetch's cheapness comes from, and it is "
     "why a churning resident set makes the prefetch MORE expensive."),
    ("Prefetch", "KT_PREFETCH_BLOCKS", "8", "int",
     "Gather grid. Swept 1/2/4/6/8/12/16/32: monotone into 8 from both sides "
     "(66.9/63.0/62.5/61.95/62.5/64.6). Already optimal -- there is no free grid "
     "win here."),
    ("Prefetch", "KT_PREFETCH_GATHER", "1", "bool",
     "Move the bytes. Off = predict and publish but transfer nothing, which is "
     "the timing probe that isolates the predictor's cost from the copy's."),
    ("Prefetch", "KT_PREFETCH_ROUTE", "1", "bool",
     "Let the GPU compute a landed expert. Off with CPUSKIP on is numerically "
     "wrong (dropped contribution) and is a probe only."),
    ("Prefetch", "KT_PREFETCH_CPUSKIP", "1", "bool",
     "Let the CPU stop computing an expert the gather landed. Irrelevant under "
     "KT_GPU_ONLY, where nothing runs on the CPU anyway."),
    ("Prefetch", "KT_PREFETCH_SELECTIVE", "0", "bool",
     "All-or-nothing rule: fetch only if the whole layer's demand fits the slots. "
     "It assumes a CPU call still happens; under GPU-only there is none, so "
     "partial coverage is pure quality and this must be 0."),

    ("Predictor", "KT_PRED_FUSED", "1", "bool",
     "One kernel for the whole selection instead of ~28 launches on four-token "
     "tensors. The unfused path is a probe, not a configuration."),
    ("Predictor", "KT_PRED_LAYER_STRIDE", "1", "int",
     "Predict on every Nth layer. THE lever on the predictor's 1.81 ms/step: that "
     "is ~24 us per layer for one kernel on four tokens, charged on all 75 layers "
     "whether or not the layer needed anything -- and a captured graph replays "
     "every node regardless. Decided before capture, so skipped layers have no "
     "nodes at all. Costs ~1/N of the predictor and buys ~1/N of the coverage; "
     "skipped misses fall back to substitution, a QUALITY cost, so score "
     "coherence on any row you intend to keep."),
    ("Predictor", "KT_PRED_P", "8", "int",
     "How many experts the prediction names. Wider set, better recall, more bytes."),
    ("Predictor", "KT_PRED_POINT", "pre", "text",
     "pre = predict from the hidden state before this layer's MoE; post = from "
     "the post-layer residual (better coverage, less lead time). Measured worth "
     "0.32 ms, not the 3.1 ms once claimed -- that number came from subtracting "
     "across two boots and is retracted."),
    ("Predictor", "KT_PRED_FUSED_DEPTH", "1", "int",
     "How many layers ahead. Depth 2 predicts WORSE here (70.1% vs 80.6% "
     "per-expert), and the bytes already fit inside one layer's shadow, so depth "
     "is a wash at best."),

    ("Cache", "KT_ADAPTIVE_DECODE", "0", "bool",
     "Promote/evict GPU-resident experts from live decode routing. NEEDS ~372 GiB "
     "of /dev/shm: it stages all 256 experts per layer so an evicted one still has "
     "weights. Measured alone: 54.31 ms/step. Measured WITH the prefetch: 57.03, "
     "i.e. it costs 0.56 ms of blocking swaps and RAISES the fetch rate, because "
     "reuse needs a stable resident set and this keeps changing it."),
    ("Cache", "KT_ADAPTIVE_PERIOD", "32", "int",
     "Steps between ticks. Swap cost is strictly proportional: 2 layers x ~9 ms "
     "per tick / 32 steps = ~0.56 ms/step. Raising this is the direct lever on "
     "the cache's price, at the cost of slower adaptation."),
    ("Cache", "KT_ADAPTIVE_LAYERS_PER_TICK", "2", "int",
     "Layers re-selected per tick, round-robin. 75 MoE layers / 2 per tick / 32 "
     "steps = ~1200 steps to cover the model once. Warm up at least that long "
     "before timing a cache row."),
    ("Cache", "KT_ADAPTIVE_MAX_SWAP", "8", "int",
     "Experts moved per layer per tick. KEEP THIS <= 16: above it the stable-slot "
     "swap declines and falls back to the full restage, which renumbers every "
     "slot and re-copies ~100 experts (144 ms vs 9 ms)."),
    ("Cache", "KT_ADAPTIVE_INCREMENTAL", "1", "bool",
     "Stable-slot swap: give each arrival the slot its departing partner "
     "vacated. Off = full restage, kept only as a diagnostic."),
    ("Cache", "KT_TIER_COUNT_MODE", "top8", "text",
     "Which firings vote for residency. top2 counts only what would have cost a "
     "CPU round-trip; top8 lets every routed expert vote, which is what GPU-only "
     "wants since residency is the only way to keep a genuine expert."),

    ("Capacity", "GPU_EXPERTS", "100", "int",
     "Resident experts per layer per card, ~9.56 MiB each. The ONLY lever on the "
     "weight-loading phase -- mem-fraction does not touch it, because the KV pool "
     "it sizes is not allocated until after the load."),
    ("Capacity", "MEM_FRACTION", "0.94", "float",
     "0.95 leaves ~150 MiB and OOMs when anything else wants room. Comparisons "
     "across different values are invalid; keep it fixed within a ladder."),
    ("Capacity", "MAX_TOTAL_TOKENS", "81920", "int",
     "KV pool size. MLA fp8 KV is 43.9 KB/token."),
    ("Capacity", "MTP", "1", "bool",
     "Speculative decode, depth 3. Note accept length is NOT a quality proxy: "
     "degenerate output inflates it (top0 reaches 3.70 while producing garbage)."),
    ("Capacity", "KT_STORE_SHM", "1", "bool",
     "The kt store lives in a named POSIX segment. REQUIRED by the prefetch: the "
     "gather maps it and reads experts straight out of host memory, and without "
     "it only rank 0 could gather -- one PCIe link cannot carry the traffic."),
    ("Capacity", "KT_GPU_PREFILL_THRESHOLD", "0", "int",
     "0 forces CPU prefill. The GPU bulk path needs ~5 GB of spare VRAM, which "
     "does not exist at 100 resident experts; leaving it on is a boot-time OOM."),
]

_OFF = {"KT_PREFETCH_SLOTS": "0", "KT_PREFETCH_GATHER": "0", "KT_PREFETCH_ROUTE": "0",
        "KT_PREFETCH_CPUSKIP": "0", "KT_PRED_FUSED": "0"}
_PF = {"KT_PREFETCH_SLOTS": "1", "KT_PREFETCH_REUSE": "2", "KT_PREFETCH_GATHER": "1",
       "KT_PREFETCH_ROUTE": "1", "KT_PREFETCH_CPUSKIP": "1", "KT_PRED_FUSED": "1"}

# Ordered worst-informed to best-informed. Each one exists to SEPARATE two
# explanations; a preset that cannot change your mind about anything is just a
# saved form.
PRESETS = {
    "1. floor -- resident only, nothing moves": {
        "env": dict(_OFF, KT_GPU_ONLY="1", KT_ADAPTIVE_DECODE="0"),
        "asks": "How fast can this possibly be? Keep every genuine expert that "
                "happens to be resident, substitute the rest, move nothing.",
        "expect": "53.48 ms/step, top8 CLEAN -- that is the operating result. "
                  "The probes: top2 degenerate, top0 degenerate (calibration "
                  "passes), so the coherence cliff sits between ~1.6 and ~3.2 "
                  "genuine experts per token. This configuration keeps ~3.2, "
                  "with no margin to spare.",
    },
    "2. prefetch only -- predictor + 1 slot + reuse": {
        "env": dict(_PF, KT_GPU_ONLY="1", KT_PRED_LAYER_STRIDE="1",
                    KT_ADAPTIVE_DECODE="0"),
        "asks": "What does MOVING experts buy over the floor? This is the best "
                "operating point the movement ladder found.",
        "expect": "56.47 ms/step (+2.99 over floor: predictor 1.81, gather ~1.2). "
                  "0.31 fetched/call, 69% of coverage free via reuse. Operating "
                  "quality (top8) is clean here AND at the floor -- what the "
                  "movement buys shows up only in the top2 STARVATION PROBE, "
                  "which goes degenerate -> clean. That is margin, not output "
                  "you would ever see at top8.",
    },
    "3. cache only -- residency adapts, nothing streams": {
        "env": dict(_OFF, KT_GPU_ONLY="1", KT_ADAPTIVE_DECODE="1",
                    KT_ADAPTIVE_PERIOD="32", KT_ADAPTIVE_MAX_SWAP="8"),
        "asks": "Can changing WHICH experts are resident do the job that moving "
                "experts does, without the transfer? THIS ROW'S QUALITY HAS "
                "NEVER BEEN SCORED -- it is the most valuable coherence run "
                "available right now.",
        "expect": "54.31 ms/step measured (+0.83 over floor = blocking swaps). "
                  "Coherence UNKNOWN. If top2 comes back clean here, this beats "
                  "the prefetch on both axes.",
    },
    "4. cache + prefetch -- both live, full cost": {
        "env": dict(_PF, KT_GPU_ONLY="1", KT_PRED_LAYER_STRIDE="1",
                    KT_ADAPTIVE_DECODE="1", KT_ADAPTIVE_PERIOD="32",
                    KT_ADAPTIVE_MAX_SWAP="8"),
        "asks": "Do the two mechanisms compose? Measured answer: no -- they "
                "fight. The cache reshuffles residency; reuse needs it stable.",
        "expect": "57.03 ms/step, WORSE than prefetch alone. wanted/call falls "
                  "12.45->10.84 but fetched/call RISES 0.31->0.37 and reuse "
                  "drops 68.7%->63.4%. top8 and top2 both clean, gate valid.",
    },
    "5. cheap both -- stride 4, tick every 128  [UNMEASURED]": {
        "env": dict(_PF, KT_GPU_ONLY="1", KT_PRED_LAYER_STRIDE="4",
                    KT_ADAPTIVE_DECODE="1", KT_ADAPTIVE_PERIOD="128",
                    KT_ADAPTIVE_MAX_SWAP="8"),
        "asks": "The 54 ms candidate: keep both mechanisms but make each cheap. "
                "Predictor on 1 layer in 4, cache ticking 4x less often.",
        "expect": "PREDICTION, NOT A RESULT: ~54.3 ms (floor 53.48 + cache 0.14 "
                  "+ predictor 0.45 + gather ~0.2). Quality is the risk -- three "
                  "layers in four lose their fetch, so score coherence before "
                  "believing any speed number here.",
    },
    "6. cheap both, harder -- stride 6, tick every 192  [UNMEASURED]": {
        "env": dict(_PF, KT_GPU_ONLY="1", KT_PRED_LAYER_STRIDE="6",
                    KT_ADAPTIVE_DECODE="1", KT_ADAPTIVE_PERIOD="192",
                    KT_ADAPTIVE_MAX_SWAP="8"),
        "asks": "If mode 5 lands above 54, this is how much further the same two "
                "knobs can go before the prefetch stops meaning anything.",
        "expect": "PREDICTION: ~54.0 ms. At this stride only 12 of 75 layers "
                  "prefetch at all, so ask whether the remaining coverage is "
                  "worth the machinery -- mode 1 is 53.48 with none of it.",
    },
}

JOBS = {}          # id -> {"name","state","out","started"}
JOB_LOCK = threading.Lock()


def _defaults():
    return {name: default for _g, name, default, _k, _h in KNOBS}


def shm_free_gib():
    try:
        st = os.statvfs("/dev/shm")
        return st.f_blocks * st.f_frsize / (1 << 30)
    except Exception:
        return -1.0


def preflight(env):
    """Refuse, in words, the failures this project has actually hit."""
    problems = []
    if env.get("KT_ADAPTIVE_DECODE") == "1":
        size = shm_free_gib()
        if 0 < size < 400:
            problems.append(
                f"/dev/shm is {size:.0f} GiB. The cache stages all 256 experts per "
                f"layer (~372 GiB); overrunning a tmpfs is SIGBUS, which shows up "
                f"as a bare 'Bus error' mid-load with no OOM message. "
                f"Fix: sudo mount -o remount,size=450G /dev/shm")
    if env.get("KT_PREFETCH_GATHER") == "1" and env.get("KT_STORE_SHM") != "1":
        problems.append(
            "KT_PREFETCH_GATHER needs KT_STORE_SHM=1 -- the gather maps the store "
            "out of shared memory. Without it the prefetch silently stays off.")
    if env.get("KT_PREFETCH_SELECTIVE") == "1" and env.get("KT_GPU_ONLY") == "1":
        problems.append(
            "KT_PREFETCH_SELECTIVE=1 under KT_GPU_ONLY: the all-or-nothing rule "
            "assumes a surviving CPU call, and there is none. Set it to 0.")
    try:
        if int(env.get("KT_ADAPTIVE_MAX_SWAP", "8")) > 16:
            problems.append(
                "KT_ADAPTIVE_MAX_SWAP > 16 makes the stable-slot swap decline and "
                "fall back to the full restage: 144 ms per swapping layer instead "
                "of ~9 ms.")
    except ValueError:
        pass
    try:
        if int(env.get("KT_PREFETCH_SLOTS", "1")) > 4 and int(env.get("GPU_EXPERTS", "100")) >= 100:
            problems.append(
                "Many landing slots at GPU_EXPERTS>=100 is a boot-time OOM: slots "
                "come out of the same budget as residents (~9.56 MiB each) and the "
                "load phase already finishes with ~100 MiB spare.")
    except ValueError:
        pass
    return problems


def server_up():
    try:
        with urllib.request.urlopen(MODEL_BASE + "/health", timeout=3):
            return True
    except Exception:
        return False


def run_job(name, argv, env=None, shell_cmd=None):
    jid = f"{name}-{int(time.time())}"
    with JOB_LOCK:
        JOBS[jid] = {"name": name, "state": "running", "out": "", "started": time.time()}

    def _worker():
        try:
            if shell_cmd:
                p = subprocess.Popen(shell_cmd, shell=True, cwd=REPO,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=env or os.environ.copy())
            else:
                p = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     env=env or os.environ.copy())
            buf = []
            for line in p.stdout:
                buf.append(line)
                with JOB_LOCK:
                    JOBS[jid]["out"] = "".join(buf[-400:])
            p.wait()
            with JOB_LOCK:
                JOBS[jid]["state"] = "done" if p.returncode == 0 else f"exit {p.returncode}"
        except Exception as exc:                      # noqa: BLE001
            with JOB_LOCK:
                JOBS[jid]["state"] = "error"
                JOBS[jid]["out"] += f"\n{exc}"

    threading.Thread(target=_worker, daemon=True).start()
    return jid


def boot(env_over):
    env = os.environ.copy()
    env.update({k: v for k, v in _defaults().items() if k != "TIER"})
    env.update({k: v for k, v in env_over.items() if k != "TIER" and v != ""})
    env["KT_TOPK_MODE_FILE"] = MODE_FILE
    env.setdefault("TRITON_CACHE_DIR",
                   "/cache/nvme0/triton-cache" if os.path.isdir("/cache/nvme0")
                   else os.path.join(SCRATCH, "triton-cache"))
    os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
    with open(MODE_FILE, "w") as fh:
        fh.write("safe" + str(env_over.get("TIER", "8")) + "\n")
    subprocess.run(["bash", "bench/_kill_servers.sh"], cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = f"exec ./run_fast.sh > {shlex.quote(BOOT_LOG)} 2>&1"
    subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     preexec_fn=os.setsid)
    return env


COUNTER_RE = re.compile(r"\[kt-prefetch\] step (\d+): ([\d.]+) fetched/call, ([\d.]+) wanted/call")
REUSE_RE = re.compile(r"covered ([\d.]+)/call = ([\d.]+) fetched \+ ([\d.]+) reused \(([\d.]+)%")
ADAPT_RE = re.compile(r"\[kt-adaptive\] layer=(\d+) swaps=(\d+) top2_cov\(after\)=([\d.]+).*took=(\d+)ms")


def active_log():
    """The log to read counters from.

    Prefer the one this UI booted. Falling back to the newest log in the
    scratch dir matters because a server started by a bench script -- or by
    hand -- is the common case, and a panel that shows nothing for it looks
    like a dead instrument rather than a different log file.
    """
    if os.path.exists(BOOT_LOG):
        newest, mtime = BOOT_LOG, os.path.getmtime(BOOT_LOG)
    else:
        newest, mtime = None, 0.0
    try:
        for fn in os.listdir(SCRATCH):
            if not fn.endswith(".log"):
                continue
            p = os.path.join(SCRATCH, fn)
            t = os.path.getmtime(p)
            if t > mtime:
                newest, mtime = p, t
    except Exception:
        pass
    return newest


def log_tail(n=60):
    try:
        with open(active_log(), "rb") as fh:
            data = fh.read()[-200000:]
        return data.decode("utf-8", "replace").splitlines()[-n:]
    except Exception:
        return []


def counters():
    out = {"fetched": None, "wanted": None, "reuse_pct": None,
           "swaps": 0, "cov": None, "swap_ms": None, "stride": None}
    try:
        with open(active_log(), "rb") as fh:
            text = fh.read()[-4000000:].decode("utf-8", "replace")
    except Exception:
        return out
    m = list(COUNTER_RE.finditer(text))
    if m:
        out["fetched"], out["wanted"] = float(m[-1].group(2)), float(m[-1].group(3))
    m = list(REUSE_RE.finditer(text))
    if m:
        out["reuse_pct"] = float(m[-1].group(4))
    a = list(ADAPT_RE.finditer(text))
    out["swaps"] = len(a)
    if a:
        out["cov"] = float(a[-1].group(3))
        out["swap_ms"] = int(a[-1].group(4))
    s = re.findall(r"share their successor's GEMM \(stride (\d+)\)", text)
    if s:
        out["stride"] = int(s[-1])
    return out


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM-5.2 expert-streaming lab</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e6edf3}
 header{padding:12px 18px;border-bottom:1px solid #30363d;display:flex;gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#0d1117;z-index:5}
 h1{font-size:15px;margin:0;font-weight:600}
 .wrap{display:grid;grid-template-columns:minmax(420px,1.15fr) minmax(380px,1fr);gap:18px;padding:18px;align-items:start}
 @media(max-width:1000px){.wrap{grid-template-columns:1fr}}
 fieldset{border:1px solid #30363d;border-radius:8px;margin:0 0 14px;padding:10px 12px}
 legend{padding:0 6px;color:#7ee787;font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
 .knob{display:grid;grid-template-columns:1fr 130px;gap:8px;padding:7px 0;border-bottom:1px solid #21262d}
 .knob:last-child{border-bottom:0}
 .kname{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#79c0ff}
 .khelp{grid-column:1/-1;color:#8b949e;font-size:12px;margin-top:2px;display:none}
 .knob.open .khelp{display:block}
 .kname:hover{cursor:help;text-decoration:underline dotted}
 input,select,button,textarea{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 8px;font:inherit}
 input[type=text],input[type=number]{width:100%;font-family:ui-monospace,monospace;font-size:12.5px}
 .changed input{border-color:#d29922;background:#221a08}
 button{cursor:pointer}
 button.primary{background:#238636;border-color:#2ea043;font-weight:600}
 button.danger{background:#5a1e1e;border-color:#8b2f2f}
 pre{background:#010409;border:1px solid #30363d;border-radius:8px;padding:10px;overflow:auto;max-height:340px;font-size:12px;white-space:pre-wrap;word-break:break-word}
 .stat{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:#8b949e}
 .stat b{color:#e6edf3;font-family:ui-monospace,monospace}
 .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}
 .up{background:#3fb950}.down{background:#f85149}
 .warn{background:#221a08;border:1px solid #d29922;border-radius:8px;padding:9px 11px;margin-bottom:12px;font-size:12.5px;color:#e3b341}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
 label.inline{font-size:12.5px;color:#8b949e}
 .note{font-size:12px;color:#8b949e;margin:6px 0 0}
 .brief{border:1px solid #1f6feb;background:#0c1d38;border-radius:8px;padding:10px 12px;margin-bottom:12px;font-size:12.5px}
 .bq{color:#79c0ff;margin-bottom:6px}
 .be{color:#8b949e}
</style></head><body>
<header>
  <h1>GLM-5.2 expert-streaming lab</h1>
  <span id="health"><span class="dot down"></span>checking…</span>
  <select id="preset"></select>
  <button onclick="applyPreset()">load mode</button>
  <button class="primary" onclick="doBoot()">boot with these settings</button>
  <button class="danger" onclick="doKill()">kill server</button>
  <span class="note">boot ≈ 6–12 min (longer with the cache: it stages all 256 experts)</span>
</header>
<div class="wrap">
  <div>
    <div id="brief"></div>
    <div id="warns"></div>
    <div id="knobs"></div>
  </div>
  <div>
    <fieldset><legend>live (no reboot)</legend>
      <div class="row">
        <label class="inline">expert tier</label>
        <select id="livetier">
          <option>8</option><option>4</option><option>2</option><option>0</option>
        </select>
        <button onclick="setTier()">apply</button>
        <span class="note">top0 must produce garbage — it is the calibration case</span>
      </div>
    </fieldset>
    <fieldset><legend>ask the model</legend>
      <div class="row">
        <label class="inline">tier (probe — leave at 8)</label>
        <select id="asktier"><option>8</option><option>4</option><option>2</option><option>0</option></select>
        <label class="inline">max tokens</label><input type="number" id="askmax" value="800" style="width:80px">
        <label class="inline">temp</label><input type="number" id="asktemp" value="0" step="0.1" style="width:60px">
        <button onclick="clearChat()">clear</button>
      </div>
      <textarea id="askbox" rows="3" style="width:100%" placeholder="Ask anything. Shift+Enter for a newline, Enter to send."></textarea>
      <div class="row"><button class="primary" onclick="ask()">ask</button>
        <span class="note">the tier here rides on the request (<code>GLM5.2-topN</code>), so it does not touch the boot setting or the live mode file</span></div>
      <pre id="chat">—</pre>
      <p class="note"><b>top8 is the configuration</b> — every GPU-resident or landed expert computed,
      everything else substituted, nothing on the CPU. The lower tiers do not move work to the CPU;
      they discard genuine experts that are already on the GPU, at identical speed, so they are
      starvation probes only. Asking the same question at top8 then top2 measures how much margin
      the current setup has; top0 must visibly collapse, and if it doesn't, distrust the boot.</p>
    </fieldset>
    <fieldset><legend>measure</legend>
      <div class="row">
        <label class="inline">tokens</label><input type="number" id="btok" value="200" style="width:80px">
        <label class="inline">runs</label><input type="number" id="bruns" value="3" style="width:60px">
        <label class="inline">tier</label><input type="text" id="btier" value="8" style="width:50px">
        <button onclick="doBench()">bench ms/step</button>
      </div>
      <div class="row">
        <label class="inline">label</label><input type="text" id="rowlabel" value="row1" style="width:80px">
        <button class="primary" onclick="doFullRow()">speed + coherence (full row)</button>
        <span class="note">warm-up, ms/step at top8 and top2, counters, then the calibrated 1200-token sweep</span>
      </div>
      <div class="row">
        <label class="inline">coherence tiers</label><input type="text" id="ctiers" value="8 2 0" style="width:90px">
        <label class="inline">tokens</label><input type="number" id="ctok" value="1200" style="width:80px">
        <button onclick="doCoh()">score coherence</button>
      </div>
      <p class="note">A coherence run counts only if <b>top8 is clean AND top0 is degenerate on the same boot</b>.
      Scoring is character-level; accept length is not a quality signal.</p>
    </fieldset>
    <fieldset><legend>counters</legend>
      <div class="stat" id="counters">—</div>
      <p class="note">reuse falling while fetched rises is the cache and the prefetch fighting:
      reuse needs a stable resident set.</p>
    </fieldset>
    <fieldset><legend>output</legend><pre id="out">—</pre></fieldset>
    <fieldset><legend>server log</legend><pre id="log">—</pre></fieldset>
  </div>
</div>
<script>
let KNOBS=[], PRESETS={}, DEFAULTS={}, job=null;
async function j(u,o){const r=await fetch(u,o);return r.json()}
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}

async function init(){
  const s=await j('/api/schema'); KNOBS=s.knobs; PRESETS=s.presets; DEFAULTS=s.defaults;
  const groups={};
  KNOBS.forEach(k=>{(groups[k.group]=groups[k.group]||[]).push(k)});
  const host=document.getElementById('knobs');
  for(const g in groups){
    const fs=el('fieldset'); fs.appendChild(el('legend',null,g));
    groups[g].forEach(k=>{
      const d=el('div','knob'); d.id='k_'+k.name;
      const n=el('div','kname',k.name); n.onclick=()=>d.classList.toggle('open');
      const inp=document.createElement(k.kind==='bool'||k.kind==='tier'?'select':'input');
      if(k.kind==='bool'){['1','0'].forEach(v=>{const o=el('option',null,v);inp.appendChild(o)})}
      else if(k.kind==='tier'){['8','4','2','0'].forEach(v=>{const o=el('option',null,v);inp.appendChild(o)})}
      else {inp.type=(k.kind==='int'||k.kind==='float')?'number':'text'; if(k.kind==='float')inp.step='0.01'}
      inp.value=k.default; inp.id='v_'+k.name;
      inp.onchange=()=>{d.classList.toggle('changed',inp.value!==DEFAULTS[k.name]);check()};
      d.appendChild(n); d.appendChild(inp); d.appendChild(el('div','khelp',k.help));
      fs.appendChild(d);
    });
    host.appendChild(fs);
  }
  const ps=document.getElementById('preset');
  Object.keys(PRESETS).forEach(p=>ps.appendChild(el('option',null,p)));
  applyPreset(); poll(); setInterval(poll,4000);
}
function values(){const o={};KNOBS.forEach(k=>{o[k.name]=document.getElementById('v_'+k.name).value});return o}
function applyPreset(){
  const name=document.getElementById('preset').value;
  const meta=PRESETS[name]||{env:{}}, p=meta.env||{};
  KNOBS.forEach(k=>{const i=document.getElementById('v_'+k.name);
    i.value=(k.name in p)?p[k.name]:DEFAULTS[k.name];
    document.getElementById('k_'+k.name).classList.toggle('changed',i.value!==DEFAULTS[k.name]);});
  const b=document.getElementById('brief');
  b.innerHTML='';
  if(meta.asks){
    const d=el('div','brief');
    d.appendChild(el('div','bq','asks: '+meta.asks));
    d.appendChild(el('div','be','expect: '+meta.expect));
    b.appendChild(d);
  }
  check();
}
async function check(){
  const r=await j('/api/preflight',{method:'POST',body:JSON.stringify(values())});
  const w=document.getElementById('warns'); w.innerHTML='';
  r.problems.forEach(p=>{const d=el('div','warn',p);w.appendChild(d)});
}
async function doBoot(){
  const r=await j('/api/preflight',{method:'POST',body:JSON.stringify(values())});
  if(r.problems.length && !confirm('Preflight found:\n\n- '+r.problems.join('\n\n- ')+'\n\nBoot anyway?'))return;
  await j('/api/boot',{method:'POST',body:JSON.stringify(values())});
  document.getElementById('out').textContent='booting… watch the server log below';
}
async function doKill(){await j('/api/kill',{method:'POST'});}
async function setTier(){
  const t=document.getElementById('livetier').value;
  const r=await j('/api/tier',{method:'POST',body:JSON.stringify({tier:t})});
  document.getElementById('out').textContent='tier -> '+r.mode+'  (per-request, no reboot)';
}
async function doBench(){
  const b={tokens:+document.getElementById('btok').value,runs:+document.getElementById('bruns').value,
           tier:document.getElementById('btier').value};
  const r=await j('/api/bench',{method:'POST',body:JSON.stringify(b)}); job=r.job;
  document.getElementById('out').textContent='running…';
}
async function doFullRow(){
  const b={label:document.getElementById('rowlabel').value||'row',
           tokens:+document.getElementById('ctok').value,warm:1400};
  const r=await j('/api/fullrow',{method:'POST',body:JSON.stringify(b)}); job=r.job;
  document.getElementById('out').textContent=
    'full row running — warm-up 1400 tok, then timing, then 3 coherence samples. ~10-15 min.';
}
async function doCoh(){
  const b={tiers:document.getElementById('ctiers').value,tokens:+document.getElementById('ctok').value};
  const r=await j('/api/coherence',{method:'POST',body:JSON.stringify(b)}); job=r.job;
  document.getElementById('out').textContent='scoring… (a 1200-token row per tier takes a few minutes)';
}
let history=[];
function clearChat(){history=[];document.getElementById('chat').textContent='—'}
document.addEventListener('keydown',e=>{
  if(e.target.id==='askbox'&&e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}
});
async function ask(){
  const box=document.getElementById('askbox'), q=box.value.trim(); if(!q)return;
  const tier=document.getElementById('asktier').value;
  const pane=document.getElementById('chat');
  history.push({role:'user',content:q}); box.value='';
  if(pane.textContent==='—')pane.textContent='';
  pane.textContent+='\n\n>>> ['+'top'+tier+'] '+q+'\n';
  const body={model:'GLM5.2-top'+tier,messages:history,stream:true,
    max_tokens:+document.getElementById('askmax').value,
    temperature:+document.getElementById('asktemp').value};
  let r;
  try{ r=await fetch('/v1/chat/completions',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); }
  catch(err){ pane.textContent+='[request failed: '+err+']\n'; return; }
  if(!r.ok){ pane.textContent+='[server returned '+r.status+' — is it booted?]\n'; return; }
  const rd=r.body.getReader(), dec=new TextDecoder(); let buf='',answer='',think=false;
  const t0=performance.now(); let toks=0;
  for(;;){
    const {value,done}=await rd.read(); if(done)break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split('\n'); buf=lines.pop();
    for(const ln of lines){
      if(!ln.startsWith('data:'))continue;
      const d=ln.slice(5).trim(); if(d==='[DONE]')continue;
      let o; try{o=JSON.parse(d)}catch(e){continue}
      const del=(o.choices||[{}])[0].delta||{};
      if(del.reasoning_content){ if(!think){pane.textContent+='(thinking) ';think=true}
        pane.textContent+=del.reasoning_content; toks++; }
      if(del.content){ if(think){pane.textContent+='\n\n';think=false}
        pane.textContent+=del.content; answer+=del.content; toks++; }
      pane.scrollTop=pane.scrollHeight;
    }
  }
  const ms=(performance.now()-t0)/Math.max(toks,1);
  pane.textContent+='\n['+toks+' chunks, '+ms.toFixed(1)+' ms/chunk]\n';
  history.push({role:'assistant',content:answer});
}
async function poll(){
  const s=await j('/api/status'+(job?('?job='+encodeURIComponent(job)):''));
  document.getElementById('health').innerHTML=
    '<span class="dot '+(s.up?'up':'down')+'"></span>'+(s.up?'server up':'server down');
  document.getElementById('log').textContent=s.log.join('\n')||'—';
  const c=s.counters; const f=[];
  if(c.fetched!=null)f.push('fetched/call <b>'+c.fetched+'</b>');
  if(c.wanted!=null)f.push('wanted/call <b>'+c.wanted+'</b>');
  if(c.reuse_pct!=null)f.push('reuse <b>'+c.reuse_pct+'%</b>');
  if(c.stride!=null)f.push('pred stride <b>'+c.stride+'</b>');
  if(c.swaps)f.push('cache swaps <b>'+c.swaps+'</b>');
  if(c.cov!=null)f.push('resident cov <b>'+c.cov+'</b>');
  if(c.swap_ms!=null)f.push('last swap <b>'+c.swap_ms+' ms</b>');
  document.getElementById('counters').innerHTML=f.join(' ')||'—';
  if(s.job)document.getElementById('out').textContent=s.job.out+'\n['+s.job.state+']';
}
init();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console for our own messages
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/schema":
            return self._send(200, json.dumps({
                "knobs": [{"group": g, "name": n, "default": d, "kind": k, "help": h}
                          for g, n, d, k, h in KNOBS],
                "presets": PRESETS, "defaults": _defaults()}))
        if path == "/api/status":
            jid = None
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for part in q.split("&"):
                    if part.startswith("job="):
                        jid = urllib.parse.unquote(part[4:])
            with JOB_LOCK:
                jobinfo = dict(JOBS.get(jid)) if jid and jid in JOBS else None
            return self._send(200, json.dumps({
                "up": server_up(), "log": log_tail(), "counters": counters(),
                "job": jobinfo}))
        return self._send(404, json.dumps({"error": "not found"}))

    def _proxy_stream(self):
        """Pass a chat completion through to the model, chunk by chunk.

        Same-origin so the browser needs no CORS, and streamed rather than
        buffered so the answer appears as it is generated -- which is also the
        only way the token cadence is visible, and cadence is what every
        measurement in this project is actually about.
        """
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        req = urllib.request.Request(
            MODEL_BASE + "/v1/chat/completions", data=raw,
            headers={"Content-Type": "application/json"})
        try:
            up = urllib.request.urlopen(req, timeout=900)
        except urllib.error.HTTPError as exc:
            return self._send(exc.code, exc.read() or b'{"error":"upstream"}')
        except Exception as exc:                       # noqa: BLE001
            return self._send(
                503, json.dumps({"error": f"model server unreachable: {exc}"}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for chunk in up:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                        # browser navigated away

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            return self._proxy_stream()
        body = self._body()
        if path == "/api/preflight":
            return self._send(200, json.dumps({"problems": preflight(body)}))
        if path == "/api/boot":
            boot(body)
            return self._send(200, json.dumps({"ok": True, "log": BOOT_LOG}))
        if path == "/api/kill":
            subprocess.run(["bash", "bench/_kill_servers.sh"], cwd=REPO,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return self._send(200, json.dumps({"ok": True}))
        if path == "/api/tier":
            mode = "safe" + str(body.get("tier", "8"))
            with open(MODE_FILE, "w") as fh:
                fh.write(mode + "\n")
            return self._send(200, json.dumps({"ok": True, "mode": mode}))
        if path == "/api/bench":
            argv = [os.path.join(REPO, ".venv/bin/python"), "bench/prefetch_rate.py",
                    "--model", "GLM5.2", "--tier", str(body.get("tier", "8")),
                    "--runs", str(body.get("runs", 3)), "--tokens", str(body.get("tokens", 200)),
                    "--label", "lab", "--out", "bench/profile_out/lab_rate.json"]
            return self._send(200, json.dumps({"job": run_job("bench", argv)}))
        if path == "/api/fullrow":
            # Speed at two tiers, then the calibrated coherence sweep, as ONE
            # job -- because a speed number without a coherence verdict is not a
            # result here (top0 is the fastest configuration there is, and it
            # produces garbage).
            label = str(body.get("label", "row")).replace("'", "")
            env = os.environ.copy()
            env["TOKENS"] = str(body.get("tokens", 1200))
            env["KT_TOPK_MODE_FILE"] = MODE_FILE
            py = os.path.join(REPO, ".venv/bin/python")
            cmd = (
                f"set -e; "
                f"echo '--- warm-up, so the cache has converged before timing ---'; "
                f"{py} bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 "
                f"  --tokens {int(body.get('warm', 1400))} --label warm-{label} "
                f"  --out bench/profile_out/lab_row.json | grep ms/step || true; "
                f"for t in 8 2; do "
                f"  {py} bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 "
                f"    --tokens 200 --label {label}-top$t "
                f"    --out bench/profile_out/lab_row.json | grep ms/step || true; "
                f"done; "
                f"echo '--- counters ---'; "
                f"grep -a 'kt-prefetch] step' {shlex.quote(str(active_log()))} | tail -2 || true; "
                f"echo '--- coherence (top0 is the calibration case) ---'; "
                f"bash bench/coherence_run.sh {label} 8 2 0")
            return self._send(200, json.dumps({
                "job": run_job("fullrow", None, env=env, shell_cmd=cmd)}))
        if path == "/api/coherence":
            tiers = str(body.get("tiers", "8 2 0"))
            env = os.environ.copy()
            env["TOKENS"] = str(body.get("tokens", 1200))
            env["KT_TOPK_MODE_FILE"] = MODE_FILE
            cmd = f"bash bench/coherence_run.sh lab {tiers}"
            return self._send(200, json.dumps({
                "job": run_job("coherence", None, env=env, shell_cmd=cmd)}))
        return self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print(f"lab UI on http://0.0.0.0:{UI_PORT}   repo={REPO}")
    print(f"boot log -> {BOOT_LOG}")
    print(f"tier mode file -> {MODE_FILE}")
    print(f"/dev/shm = {shm_free_gib():.0f} GiB "
          f"(the adaptive cache needs ~372 GiB of it)")
    ThreadingHTTPServer(("0.0.0.0", UI_PORT), Handler).serve_forever()
