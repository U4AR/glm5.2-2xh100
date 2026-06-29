"""Ground-truth per-step structure: use ProfilerStep markers as step boundaries.
Count host CUDA-API calls per step (cudaGraphLaunch, cudaEventSynchronize, ...),
and classify what blocks the host. No heuristics."""
import gzip, json, sys
from collections import defaultdict
import bisect

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
ev = data["traceEvents"]

# step boundaries from ProfilerStep markers (cpu_op / user_annotation)
steps = sorted([(e["ts"], e["ts"]+e.get("dur",0), e["name"])
                for e in ev if e.get("ph")=="X" and "ProfilerStep" in e.get("name","")],
               key=lambda x: x[0])
print(f"ProfilerStep markers: {len(steps)}")
if steps:
    durs = [(b-a)/1000 for a,b,_ in steps]
    print(f"  per-step wall: min {min(durs):.2f} max {max(durs):.2f} mean {sum(durs)/len(durs):.2f} ms")

# Use the middle steps (steady state), skip first/last 2
core = steps[2:-2] if len(steps) > 6 else steps
if not core:
    core = steps
t0, t1 = core[0][0], core[-1][1]
nstep = len(core)
print(f"core steady steps: {nstep}, window {(t1-t0)/1000:.1f} ms, {(t1-t0)/1000/nstep:.2f} ms/step")

# host API calls (cuda_runtime) in the core window
api_cnt = defaultdict(int); api_dur = defaultdict(float)
for e in ev:
    if e.get("ph")!="X" or e.get("cat")!="cuda_runtime": continue
    ts = e["ts"]
    if ts < t0 or ts > t1: continue
    api_cnt[e["name"]] += 1; api_dur[e["name"]] += e.get("dur",0)
print(f"\n=== host CUDA-API calls per step (cat=cuda_runtime, core window) ===")
print(f"{'api':40s} {'/step':>8} {'tot ms':>9} {'avg us':>8}")
for n in sorted(api_cnt, key=lambda x:-api_dur[x])[:18]:
    print(f"{n[:40]:40s} {api_cnt[n]/nstep:>8.1f} {api_dur[n]/1000:>9.1f} {api_dur[n]/api_cnt[n]:>8.1f}")

# GPU kernel launches per step = how many distinct kernels actually run (proxy for graph contents)
kern = [e for e in ev if e.get("ph")=="X" and e.get("cat")=="kernel" and t0<=e["ts"]<=t1]
print(f"\nGPU kernels in window: {len(kern)} -> {len(kern)/nstep:.0f}/step")

# Is the model ONE graph or many? cudaGraphLaunch/step is the answer.
gl = api_cnt.get("cudaGraphLaunch",0)/nstep
print(f"\n>>> cudaGraphLaunch/step = {gl:.2f}  (1.0 => single fused graph; >1 => segmented)")
