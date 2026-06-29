import gzip, json, sys, re
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path) as f:
    data = json.load(f)
ev = data["traceEvents"]

# categorize by cat
by_cat = defaultdict(float)
by_cat_cnt = defaultdict(int)
for e in ev:
    if e.get("ph") != "X":
        continue
    c = e.get("cat", "?")
    d = e.get("dur", 0)
    by_cat[c] += d
    by_cat_cnt[c] += 1

print("=== time (ms) by cat (ph=X complete events) ===")
for c, v in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"{c:20s} {v/1000:12.2f} ms   n={by_cat_cnt[c]}")

# count ProfilerStep markers = number of forward steps captured
steps = [e for e in ev if e.get("ph") == "X" and "ProfilerStep" in e.get("name", "")]
nstep = len(steps)
print(f"\nProfilerStep events: {nstep}")
if steps:
    span = max(e["ts"]+e["dur"] for e in steps) - min(e["ts"] for e in steps)
    print(f"wall span of steps: {span/1000:.2f} ms  -> {span/1000/max(nstep,1):.2f} ms/step")

# bucket GPU kernels by name pattern
PATS = [
    ("attention",  re.compile(r"flash|mla|attn|fmha|rotary|rope", re.I)),
    ("moe_gemm",   re.compile(r"cutlass|w4a8|gemm|moe|grouped|scaled_mm|fp8|int4|marlin", re.I)),
    ("allreduce",  re.compile(r"nccl|all.?reduce|reduce.?scatter|all.?gather|ncclDevKernel", re.I)),
    ("norm_act",   re.compile(r"rmsnorm|layernorm|norm|silu|gelu|act|add|elementwise|mul|residual", re.I)),
    ("copy",       re.compile(r"memcpy|copy|cast|convert|index|gather|scatter", re.I)),
    ("sample",     re.compile(r"sampl|argmax|topk|softmax|logit|embed|vocab", re.I)),
]
def bucket(name):
    for label, pat in PATS:
        if pat.search(name):
            return label
    return "other_gpu"

# pick GPU kernel events: cat == 'kernel'
kern = defaultdict(float)
kern_cnt = defaultdict(int)
kern_total = 0.0
gpu_span_min = None; gpu_span_max = None
for e in ev:
    if e.get("ph") != "X" or e.get("cat") != "kernel":
        continue
    d = e.get("dur", 0)
    b = bucket(e.get("name", ""))
    kern[b] += d; kern_cnt[b] += 1; kern_total += d
    ts = e["ts"]
    gpu_span_min = ts if gpu_span_min is None else min(gpu_span_min, ts)
    gpu_span_max = max(gpu_span_max or 0, ts + d)

print("\n=== GPU kernel time (cat='kernel') bucketed ===")
for b, v in sorted(kern.items(), key=lambda x: -x[1]):
    pct = 100*v/kern_total if kern_total else 0
    print(f"{b:12s} {v/1000:10.2f} ms  {pct:5.1f}%  n={kern_cnt[b]}"
          + (f"  {v/nstep/1000:.3f} ms/step" if nstep else ""))
print(f"{'TOTAL GPU':12s} {kern_total/1000:10.2f} ms"
      + (f"   {kern_total/nstep/1000:.3f} ms/step busy" if nstep else ""))
if gpu_span_min:
    span = (gpu_span_max - gpu_span_min)/1000
    print(f"GPU wall span: {span:.2f} ms; GPU busy fraction: {100*kern_total/1000/span:.1f}%")
    if nstep:
        print(f"  -> wall {span/nstep:.3f} ms/step, busy {kern_total/nstep/1000:.3f} ms/step, "
              f"idle/gap {(span-kern_total/1000)/nstep:.3f} ms/step")

# top 15 kernels by name
top = defaultdict(float); topc=defaultdict(int)
for e in ev:
    if e.get("ph")=="X" and e.get("cat")=="kernel":
        top[e["name"]] += e.get("dur",0); topc[e["name"]]+=1
print("\n=== top 15 GPU kernels by total time ===")
for n,v in sorted(top.items(), key=lambda x:-x[1])[:15]:
    print(f"{v/1000:9.2f} ms  n={topc[n]:5d}  {n[:90]}")
