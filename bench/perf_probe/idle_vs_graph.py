"""Where is the GPU idle relative to the single fused CUDA graph per step?
- Build GPU busy/idle from merged kernel+memcpy+memset intervals.
- Mark the cudaGraphLaunch and cudaEventSynchronize host events.
- For each idle gap, classify: INSIDE a graph-launch span (intra-graph stall, =kt host-node/
  cross-stream serialization captured in the graph) vs BETWEEN launches (inter-step host gap).
"""
import gzip, json, sys
from collections import defaultdict
import bisect

path = sys.argv[1]
d = json.load(gzip.open(path))
ev = d["traceEvents"]

GPU = {"kernel","gpu_memcpy","gpu_memset"}
iv = sorted((e["ts"], e["ts"]+e.get("dur",0)) for e in ev
            if e.get("ph")=="X" and e.get("cat") in GPU)
merged=[]
for s,t in iv:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],t)
    else: merged.append([s,t])
span=merged[-1][1]-merged[0][0]; busy=sum(t-s for s,t in merged)
idle=span-busy
gaps=[(merged[i-1][1],merged[i][0],merged[i][0]-merged[i-1][1])
      for i in range(1,len(merged)) if merged[i][0]>merged[i-1][1]]

# host events of interest
def get(cat,name=None):
    return sorted([(e["ts"],e["ts"]+e.get("dur",0)) for e in ev
                   if e.get("ph")=="X" and e.get("cat")==cat and (name is None or e["name"]==name)])
glaunch = get("cuda_runtime","cudaGraphLaunch")
esync   = get("cuda_runtime","cudaEventSynchronize")
NS=len(glaunch)
print(f"span {span/1000:.1f}ms busy {busy/1000:.1f}ms ({100*busy/span:.0f}%) idle {idle/1000:.1f}ms ({100*idle/span:.0f}%)")
print(f"graph launches {NS} -> per-step wall {span/1000/NS:.2f}ms busy {busy/1000/NS:.2f} idle {idle/1000/NS:.2f}")

def covers(spans, t0, t1):
    # total overlap of [t0,t1] with the list of host spans
    ov=0.0
    for s,t in spans:
        if t<t0: continue
        if s>t1: break
        ov+=min(t,t1)-max(s,t0)
    return ov

# classify idle gaps >=20us
big=[g for g in gaps if g[2]>=20]
bt=sum(g[2] for g in big)
in_glaunch=0.0; in_esync=0.0; uncovered=0.0
for g0,g1,g in big:
    cg=covers(glaunch,g0,g1)
    ce=covers(esync,g0,g1)
    # a gap can overlap both; attribute to the max single owner, remainder uncovered
    owner=max(cg,ce)
    in_glaunch+= cg if cg>=ce else 0
    in_esync  += ce if ce>cg else 0
    uncovered += max(0, g-max(cg,ce))
print(f"\nbig idle gaps (>=20us): {len(big)} ({len(big)/NS:.0f}/step), {bt/1000:.1f}ms = {100*bt/idle:.0f}% of idle")
print(f"  idle DURING cudaGraphLaunch span : {in_glaunch/1000:.1f}ms ({100*in_glaunch/bt:.0f}% of big idle)  [intra-graph stall]")
print(f"  idle DURING cudaEventSynchronize : {in_esync/1000:.1f}ms ({100*in_esync/bt:.0f}%)               [host blocked on event]")
print(f"  idle covered by NEITHER          : {uncovered/1000:.1f}ms ({100*uncovered/bt:.0f}%)               [pure inter-step host/python]")

# histogram of big gap sizes
from collections import Counter
buck=[(20,100),(100,500),(500,2000),(2000,1e9)]
h=defaultdict(lambda:[0,0.0])
for *_,g in big:
    for b in buck:
        if b[0]<=g<b[1]: h[b][0]+=1;h[b][1]+=g;break
print("\n  gap-size histogram:")
for b in buck:
    c,tt=h[b]; print(f"    {b[0]:>5}-{int(b[1]) if b[1]<1e9 else 'inf':>6}us  n={c:4d} ({c/NS:5.1f}/step)  {tt/1000:7.1f}ms  {100*tt/idle:4.0f}% idle")
