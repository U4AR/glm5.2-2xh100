"""For the 78 per-step intra-graph idle gaps: what runs on ANY thread during them?
Also: identify the 13 eager (non-captured) kernels per step (cudaLaunchKernel correlates)."""
import gzip, json, sys
from collections import defaultdict, Counter

d = json.load(gzip.open(sys.argv[1]))
ev = d["traceEvents"]
GPU = {"kernel","gpu_memcpy","gpu_memset"}

iv = sorted((e["ts"], e["ts"]+e.get("dur",0)) for e in ev if e.get("ph")=="X" and e.get("cat") in GPU)
merged=[]
for s,t in iv:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],t)
    else: merged.append([s,t])
gaps=[(merged[i-1][1],merged[i][0]) for i in range(1,len(merged)) if merged[i][0]-merged[i-1][1]>=100]
NS=sum(1 for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="cudaGraphLaunch")
print(f"{len(gaps)} gaps>=100us ({len(gaps)/NS:.0f}/step)")

# all non-GPU events, indexed by start; find overlap with gaps -> tally by (cat,name) and by tid
host=[e for e in ev if e.get("ph")=="X" and e.get("cat") not in GPU and e.get("cat")!="Trace"]
host.sort(key=lambda e:e["ts"])
hts=[e["ts"] for e in host]
import bisect
nm_t=defaultdict(float); nm_c=defaultdict(int); tid_t=defaultdict(float); cat_t=defaultdict(float)
for g0,g1 in gaps:
    lo=bisect.bisect_left(hts,g0-60000)
    for e in host[lo:]:
        if e["ts"]>g1: break
        s=e["ts"]; t=s+e.get("dur",0); ov=min(t,g1)-max(s,g0)
        if ov<=0: continue
        key=(e.get("cat"),e["name"][:46])
        nm_t[key]+=ov; nm_c[key]+=1; tid_t[e.get("tid")]+=ov; cat_t[e.get("cat")]+=ov
print("\n=== host activity overlapping intra-graph idle gaps, by category (ms) ===")
for c,v in sorted(cat_t.items(),key=lambda x:-x[1]): print(f"  {str(c):20s} {v/1000:8.1f}")
print("\n=== by thread tid (ms overlap) ===")
for tid,v in sorted(tid_t.items(),key=lambda x:-x[1])[:8]: print(f"  tid {tid:>8}  {v/1000:8.1f}ms")
print("\n=== top named ops in gaps (cat,name -> ms, count) ===")
for (c,n),v in sorted(nm_t.items(),key=lambda x:-x[1])[:22]:
    print(f"  {v/1000:7.1f}ms n={nm_c[(c,n)]:5d}  [{c}] {n}")

# thread names
tnames={}
for e in ev:
    if e.get("ph")=="M" and e.get("name")=="thread_name":
        tnames[(e.get("pid"),e.get("tid"))]=e.get("args",{}).get("name")
print("\n=== thread names for top gap-filling tids ===")
seen=set()
for tid,v in sorted(tid_t.items(),key=lambda x:-x[1])[:8]:
    for (pid,t),nm in tnames.items():
        if t==tid and tid not in seen:
            print(f"  tid {tid}: {nm}"); seen.add(tid)
