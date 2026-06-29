import json, time, urllib.request, sys
URL="http://localhost:8000/generate"
PROMPT=("[gMASK]<sop><|user|>\nWrite a long detailed technical essay about how CPUs "
        "and GPUs differ in architecture and workloads.<|assistant|>\n")
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
runs=int(sys.argv[2]) if len(sys.argv)>2 else 5
def one():
    body=json.dumps({"text":PROMPT,"sampling_params":{"temperature":0,"max_new_tokens":N}}).encode()
    r=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t=time.time(); o=json.load(urllib.request.urlopen(r,timeout=600)); dt=time.time()-t
    m=o["meta_info"]; ct=m["completion_tokens"]
    # decode tok/s excludes prefill: use (ct-1)/(e2e - ttft) if available else ct/dt
    return ct, dt, m.get("e2e_latency",dt)
tps=[]
for i in range(runs):
    ct,dt,e2e=one(); r=ct/dt; tps.append(r)
    print(f"run{i}: {ct} tok in {dt:.2f}s -> {r:.2f} tok/s")
tps.sort()
print(f"median {tps[len(tps)//2]:.2f} tok/s  (min {tps[0]:.2f} max {tps[-1]:.2f})")
