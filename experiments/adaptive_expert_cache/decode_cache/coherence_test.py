#!/usr/bin/env python3
"""Fair coherence test: normal reasoning (no /nothink), verifiable facts."""
import json, sys, urllib.request
BASE="http://127.0.0.1:8000"; TIER=sys.argv[1] if len(sys.argv)>1 else "top2"
TESTS=[
    ("What is 17 times 23? Show your reasoning then give the final number.", "391"),
    ("What is the capital of France? One word.", "Paris"),
    ("List the first 5 prime numbers.", "2"),
    ("Spell 'accommodation' backwards.", "noit"),
]
def ask(p, mx=400):
    body=json.dumps({"model":f"GLM5.2-{TIER}","messages":[{"role":"user","content":p}],
                     "temperature":0.0,"max_tokens":mx,"stream":False}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    o=json.loads(urllib.request.urlopen(req,timeout=300).read()); m=o["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), o["choices"][0].get("finish_reason")
print(f"=== coherence @ {TIER} ===")
for p,exp in TESTS:
    c,r,fr=ask(p)
    ok = exp.lower() in (c+r).lower()
    print(f"[{'OK ' if ok else 'BAD'}] {p[:45]:45s} -> content={c[:50]!r} finish={fr}")
