"""Capture a fixed number of steady-state DECODE steps with the torch profiler.

Starts a long generation in a background thread (so the model is in decode), then
asks /start_profile to record N steps and auto-stop. Trace lands in OUT_DIR.
"""
import json, time, urllib.request, threading, sys, os

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof"
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
os.makedirs(OUT_DIR, exist_ok=True)
URL = "http://localhost:8000"
PROMPT = ("[gMASK]<sop><|user|>\nWrite a long detailed technical essay about how CPUs "
          "and GPUs differ in architecture and workloads.<|assistant|>\n")

def gen():
    body = json.dumps({"text": PROMPT,
                       "sampling_params": {"temperature": 0, "max_new_tokens": 400}}).encode()
    req = urllib.request.Request(URL + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    print("gen done:", r["meta_info"]["completion_tokens"], "tok",
          "e2e", round(r["meta_info"]["e2e_latency"], 2), "s")

t = threading.Thread(target=gen)
t.start()
# let it finish prefill + reach steady decode
time.sleep(4)

body = json.dumps({"output_dir": OUT_DIR, "num_steps": NSTEPS,
                   "activities": ["CPU", "GPU"]}).encode()
req = urllib.request.Request(URL + "/start_profile", data=body,
                             headers={"Content-Type": "application/json"})
print("start_profile:", urllib.request.urlopen(req, timeout=60).read().decode()[:200])
t.join()
print("done; trace dir:", OUT_DIR)
print(os.listdir(OUT_DIR))
