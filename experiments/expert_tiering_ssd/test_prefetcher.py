#!/usr/bin/env python3
"""Check the background expert prefetcher does what the design needs.

The point of it is that the disk half of a promotion (~6 MiB x3) overlaps token
generation, leaving only the buffer write on the forward thread. So:

  1. request() must NOT block the caller;
  2. the read must actually happen on another thread;
  3. take() must hand back exactly what was read, once;
  4. the ready set must stay bounded -- holding many un-staged experts would
     defeat the RAM budget the tier exists to enforce (~19 MiB each);
  5. a failing read must not kill the thread or wedge the queue.

Run: .venv/bin/python experiments/expert_tiering_ssd/test_prefetcher.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
os.environ.setdefault("KT_RAM_EXPERTS", "32")
from sglang.srt.layers.moe.kt_ep_wrapper import _KtExpertPrefetcher  # noqa: E402

READ_SECONDS = 0.05
main_thread = threading.get_ident()
threads_used = set()


class FakeWrapper:
    def __init__(self, layer):
        self.layer = layer

    def read_expert(self, expert_id):
        threads_used.add(threading.get_ident())
        if expert_id == 666:
            raise RuntimeError("simulated bad read")
        time.sleep(READ_SECONDS)
        return {"layer": self.layer, "expert": expert_id}


pf = _KtExpertPrefetcher(lambda li: FakeWrapper(li), max_ready=4)

# 1. request() returns immediately even though each read takes READ_SECONDS.
t0 = time.perf_counter()
for e in range(4):
    pf.request(3, e)
elapsed = time.perf_counter() - t0
assert elapsed < READ_SECONDS, f"request() blocked for {elapsed*1000:.0f} ms"
print(f"ok  1. request() is non-blocking ({elapsed*1000:.2f} ms for 4 requests, "
      f"each read takes {READ_SECONDS*1000:.0f} ms)")

deadline = time.time() + 10
while time.time() < deadline and pf.stats()[0] < 4:
    time.sleep(0.02)
ready, inflight = pf.stats()
assert ready == 4, f"only {ready} of 4 became ready"
print(f"ok  2. reads completed off-thread (ready={ready}, inflight={inflight})")
assert main_thread not in threads_used, "a read ran on the calling thread"
print(f"ok  2b. no read ran on the caller's thread ({len(threads_used)} worker thread(s))")

# 3. take() returns the payload once, then None.
w = pf.take(3, 1)
assert w == {"layer": 3, "expert": 1}, f"wrong payload {w}"
assert pf.take(3, 1) is None, "take() returned the same weights twice"
print("ok  3. take() hands back the right weights exactly once")

# 4. bounded: with max_ready=4 and one consumed, at most 4 are ever held.
for e in range(20, 40):
    pf.request(3, e)
time.sleep(READ_SECONDS * 6)
ready, inflight = pf.stats()
assert ready + inflight <= 4, f"prefetcher held {ready}+{inflight} > max_ready=4"
print(f"ok  4. ready set stays bounded (ready={ready}, inflight={inflight}, cap=4)")

# 4b. prune() must free the bounded set for new candidates. Without it a hot
# new expert can never be prefetched once the set fills with stale ones.
before = pf.stats()[0]
dropped = pf.prune([(3, 21)])          # only one of the held keys stays a candidate
after = pf.stats()[0]
assert dropped >= 1 and after < before, f"prune dropped {dropped}, {before}->{after}"
pf.request(7, 1)
deadline = time.time() + 10
got = None
while time.time() < deadline and got is None:
    got = pf.take(7, 1)
    time.sleep(0.02)
assert got is not None, "a new candidate still could not be prefetched after prune()"
print(f"ok  4b. prune() frees the set ({before} -> {after}) so new candidates can fetch")

# 5. a failing read must not wedge the thread.
pf.prune([])
pf.request(3, 666)
time.sleep(READ_SECONDS * 4)
pf.request(9, 7)
time.sleep(READ_SECONDS * 2)
# confirm the worker still serves requests after the failure
pf.request(9, 8)
deadline = time.time() + 10
got = None
while time.time() < deadline and got is None:
    got = pf.take(9, 8)
    time.sleep(0.02)
assert got is not None, "prefetch thread died after a failing read"
print("ok  5. a failing read is contained; the thread keeps serving")

print("\nprefetcher behaves as the design requires")
