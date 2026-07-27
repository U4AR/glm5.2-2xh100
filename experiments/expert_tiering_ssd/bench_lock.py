"""Exclusive lock for anything that benchmarks the server.

The server runs with CUDA_GRAPH_MAX_BS=1. A second concurrent request pushes
the batch to size 2, which falls outside the captured CUDA graph and drops both
requests into eager mode -- a 43 tok/s pass was recorded as 16.86 that way, and
the interloper read 9 tok/s instead of ~40. Neither number announced itself as
wrong.

So benchmarking is made mutually exclusive by construction, and a second
attempt FAILS LOUDLY rather than waiting or proceeding. A timing harness that
silently queues behind another one is still measuring the wrong thing.
"""
import fcntl
import os
import sys
from contextlib import contextmanager

LOCK_PATH = os.environ.get("KT_BENCH_LOCK", "/tmp/kt_bench.lock")


@contextmanager
def exclusive_bench(name: str = "bench"):
    fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = ""
        try:
            with open(LOCK_PATH) as r:
                holder = r.read().strip()
        except Exception:
            pass
        print(
            f"REFUSING TO RUN: another benchmark holds {LOCK_PATH}"
            f"{' (' + holder + ')' if holder else ''}.\n"
            "Concurrent requests break the CUDA-graph batch size and both "
            "measurements become meaningless. Wait for it to finish.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        fh.write(f"{name} pid={os.getpid()}")
        fh.flush()
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
