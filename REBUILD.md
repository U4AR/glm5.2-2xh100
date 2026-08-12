# Can this repo rebuild a working server? — audit, 2026-08-12

Short answer: **yes.** `setup.sh` bootstraps a functioning GLM-5.2 server from
public sources plus this repo. Below is what was verified, and the three gaps
found during the teardown audit and closed.

## The path

    git clone <this repo> && cd RunGLM
    ./setup.sh                              # builds kt-kernel + sglang, restores overlays
    python int4_scripts/download_w4afp8.py  # 373 GB of weights, ~3 min on this link
    bash bench/deepswe/start_server.sh      # or ./run_fast.sh for the shipped defaults

## Why it works — the non-obvious part

sglang is **not** installed from PyPI. It is built from
`ktransformers/third_party/sglang`, so a plain `pip install` would overwrite
every patched file. `setup.sh` handles this in the right order:

1. copies every tracked `.venv/**/*.py` overlay to a temp dir **before** the build
2. builds kt-kernel and installs sglang (which overwrites `.venv`)
3. `scripts/apply_runtime_overlays.sh` puts the overlays back

The overlays are the project. They carry the top-K expert substitution
(`deepseek_v2.py` — GlmMoeDsa routes through there, **not** `glm4_moe.py`), the
MTP-under-CUDA-graphs fix, the NSA multistep draft fix, the W4AFP8 `-1` topk
remap, and the per-request intelligence tier.

Pins, all verified reachable on 2026-08-12:

| input | pinned to | status |
|---|---|---|
| ktransformers | `U4AR/ktransformers` @ `512802b9`, branch `glm5.2-2xh100-stable` | public, exists |
| sglang source | that repo's `third_party/sglang` submodule | fetched by `--recursive` |
| python deps | `requirements-lock.txt` | 209 pins |
| weights | `int4_scripts/download_w4afp8.py` | re-downloadable |

## Three gaps found and closed

**1. Three overlays were modified but untracked.** `setup.sh` only restores what
`git ls-files '.venv/**'` returns, so an untracked overlay is silently lost on
rebuild — and two of them matter:

* `srt/managers/scheduler.py` (+42) bounds generation by the KV pool rather than
  by model context. Without it, an omitted OpenAI `max_tokens` lets a single
  request grow to "context minus prompt", which overruns a small
  `max_total_tokens`.
* `srt/managers/schedule_policy.py` (+17) handles a single request that cannot
  fit the pool — the `retract_decode` failure mode.
* `srt/models/glm4_moe.py` (+35) is inert (dead experimental reroute) but kept
  whole rather than half-restored.

All three are now in the `.gitignore` allowlist and tracked.

**2. The exact sglang base was only on this machine.** The venv patches apply to
`kvcache-ai/sglang` @ `51032b712`, which is not a branch head on the public repo.
Pushed to `U4AR/sglang-kt-glm52`, branch `glm52-base`, and its 5-line
uncommitted edit is saved at `backup/sglang-base/local-uncommitted.patch`. That
edit gates sglang's disconnect-abort behind `SGLANG_DISABLE_DISCONNECT_ABORT`,
defaulting to **disabled** — which is why an abandoned client request never frees
its server slot here, a behaviour that cost two debugging sessions.

**3. The WIP ktransformers fork could not be pushed at all.** The local fork sits
6 commits past `6c9c95601d97` (packed RAWINT4 CPU MoE kernel,
`write_weights_to_buffer`, three-tier expert store) but is a **shallow clone**, so
any push is rejected with `did not receive expected object`. Exported to
`backup/ktransformers-patches/` as `git format-patch` output; apply with `git am`
onto upstream `6c9c95601d97`.

Note these 6 commits are a *different lineage* from the `512802b9` that
`setup.sh` pins. `setup.sh` gives you the **working** stack; the patches are the
**experimental** one. Do not mix them without reading both.

## What setup.sh does NOT do — must be redone by hand

Host configuration, none of which is code (see `bench/deepswe/STATUS.md`):

* rootless Docker's `--disable-host-loopback` must be removed, or no container
  can reach the server at all; the host is then `10.0.2.2` from inside one
* docker data-root must live off `/` if root is near full
* Pier's squid `Safe_ports` needs 8000 added
* `--cpus ignore --memory ignore` if the host has no cgroup CPU delegation
* `/tmp/kt_topk_mode` must exist or `-topN` tiering is silently a no-op

## Verifying the rebuild actually worked

Three checks, in order, each of which failed silently at some point here:

1. `grep -cE 'Scheduler hit an exception|OutOfMemoryError' server.log` → **0**.
   A boot-time scheduler OOM still answers `/health` with 200 and dies an hour
   later.
2. Ask for ~400 tokens and time it → **~34 tok/s**. If top8 and top2 are the same
   speed, the substitution overlay did not land.
3. For any agentic workload, `grep -oE '#cached-token: [0-9]+' server.log` →
   **non-zero and rising**. Zero means `SLEEP_ON_IDLE` is flushing the prefix
   cache every step: an 11x slowdown invisible to health, VRAM, and queue depth.
