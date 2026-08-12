# DeepSWE tier benchmark — status 2026-08-12 (stopped for teardown, no graded result)

> **Day 2 summary (2026-08-12).** Three more runs, still **zero graded verdicts**.
> Two died from harness faults (both mine, both config-drift between the launcher
> and `watchdog.sh`); the third ended cleanly on a verified harness by exhausting a
> full 131,072-token context in 99 steps without submitting.
>
> **The measurement that survives scrutiny.** On the final run — correct config, one
> server crash survived via retries, no truncation by me — the top-2 tier used
> **130,476 tokens over 99 steps** and never produced a submission. Datacurve's
> graded GLM-5.2 solves the same task in a median of **48 steps / 61,110 tokens**
> (2 of 4 rollouts pass). That is ~2x the context and ~2x the steps without
> converging, and the pattern repeated across three context budgets (81,920 →
> 131,072 twice).
>
> **Two confounds prevent calling it tier degradation.** (1) No `top8` control was
> ever run on this box. (2) The final runs used `temperature=1.2, top_p=0.95` at the
> user's request, above the graded default — high temperature plausibly produces
> exactly this explore-forever behaviour. **The single most valuable next run is
> `top8` at the identical T=1.2**, which changes only the tier and separates the two.
>
> **Fixed on day 2**, both now permanent:
> * `server_env.sh` is the single source of server config, sourced by
>   `start_server.sh` AND `watchdog.sh`. Duplicating it destroyed two runs — a stale
>   `MEM_FRACTION=0.88` (boot loop) and a stale `MAX_TOTAL_TOKENS=81920` with no
>   `CONTEXT_LENGTH`, which silently shrank the context to 81,920 under a 125k
>   conversation and killed a 200-step agent instantly.
> * The watchdog now restarts only on **positive evidence of death** (schedulers
>   gone / VRAM released), not on a silent `/health`. Its old all-signals-green rule
>   let one slow probe outvote `scheds=2, ooms=0, vram rising` and it killed a
>   perfectly healthy server mid-run. Health timeout 10s → 45s.
> * `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=25` gives ~22 min of retry cover, so the
>   agent survives a server restart mid-conversation. **Demonstrated twice**,
>   including across a full reconfiguration. Caveat: retries make a *deterministic*
>   failure worse — they resurrected a request that OOM'd the server on contact, 24
>   times.
> * **KV pool must EXCEED context length**, not equal it (now 163,840 vs 131,072).
>   Pool == context left zero slack, nothing to retract, and gave a deterministic
>   `Out of memory even after retracting all other requests in the decode batch`.
>
> Everything below is the day-1 record and remains accurate.

---

# Day 1 — status 2026-08-11

## What this is

Measuring the quality cost of the top-2 expert-substitution tier against GLM-5.2's
**measured** border, using DeepSWE v1.1 — the only public source of per-question
GLM-5.2 pass/fail (`https://deepswe.datacurve.ai/artifacts/v1.1/trials.json`).
At `effort=high`, 66 of 113 tasks are fractional (some of 4 rollouts pass, some
fail); those are the border. See `../../memory/glm52-deepswe-per-question.md`.

## Result so far

**Zero graded verdicts.** Seven attempts, seven infrastructure failures, none of
them the model. The verifier has never run. Every `jobs/ERR*` directory is a
server-side fault, NOT a task failure — do not read them as scores.

| # | died at | cause |
|---|---------|-------|
| 1 | step 11  | `MEM_FRACTION=0.95`: TP1 OOM'd creating the MTP draft at boot with 43 MiB free. Server ran orphaned for an hour, then died. `/health` said 200 throughout. |
| 2 | step ~15 x3 | MTP draft's triton `forward_extend` needs ~542 MiB per concurrent prefill; 3 agents, 472 MiB free. 4 OOMs. |
| 3 | boot | `MEM_FRACTION=0.88` is BELOW the weight floor — lowering it is not the lever. |
| 4 | 4 steps/32 min | `max_running_requests=2` vs 3 agents + litellm retry zombies -> `#queue-req: 5`, starvation. Abandoned client requests do not free their server slot. |
| 5 | stalled | **`--sleep-on-idle` (default ON) flushed the prefix cache every step.** `#cached-token: 0` for hours; every step re-prefilled the whole context on the CPU path. 100 s/step. |
| 6 | step 82 | GPU prefill + 98k KV pool together: pool filled to 94.8/95.8 GB, prefill workspace no longer fit. |
| 7 | step 142 | `ContextWindowExceededError`: 81,994 > 81,920. Server healthy, no OOM. |

## The one real signal

Run 7 reached **142 steps / 81,201 peak context** on `ts-pattern-match-each`
without submitting. Datacurve's graded GLM-5.2 solved it in a median of **48 steps
/ 61,110 tokens**. ~3x the steps is what substitution degradation would look
like — but it is a HYPOTHESIS, not a finding: one rollout, no top8 control on this
box, and a truncated run is indistinguishable from a slow-but-correct one. It
becomes evidence only if top8 finishes this task in ~50 steps on the same server.

## Working config (booted clean 12:18, still running)

    GPU_EXPERTS=88 MEM_FRACTION=0.94 MAX_TOTAL_TOKENS=131072 CONTEXT_LENGTH=131072 \
    CHUNKED_PREFILL=2048 MAX_RUNNING=8 SLEEP_ON_IDLE=0 KT_GPU_PREFILL_THRESHOLD=0 \
    TRITON_CACHE_DIR=/data/triton-cache ./run_fast.sh

`SLEEP_ON_IDLE=0` is the important one: it is the difference between 9 s/step and
100 s/step for any agentic workload, and `run_server_int4.sh` defaults it to 1.
`GPU_EXPERTS=88` (down from 96) pays the VRAM for the 131k context. Untested at
this size — verify `#cached-token` is non-zero and watch VRAM on the first run.

## To resume

    cd /data/models/RunGLM/bench/deepswe
    setsid nohup bash ./watchdog.sh > /dev/null 2>&1 < /dev/null &     # config inside MUST match the launch
    setsid nohup pier run -p deep-swe/tasks/ts-pattern-match-each \
      -a mini-swe-agent -m openai/GLM5.2-top2 --ak model_class=litellm \
      --ae OPENAI_BASE_URL=http://10.0.2.2:8000/v1 \
      --ae OPENAI_API_KEY=dummy --ae MSWEA_API_KEY=dummy \
      --cpus ignore --memory ignore --agent-timeout-multiplier 4 -n 1 \
      -o jobs --job-name top2_ts-pattern-match-each -y > logs_top2_ts.log 2>&1 &

Then the same with `-m openai/GLM5.2-top8` for the control. ~25 min per tier per
task. `python report.py top2 top8` prints the border table.

### Watch these three, in this order

1. `grep -oE '#cached-token: [0-9]+' server.log | tail` — MUST be non-zero and
   growing. Zero means the prefix cache is dead and the run will take 11x longer.
2. `grep -oE '#queue-req: [0-9]+' server.log | tail` — must stay ~0.
3. `tail watchdog.log` — `ooms` must stay 0; a rising count means agents errored.

Health, VRAM and scheduler count were all green through failures 4, 5 and 7. They
are necessary, not sufficient.

## Host changes made (revert if unwanted)

- `~/.config/systemd/user/docker.service.d/host-loopback.conf` — re-enables
  container->host access (host is `10.0.2.2`). **Weakens container isolation
  host-wide.** Delete + `systemctl --user restart docker` to revert.
- `~/.config/docker/daemon.json` — data-root moved to `/data/docker-bel` (root
  disk was 93% full). Stale 23 GB copy still at `~/.local/share/docker/overlay2`,
  safe to delete.
- Patched `pier/environments/agent_setup.py`: squid `Safe_ports` +8000.
