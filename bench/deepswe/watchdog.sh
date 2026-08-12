#!/usr/bin/env bash
# Watch the sglang server while a long agent benchmark runs, and restart it if it dies.
#
#   setsid nohup bash bench/deepswe/watchdog.sh > /dev/null 2>&1 < /dev/null &
#   tail -f bench/deepswe/watchdog.log
#
# WHY: the 06:42 boot answered /health with 200 for an hour while one of its two
# schedulers had already died of CUDA OOM at startup, then went down mid-task. The
# agent saw only "Connection error" at step 11, 50 minutes after the fact. So this
# checks three independent things, not just /health:
#
#   health      the HTTP endpoint            -- catches a hung or dead front end
#   schedulers  2 sglang::scheduler procs    -- catches a half-crashed boot that
#                                              still serves; /health cannot see this
#   vram        >1 GiB resident on GPU 0     -- catches weights being torn down
#
# A restart is only triggered after FAILS_BEFORE_RESTART consecutive bad checks, so
# a single slow poll during a heavy prefill does not bounce a healthy server.
set -uo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
REPO=/data/models/RunGLM
LOG="$HERE/watchdog.log"
INTERVAL="${INTERVAL:-60}"
FAILS_BEFORE_RESTART="${FAILS_BEFORE_RESTART:-2}"
fails=0
restarts=0
ooms_seen=$(grep -cE 'Scheduler hit an exception|OutOfMemoryError' server.log 2>/dev/null)

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG"; }

log "watchdog start (interval ${INTERVAL}s, restart after ${FAILS_BEFORE_RESTART} bad checks)"

while true; do
  # 10s was too tight: /health queues behind a long forward pass, and a 100k-token
  # prefill on the CPU expert path can hold it far longer than that.
  health=$(curl -s -m 45 -o /dev/null -w '%{http_code}' localhost:8000/health 2>/dev/null)
  # Exclude zombies: a killed server leaves <defunct> sglang::schedul entries that
  # pgrep happily counts, so the "2 schedulers alive" check would pass on a corpse.
  scheds=$(ps -eo stat=,comm= 2>/dev/null | awk '$1 !~ /Z/ && $2 ~ /sglang::schedul/' | wc -l)
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  vram=${vram:-0}
  # A boot-time scheduler exception is invisible to /health but fatal later.
  # `grep -c` exits 1 on zero matches, so a `|| echo 0` fallback would append a
  # SECOND line to the captured value; let the 0 it already printed stand.
  ooms=$(grep -cE 'Scheduler hit an exception|OutOfMemoryError' server.log 2>/dev/null)
  agents=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c 'main-1')

  # A RISING oom count is the failure this file missed once already: on 2026-08-11
  # the MTP draft's triton forward_extend OOM'd four times under 3 concurrent
  # prefills, killing every agent, while health stayed 200 and both schedulers
  # stayed alive the whole time. Count, don't just report.
  if [ "$ooms" -gt "$ooms_seen" ]; then
    log "!!! server.log OOM/exception count rose $ooms_seen -> $ooms (agents will have errored)"
    ooms_seen=$ooms
    fails=$FAILS_BEFORE_RESTART   # force a restart; a degraded server keeps failing
  fi

  # A restart needs POSITIVE EVIDENCE OF DEATH, not merely absent evidence of life.
  # On 2026-08-12 this killed a perfectly healthy server: /health exceeded the 10s
  # curl timeout while the scheduler prefilled a 113k-token context, and the old
  # all-signals-must-be-green rule let one slow probe outvote `scheds=2, ooms=0,
  # vram RISING`. A busy server and a dead one both fail to answer; only the
  # corroborating signals tell them apart.
  #   dead    = schedulers gone, or VRAM released, or the OOM counter rose
  #   slow    = health silent while schedulers hold and VRAM is steady/climbing
  dead=0
  [ "$scheds" -lt 2 ] && dead=1
  [ "$vram" -le 1024 ] && dead=1

  if [ "$dead" -eq 0 ]; then
    [ "$fails" -gt 0 ] && log "recovered (health=$health scheds=$scheds vram=${vram}MiB)"
    fails=0
    if [ "$health" = "200" ]; then
      log "ok health=$health scheds=$scheds vram=${vram}MiB agents=$agents ooms=$ooms"
    else
      log "BUSY health=$health scheds=$scheds vram=${vram}MiB agents=$agents ooms=$ooms (alive; not restarting)"
    fi
  else
    fails=$((fails + 1))
    log "BAD ($fails/$FAILS_BEFORE_RESTART) health=$health scheds=$scheds vram=${vram}MiB agents=$agents ooms=$ooms"
    if [ "$fails" -ge "$FAILS_BEFORE_RESTART" ]; then
      restarts=$((restarts + 1))
      log "!!! RESTARTING SERVER (restart #$restarts) -- any in-flight agent step will have errored"
      pkill -9 -f 'sglang::' 2>/dev/null
      pkill -9 -f 'launch_server' 2>/dev/null
      sleep 10
      # Sourced, never duplicated: hardcoding the config here has silently drifted
      # from the launch config twice, each time destroying a run.
      ( cd "$REPO" && . "$HERE/server_env.sh" \
        && setsid nohup ./run_fast.sh >> "$HERE/server.log" 2>&1 < /dev/null & )
      log "waiting for server to come back..."
      for _ in $(seq 1 40); do
        sleep 15
        [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8000/health)" = "200" ] && break
      done
      log "back up: health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8000/health)"
      fails=0
    fi
  fi
  sleep "$INTERVAL"
done
