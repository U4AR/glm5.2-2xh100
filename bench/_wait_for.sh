# Optional queue chaining, OFF by default.
#
# These harnesses each own the whole box (they kill and re-boot the server), so
# during development they were chained by having each one spin until the
# previous one printed its done-marker. That chaining was hardcoded, which made
# every script unrunnable anywhere else: a fresh clone would sit for 2.5 hours
# waiting on a marker file no one was ever going to write.
#
# Now it is opt-in. Standalone (the normal case, and the only case on another
# machine) the script starts immediately.
#
#   wait_for_marker "=== blocks_fine done ==="  blocks_fine.out
#
# waits only when RUNGLM_CHAIN=1. Set that when queueing several ladders
# back-to-back on one box, exactly as the original runs did.
wait_for_marker () {
  local marker="$1" file="$2"
  [ "${RUNGLM_CHAIN:-0}" = "1" ] || return 0
  echo "chaining: waiting for '$marker' in $SP/$file"
  local i
  for i in $(seq 1 "${RUNGLM_CHAIN_TRIES:-900}"); do
    grep -qaF "$marker" "$SP/$file" 2>/dev/null && return 0
    sleep 10
  done
  echo "chaining: TIMED OUT waiting for '$marker'; starting anyway"
}
