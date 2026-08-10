#!/usr/bin/env bash
# Re-run the prediction-accuracy sweep, this time WITH its controls armed.
#
# The numbers on disk were doubted, and the doubt was half right. The harness
# does carry a known-answer control -- depth 0 predicts layer L from layer L's
# own hidden state with layer L's own gate, which must read 100% -- and it reads
# exactly 100.00%. So `direct`, the scoring path and the ground truth are sound.
#
# But NOTHING validated the walk. Its one built-in cross-check (`chain_post` at
# d=1 degenerates to `post`) FAILED in that data, 79.32% against 84.79%, because
# the walk loop applied layer L's MoE on top of a state that already contained
# it. That fix is in the source now but has never been re-measured, and `chain`
# -- the arm whose +3.4/+6.9 point advantage is the whole question -- has no
# control at all.
#
# This run arms two:
#
#   chain_exact @ d1  ==  post @ d1     KT_CHAIN_EXACT=1 walks with GENUINE full
#                                       routing, so after layer L the state is
#                                       exactly r_next, which is what `post`
#                                       reads. Any gap is the walk machinery.
#   chain_post  @ d1  ==  post @ d1     the repaired degeneracy.
#
# Plus depth 0 == 100%, which must keep holding.
#
# GPU_EXPERTS=100 on purpose, NOT the script's 104 default: residency decides
# which experts count as "needed", and 100 + 4 slots is the configuration every
# prefetch number in Stage H12-H14 was measured in. (It also explains why
# whole-layer coverage read 35.3% here against the 60.8% recorded at 104 --
# fewer residents, more experts needed per layer, a harder bar. Unconfirmed
# until this run, which is another reason to pin residency.)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

wait_for_marker "=== cost_isolate done ===" cost_isolate.out
grep -qaF "=== cost_isolate done ===" "$SP/cost_isolate.out" 2>/dev/null || {
  echo "cost_isolate never finished; not starting chain re-run"; exit 1; }

# chain_predict_run.sh deletes the rank files on entry. Keep yesterday's, so the
# new run can be compared against them instead of replacing them.
mkdir -p bench/profile_out/prev
for f in bench/profile_out/chain_predict.rank*.json; do
  [ -e "$f" ] && cp -f "$f" "bench/profile_out/prev/$(basename "$f" .json).2026-08-07.json"
done

sleep 20
KT_CHAIN_EXACT=1 GPU_EXPERTS=100 KT_CHAIN_P=1,2,3,4 exec bash bench/chain_predict_run.sh
