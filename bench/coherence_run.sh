#!/usr/bin/env bash
# Coherence, measured long enough to actually fail.
#
# Substitution damage does not show at 200 tokens -- `bench/coherence.py` scored
# every 200-token sample on this box as clean, including tiers known to be bad.
# The failure mode is a slow collapse: the model stays locally fluent and starts
# looping somewhere past a few hundred tokens. So this generates LONG.
#
# CALIBRATION FIRST, VERDICT SECOND. The script always samples top0 and top8 on
# the SAME boot as whatever is being judged:
#
#   top8  full routing            -- known good, must score clean
#   top0  substitute everything   -- known DEGENERATE, must score degenerate
#
# If top0 scores clean, the detector is not sensitive enough at this length and
# NOTHING ELSE IN THE TABLE MAY BE READ AS A PASS. That is the whole point: a
# coherence check that cannot fail on the known-bad case is not a check, it is a
# rubber stamp. Same discipline as the depth-0 control in the prediction sweep.
#
# Usage: bench/coherence_run.sh <label> [tiers...]   (server must already be up)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
LABEL="${1:?need a label}"; shift
TIERS=("$@"); [ ${#TIERS[@]} -eq 0 ] && TIERS=(8 4 2 0)
OUT="$SP/coh_$LABEL"; mkdir -p "$OUT"
TOKENS=${TOKENS:-1200}

# Long-form, open-ended, and specific enough that drift is visible: a model that
# has lost the thread stops naming the right concepts, and looping shows up fast
# in a list-structured answer.
PROMPT="Explain in depth how a Mixture-of-Experts transformer works: the router, expert capacity, load balancing, why sparsity helps, and the failure modes in practice. Be thorough and specific."

for t in "${TIERS[@]}"; do
  echo "--- tier $t, $TOKENS tokens ---"
  curl -s -m 900 http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"GLM5.2-top$t\",\"max_tokens\":$TOKENS,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
  | .venv/bin/python -c "
import json,sys
s=sys.stdin.read()
try:
    d=json.loads(s)
except Exception:
    print('BAD RESPONSE:', s[:200]); raise SystemExit
c=(d.get('choices') or [{}])[0]
m=c.get('message') or {}
# The reply may arrive as reasoning_content rather than content -- that is what
# made two earlier sweeps look like they produced nothing at all.
txt=(m.get('content') or '') + (m.get('reasoning_content') or '')
open('$OUT/tier$t.txt','w').write(txt)
print('  chars', len(txt), 'finish', c.get('finish_reason'))
" || echo "  request failed"
done

echo
echo "=== coherence scores ($LABEL) ==="
.venv/bin/python bench/coherence.py "$OUT" --out "bench/profile_out/coh_$LABEL.json"
echo
.venv/bin/python - <<PY
import json
d = json.load(open("bench/profile_out/coh_$LABEL.json"))
t0 = d.get("tier0.txt", {}); t8 = d.get("tier8.txt", {})
ok8 = t8.get("verdict") == "clean"
bad0 = t0.get("verdict", "").startswith("DEGENERATE")
print("CALIBRATION: top8 clean =", ok8, " top0 degenerate =", bad0)
if not (ok8 and bad0):
    print("  ** DETECTOR NOT CALIBRATED AT THIS LENGTH -- no row here is a pass **")
else:
    print("  detector separates the known cases; other rows are readable")
PY
