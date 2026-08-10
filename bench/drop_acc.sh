#!/usr/bin/env bash
# Does the CHEAP substitution predict as well as the expensive one?
#
# Stage H2 found the walk's biggest attributed cost is the resident-substitution
# SEARCH: 6.55 ms/step, ~20 tiny kernels (a topk across all 256 experts, two
# argsorts, gathers, scatters). KT_CHAIN_PF_SUB=drop replaces it with "keep the
# routed slots that are already resident, zero the rest, renormalise" and cuts
# that to 1.55 ms.
#
# But the walk is PAID FOR its fidelity: a worse hidden state means a worse
# guess at the next router, which is the whole product. So the cost win is not
# bankable until the accuracy is measured. This boots the eager instrument once
# with BOTH arms in the same run -- `chain` (search) and `chain_drop` (drop) --
# so the comparison is same-boot, same-trajectory, same-token.
#
# Waits for bench/walk_fix.sh to finish first so the two never share the GPU.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

while pgrep -f "bash bench/walk_fix.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== walk_fix.sh finished, starting drop-accuracy run ==="
sleep 20

# `chain_post` rides along because TODO item 6's defect is now fixed (its walk
# used to double-apply layer L) and it is the exact analogue of the shipped
# mechanism: start from the layer's REAL output, approximate only what follows.
# It carries its own validation -- at d=1 it must now equal `post` exactly.
KT_CHAIN_ARMS=${KT_CHAIN_ARMS:-direct,post,chain,chain_drop,chain_post,chain_shared} \
KT_CHAIN_DEPTH=4 \
KT_CHAIN_STRIDE=6 \
KT_CHAIN_P=2,3,4 \
KT_CHAIN_WALK_K=8 \
TOKENS=300 REQS=2 \
  bash bench/chain_predict_run.sh 2>&1 | tee "$SP/drop_acc.out"
echo "=== drop_acc done ==="
