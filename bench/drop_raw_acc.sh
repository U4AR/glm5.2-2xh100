#!/usr/bin/env bash
# TODO item 9: why is DROP worse than OMITTING the routed experts entirely?
#
# `chain_drop` (keep resident slots, zero the rest, RENORMALISE) reads 72.3%
# whole-layer coverage at d=1. `chain_shared` (run no routed experts at all)
# reads 75.1%. Adding a wrong routed update is worse than adding none, which is
# not obvious and is not yet explained.
#
# The only thing drop does that shared-only does not is the renormalise -- it
# rescales the two or three surviving experts up to carry the full routed
# weight, giving an update of roughly the right MAGNITUDE in the wrong
# DIRECTION. `chain_drop_raw` removes exactly that term and nothing else.
#
# Predictions, committed to before the run:
#   drop_raw >= chain_shared (75.1 @ d=1)  -> the renormalise is the culprit,
#                                             and a cheap variant is still live
#   drop_raw ~= chain_drop   (72.3 @ d=1)  -> the renormalise is innocent; the
#                                             damage is routing through resident
#                                             experts with no stand-in, i.e. the
#                                             search's substitutes do real work
#
# Same boot, same trajectory, all arms together, so nothing is compared across
# reboots. Waits for the item-3 queue to clear.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

while pgrep -f "bash bench/retry_queue.sh" >/dev/null 2>&1 ||
      pgrep -f "bash bench/item3_attrib.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== item-3 queue clear, starting drop_raw accuracy run ==="
sleep 20

KT_CHAIN_ARMS=direct,post,chain,chain_drop,chain_drop_raw,chain_post,chain_shared \
  bash bench/drop_acc.sh 2>&1 | tee "$SP/drop_raw_acc.out"
echo "=== drop_raw_acc done ==="
