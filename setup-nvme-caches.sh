#!/usr/bin/env bash
# Format (first boot only) and mount the two local 3.5T NVMe scratch disks at
# /cache/nvme0 and /cache/nvme1. Restored 2026-06-22 after the original
# /usr/local/sbin/setup-nvme-caches.sh went missing (service was failing 203/EXEC).
# Preserves data if a filesystem already exists (only mkfs when unformatted),
# so a reboot does not wipe scratch. These are scratch disks: never put final
# weights/checkpoints/unique datasets here (see vm-storage-layout skill).
set -euo pipefail

map_dev() { :; }
declare -A MOUNTS=(
  [/dev/nvme0n1]=/cache/nvme0
  [/dev/nvme1n1]=/cache/nvme1
)

for dev in "${!MOUNTS[@]}"; do
  mp="${MOUNTS[$dev]}"
  [ -b "$dev" ] || { echo "skip: $dev not present"; continue; }
  mkdir -p "$mp"
  if ! blkid "$dev" >/dev/null 2>&1; then
    echo "formatting $dev (no filesystem found)"
    mkfs.ext4 -F -m0 -L "$(basename "$mp")" "$dev"
  fi
  if ! mountpoint -q "$mp"; then
    mount "$dev" "$mp"
    echo "mounted $dev -> $mp"
  fi
  chown bel:bel "$mp"
done

# --- NVMe swap (REQUIRED to serve GLM-5.2-FP8) -----------------------------
# The 704GB FP8 expert footprint is ~649GB anonymous vs only 629GB RAM, so with
# no swap the host OOM-kills the CPU expert build at ~layer 68/78. 256G (nvme0)
# + 128G (nvme1) of swap lets the cold LRU expert pages spill to NVMe so the
# build finishes and the server stays up. swappiness=10 keeps hot pages in RAM.
# Files live on the (now-persistent) scratch fs; only (re)created when missing.
declare -A SWAPS=(
  [/cache/nvme0/swapfile]=256G
  [/cache/nvme1/swapfile]=128G
)
for sf in "${!SWAPS[@]}"; do
  d="$(dirname "$sf")"
  mountpoint -q "$d" || { echo "skip swap $sf: $d not mounted"; continue; }
  if [ ! -f "$sf" ]; then
    echo "creating swapfile $sf (${SWAPS[$sf]})"
    fallocate -l "${SWAPS[$sf]}" "$sf"
    chmod 600 "$sf"
    mkswap "$sf" >/dev/null
  fi
  swapon "$sf" 2>/dev/null || true
done
sysctl -q vm.swappiness=10 || true
