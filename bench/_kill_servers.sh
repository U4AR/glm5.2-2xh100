#!/usr/bin/env bash
# Kill sglang servers from a FILE, never from an inline shell command.
# An inline `pkill -f sglang.launch_server` matches the calling shell's own
# command line (the pattern is in it) and kills the caller -- exit 144. This has
# cost two shells already.
pkill -f sglang.launch_server >/dev/null 2>&1
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
echo "servers down, VRAM free"
