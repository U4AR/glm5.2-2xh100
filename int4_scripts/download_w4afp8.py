"""
Download PhalaCloud/GLM-5.2-W4AFP8 (e.g. after an ephemeral-disk wipe).
Resumable (snapshot_download skips already-complete files). ~373GB / 40 shards.

Destination defaults to ./weights/GLM-5.2-W4AFP8 under the repo (matches
config.sh). Override in one of two ways:
    WEIGHTS_DIR=/mnt/nvme python download_w4afp8.py   # sets the parent dir
    WEIGHTS=/mnt/nvme/GLM-5.2-W4AFP8 python download_w4afp8.py   # exact dir
Point this at fast scratch with ~380GB free.

NFS/network-filesystem safety: hf-xet's parallel writes stall on some NFS
mounts, so Xet is disabled and the worker count is low by default. If your
destination is a fast local disk you can crank throughput back up:
    HF_MAX_WORKERS=16 python download_w4afp8.py
"""
import os

# IMPORTANT: set these BEFORE importing huggingface_hub.
# Disable Xet (its parallel writes stall on some NFS mounts) and give the
# plain HTTPS transfer generous timeouts so slow mounts don't error out.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
# Don't inherit an accidental high-performance Xet mode from the environment.
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

from huggingface_hub import snapshot_download

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_weights_dir = os.environ.get("WEIGHTS_DIR", os.path.join(_repo_root, "weights"))
DST = os.environ.get("WEIGHTS", os.path.join(_weights_dir, "GLM-5.2-W4AFP8"))
# One worker is safest on NFS; bump HF_MAX_WORKERS for fast local disks.
MAX_WORKERS = int(os.environ.get("HF_MAX_WORKERS", "1"))

os.makedirs(DST, exist_ok=True)
p = snapshot_download(
    repo_id="PhalaCloud/GLM-5.2-W4AFP8",
    repo_type="model",
    local_dir=DST,
    max_workers=MAX_WORKERS,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "tokenizer*", "*.model"],
    force_download=False,
)
print("DOWNLOAD COMPLETE ->", p, flush=True)
