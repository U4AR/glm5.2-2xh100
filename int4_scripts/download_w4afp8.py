"""
Download PhalaCloud/GLM-5.2-W4AFP8 (e.g. after an ephemeral-disk wipe).
Resumable (snapshot_download skips already-complete files). ~373GB / 40 shards.

Destination defaults to ./weights/GLM-5.2-W4AFP8 under the repo; override with
the WEIGHTS env var, e.g.  WEIGHTS=/mnt/nvme/GLM-5.2-W4AFP8 python download_w4afp8.py
(point this at fast scratch with ~380GB free).
"""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.environ.get("WEIGHTS", os.path.join(_repo_root, "weights", "GLM-5.2-W4AFP8"))
os.makedirs(DST, exist_ok=True)
p = snapshot_download(
    repo_id="PhalaCloud/GLM-5.2-W4AFP8",
    local_dir=DST,
    max_workers=16,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "tokenizer*", "*.model"],
)
print("DOWNLOAD COMPLETE ->", p, flush=True)
