"""
Re-download PhalaCloud/GLM-5.2-W4AFP8 to /cache/nvme0 after an ephemeral-disk wipe.
Resumable (snapshot_download skips already-complete files). ~373GB / 40 shards.
"""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download

DST = "/cache/nvme0/models/GLM-5.2-W4AFP8"
os.makedirs(DST, exist_ok=True)
p = snapshot_download(
    repo_id="PhalaCloud/GLM-5.2-W4AFP8",
    local_dir=DST,
    max_workers=16,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "tokenizer*", "*.model"],
)
print("DOWNLOAD COMPLETE ->", p, flush=True)
