"""
Drive the REAL kt RAWINT4 MoE kernel on one GLM-5.2-W4AFP8 layer (all experts on
CPU) and compare to a reference dequant+SwiGLU matmul. Lets us sweep packing
conventions (nibble order via KT_INT4_NIBBLE_SWAP, sign via KT_INT4_WEIGHT_XOR,
scale transpose via REF_*) without 4-min server restarts.
"""
import os, json, numpy as np, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from safetensors import safe_open
from kt_kernel.utils.amx import NativeMoEWrapper
from _paths import W4  # repo-relative; see int4_scripts/_paths.py

L, HID, MOE, NE, K = 3, 6144, 2048, 256, 8

mask = torch.zeros(NE, dtype=torch.bool)  # all experts on CPU
w = NativeMoEWrapper(
    layer_idx=L, num_experts=NE, num_experts_per_tok=K,
    hidden_size=HID, moe_intermediate_size=MOE,
    gpu_experts_mask=mask, cpuinfer_threads=64, threadpool_count=2,
    weight_path=W4, chunked_prefill_size=2048, method="RAWINT4", numa_nodes=[0, 1],
)
w.load_weights(torch.arange(NE, dtype=torch.long))

torch.manual_seed(0)
x = (torch.randn(1, HID, dtype=torch.bfloat16, device="cuda") * 0.1)
topk_ids = torch.arange(K, dtype=torch.long, device="cuda").view(1, K)
topk_w = torch.full((1, K), 1.0 / K, dtype=torch.float32, device="cuda")
stream = torch.cuda.current_stream().cuda_stream
out = w.forward(x, topk_ids, topk_w, stream)
torch.cuda.synchronize()
out = out.float().cpu()[0]

# ---- reference ----
idx = json.load(open(f"{W4}/model.safetensors.index.json"))["weight_map"]
def load_expert(e):
    base = f"model.layers.{L}.mlp.experts.{e}"
    sh = idx[f"{base}.gate_proj.weight"]
    t = {}
    with safe_open(f"{W4}/{sh}", framework="pt") as f:
        for p in ("gate", "up", "down"):
            t[p] = f.get_tensor(f"{base}.{p}_proj.weight").numpy().astype(np.uint8)
            t[p + "_s"] = f.get_tensor(f"{base}.{p}_proj.weight_scale_inv").float().numpy()
    return t

def dequant(packed, scale):
    lo = (packed & 0xF).astype(np.int16); lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((packed >> 4) & 0xF).astype(np.int16); hi = np.where(hi >= 8, hi - 16, hi)
    o = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    o[:, 0::2] = lo; o[:, 1::2] = hi
    sc = np.repeat(scale, 128, axis=1)[:, :o.shape[1]]
    return o * sc  # [out, in]

xf = x.float().cpu().numpy()[0]
ref = np.zeros(HID, dtype=np.float32)
for e in range(K):
    t = load_expert(e)
    g = dequant(t["gate"], t["gate_s"]); u = dequant(t["up"], t["up_s"]); d = dequant(t["down"], t["down_s"])
    gx = g @ xf; ux = u @ xf
    act = (gx / (1 + np.exp(-gx))) * ux
    ref += (1.0 / K) * (d @ act)
ref = torch.from_numpy(ref)

cos = torch.nn.functional.cosine_similarity(out, ref, dim=0).item()
print(f"\n==> cos(kernel, ref) = {cos:+.5f}")
print("kernel norm", out.norm().item(), "ref norm", ref.norm().item())
print("kernel[:6]", out[:6].tolist())
print("ref[:6]   ", ref[:6].tolist())
