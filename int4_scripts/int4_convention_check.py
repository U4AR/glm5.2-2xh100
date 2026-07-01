"""
Determine the W4AFP8 int4 packing/sign/scale convention by cross-checking
dequant(W4AFP8 expert) against dequant(FP8 expert) for the SAME expert.
Both models are GLM-5.2 -> expert weights should match within quant error.
"""
import json, struct, numpy as np, torch
from safetensors import safe_open
from _paths import W4, FP8  # paths configured in int4_scripts/_paths.py (repo-relative)
L, E = 3, 0
proj = "gate_proj"

def find_shard(model_dir, key):
    idx=json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    return idx[key]

# ---- FP8 reference dequant ----
base=f"model.layers.{L}.mlp.experts.{E}.{proj}"
sh=find_shard(FP8, f"{base}.weight")
with safe_open(f"{FP8}/{sh}", framework="pt") as f:
    w_fp8 = f.get_tensor(f"{base}.weight")            # F8_E4M3 [2048,6144]
    s_fp8 = f.get_tensor(f"{base}.weight_scale_inv")  # F32 [16,48]
w_fp8f = w_fp8.to(torch.float32)
# expand 128x128 block scale
s_exp = s_fp8.repeat_interleave(128,0)[:w_fp8f.shape[0]].repeat_interleave(128,1)[:, :w_fp8f.shape[1]]
ref = (w_fp8f * s_exp)  # [2048,6144]
print("FP8 ref shape", ref.shape, "abs-mean", ref.abs().mean().item())

# ---- W4AFP8 raw ----
sh2=find_shard(W4, f"{base}.weight")
with safe_open(f"{W4}/{sh2}", framework="pt") as f:
    w_i8 = f.get_tensor(f"{base}.weight")             # I8 [2048,3072]
    s_w4 = f.get_tensor(f"{base}.weight_scale_inv")   # BF16 [2048,48]
w_i8 = w_i8.numpy().astype(np.uint8)   # raw bytes
s_w4 = s_w4.to(torch.float32).numpy()  # [2048,48]
print("w4 packed", w_i8.shape, "scale", s_w4.shape)

def unpack(order, sign):
    lo = w_i8 & 0x0F
    hi = (w_i8 >> 4) & 0x0F
    if sign=="xor8":
        lo = (lo.astype(np.int16) ^ 0x8) - 8
        hi = (hi.astype(np.int16) ^ 0x8) - 8
    elif sign=="sub8":
        lo = lo.astype(np.int16) - 8
        hi = hi.astype(np.int16) - 8
    elif sign=="twos":
        lo = np.where(lo>=8, lo.astype(np.int16)-16, lo.astype(np.int16))
        hi = np.where(hi>=8, hi.astype(np.int16)-16, hi.astype(np.int16))
    # interleave to width 6144
    out=np.empty((w_i8.shape[0], w_i8.shape[1]*2), dtype=np.int16)
    if order=="lo_first":
        out[:,0::2]=lo; out[:,1::2]=hi
    else:
        out[:,0::2]=hi; out[:,1::2]=lo
    return out

def score(deq):
    a=torch.from_numpy(deq.astype(np.float32))
    # group scale along K=6144, group128 -> 48 groups
    sc=torch.from_numpy(s_w4).repeat_interleave(128,1)[:, :a.shape[1]]
    for mode,scaled in [("mul", a*sc), ("div", a/sc)]:
        cos=torch.nn.functional.cosine_similarity(scaled.flatten(), ref.flatten(), dim=0).item()
        rel=((scaled-ref).abs().mean()/ref.abs().mean()).item()
        yield mode, cos, rel

best=None
for order in ("lo_first","hi_first"):
    for sign in ("xor8","sub8","twos"):
        deq=unpack(order,sign)
        for mode,cos,rel in score(deq):
            tag=f"order={order:8s} sign={sign:5s} scale={mode}"
            print(f"  {tag}: cos={cos:+.4f} rel_err={rel:.3f}")
            if best is None or cos>best[1]:
                best=(tag,cos,rel)
print("\nBEST:", best)
