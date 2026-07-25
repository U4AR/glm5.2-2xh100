"""
Drive the real SGLang W4AFP8 GPU path on one GLM-5.2 MoE layer (experts
0..NE-1, all GPU) and compare to a reference dequant+SwiGLU matmul.

SM90+ defaults to the existing CUTLASS W4A8 method. SM80-SM89 defaults to the
portable Marlin W4A16 method. KT_W4AFP8_GPU_BACKEND can force either backend.

If cos(kernel, ref) is high here but the full server garbages, the bug is in the
kt_ep_wrapper integration (weight loading / expert index mapping), not the kernel.
If cos is low/NaN here, the format/scale handling of OUR W4AFP8 weights is wrong.

env: LAYER (default 3), NE (default 8), DUMP=1 to print intermediate stats.
"""
import os, json, numpy as np, torch
from types import SimpleNamespace
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from safetensors import safe_open

# --- avoid needing a real distributed/parallel init ---
import sglang.srt.layers.moe.cutlass_w4a8_moe as cwm
cwm.get_moe_expert_parallel_world_size = lambda: 1  # monkeypatch: single rank
from sglang.srt.server_args import set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(
    SimpleNamespace(enable_deterministic_inference=False)
)

from sglang.srt.layers.quantization.w4afp8 import (
    W4AFp8Config,
    W4AFp8MarlinMoEMethod,
    W4AFp8MoEMethod,
)
from _paths import W4  # repo-relative; see int4_scripts/_paths.py

LAYER = int(os.getenv("LAYER", "3"))
HID, MOE, GROUP = 6144, 2048, 128
NE = int(os.getenv("NE", "8"))
K = NE  # route the single token to all NE experts (deterministic)
DEV = "cuda"

idx = json.load(open(f"{W4}/model.safetensors.index.json"))["weight_map"]

def names_for(e):
    base = f"model.layers.{LAYER}.mlp.experts.{e}"
    return base

# W4AFP8 names input scales as w1/w2/w3 (gate/down/up):
#   w1.input_scale = gate, w3.input_scale = up, w2.input_scale = down
probe = f"model.layers.{LAYER}.mlp.experts.0."
have_inscale = (probe + "w1.input_scale") in idx
print("has w1/w2/w3.input_scale:", have_inscale)

def get(name):
    sh = idx[name]
    with safe_open(f"{W4}/{sh}", framework="pt") as f:
        return f.get_tensor(name)

# ---------------- build the layer + method ----------------
quant = W4AFp8Config.from_config({"quant_method": "w4afp8"})
gpu_backend = os.getenv("KT_W4AFP8_GPU_BACKEND", "auto").strip().lower()
capability = torch.cuda.get_device_capability()
use_marlin = gpu_backend in ("marlin", "marlin_sm80", "sm89") or (
    gpu_backend == "auto" and capability[0] < 9
)
method = (
    W4AFp8MarlinMoEMethod(quant)
    if use_marlin
    else W4AFp8MoEMethod(quant)
)
print(
    f"GPU backend: {type(method).__name__} on "
    f"SM{capability[0]}{capability[1]}"
)
layer = torch.nn.Module().to(DEV)
layer.to(DEV)

def _wl(*a, **k):
    pass

# create_weights registers params ON CPU by default; move to cuda after fill
method.create_weights(
    layer, num_experts=NE, hidden_size=HID,
    intermediate_size_per_partition=MOE, params_dtype=torch.bfloat16,
    weight_loader=_wl,
)

# ---------------- load real W4AFP8 tensors ----------------
# w13_weight[e, 0:MOE] = gate, [MOE:2MOE] = up   (int8 packed [out, in/2])
# w2_weight[e]         = down
# scales analogous; input scales: w13[:,0]=gate, w13[:,1]=up; w2=down
w13 = layer.w13_weight.data
w2 = layer.w2_weight.data
w13s = layer.w13_weight_scale_inv.data
w2s = layer.w2_weight_scale_inv.data
w13is = layer.w13_input_scale.data
w2is = layer.w2_input_scale.data

for e in range(NE):
    base = names_for(e)
    g = get(f"{base}.gate_proj.weight")          # int8 [MOE, HID/2]
    u = get(f"{base}.up_proj.weight")
    d = get(f"{base}.down_proj.weight")           # int8 [HID, MOE/2]
    gs = get(f"{base}.gate_proj.weight_scale_inv").float()  # [MOE, HID/GROUP]
    us = get(f"{base}.up_proj.weight_scale_inv").float()
    ds = get(f"{base}.down_proj.weight_scale_inv").float()  # [HID, MOE/GROUP]
    w13[e, 0:MOE] = g.to(torch.int8)
    w13[e, MOE:2 * MOE] = u.to(torch.int8)
    w2[e] = d.to(torch.int8)
    w13s[e, 0:MOE] = gs
    w13s[e, MOE:2 * MOE] = us
    w2s[e] = ds
    if have_inscale:
        w13is[e, 0] = get(f"{base}.w1.input_scale").to(torch.bfloat16).reshape(())  # gate
        w13is[e, 1] = get(f"{base}.w3.input_scale").to(torch.bfloat16).reshape(())  # up
        w2is[e] = get(f"{base}.w2.input_scale").to(torch.bfloat16).reshape(())      # down

# move everything to cuda
for nm in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv",
           "w13_input_scale", "w2_input_scale"):
    p = getattr(layer, nm)
    p.data = p.data.to(DEV)
# strides were created on CPU device (layer had no device) -> move
for nm in ("a_strides1", "b_strides1", "c_strides1", "a_strides2", "b_strides2",
           "c_strides2", "s_strides13", "s_strides2", "expert_offsets",
           "problem_sizes1", "problem_sizes2"):
    setattr(method, nm, getattr(method, nm).to(DEV))

print("input scales (gate,up,down) expert0:", w13is[0].tolist(), float(w2is[0]))

# ---------------- process_weights_after_loading ----------------
if os.getenv("SKIP_INTERLEAVE") == "1":
    # Replicate ONLY the input_scale collapse (kt's _prepare_weight_fp8 no-op
    # postprocess never interleaves the weight scales). Tests whether missing
    # interleave alone reproduces the server garbage.
    import torch as _t
    from torch.nn.parameter import Parameter as _P
    dev = layer.w2_weight.device
    layer.w13_weight_scale_inv = _P(layer.w13_weight_scale_inv.to(_t.bfloat16), requires_grad=False)
    layer.w2_weight_scale_inv = _P(layer.w2_weight_scale_inv.to(_t.bfloat16), requires_grad=False)
    layer.w13_input_scale = _P(_t.tensor([layer.w13_input_scale.max().float().item()], dtype=_t.float32, device=dev), requires_grad=False)
    layer.w2_input_scale = _P(_t.tensor([layer.w2_input_scale.max().float().item()], dtype=_t.float32, device=dev), requires_grad=False)
    print("[SKIP_INTERLEAVE] scales NOT interleaved (only cast + input_scale collapse)")
else:
    method.process_weights_after_loading(layer)

# ---------------- run selected GPU backend ----------------
torch.manual_seed(0)
x = (torch.randn(1, HID, dtype=torch.bfloat16, device=DEV) * 0.1)
topk_ids = torch.arange(K, dtype=torch.int32, device=DEV).view(1, K)
topk_w = torch.full((1, K), 1.0 / K, dtype=torch.float32, device=DEV)
method.create_moe_runner(
    layer, SimpleNamespace(activation="silu", routed_scaling_factor=1.0)
)
dispatch = SimpleNamespace(
    hidden_states=x,
    topk_output=(
        topk_w,
        topk_ids,
        torch.zeros((1, NE), dtype=torch.float32, device=DEV),
    ),
)
out = method.apply(layer, dispatch).hidden_states
torch.cuda.synchronize()
out = out.float().cpu()[0]

# ---------------- reference ----------------
def dequant(packed_i8, scale):  # packed [out, in/2] int8 -> float [out, in]
    pk = packed_i8.numpy().astype(np.uint8)
    lo = (pk & 0xF).astype(np.int16); lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((pk >> 4) & 0xF).astype(np.int16); hi = np.where(hi >= 8, hi - 16, hi)
    o = np.empty((pk.shape[0], pk.shape[1] * 2), dtype=np.float32)
    o[:, 0::2] = lo; o[:, 1::2] = hi
    sc = np.repeat(scale.numpy(), GROUP, axis=1)[:, :o.shape[1]]
    return o * sc

xf = x.float().cpu().numpy()[0]
ref = np.zeros(HID, np.float32)
for e in range(K):
    base = names_for(e)
    g = dequant(get(f"{base}.gate_proj.weight"), get(f"{base}.gate_proj.weight_scale_inv").float())
    u = dequant(get(f"{base}.up_proj.weight"), get(f"{base}.up_proj.weight_scale_inv").float())
    d = dequant(get(f"{base}.down_proj.weight"), get(f"{base}.down_proj.weight_scale_inv").float())
    gx = g @ xf; ux = u @ xf
    act = (gx / (1 + np.exp(-gx))) * ux
    ref += (1.0 / K) * (d @ act)
ref = torch.from_numpy(ref)

cos = torch.nn.functional.cosine_similarity(out, ref, dim=0).item()
print(f"\n==> cos(kernel, ref) = {cos:+.5f}")
print("kernel nan?", torch.isnan(out).any().item(), "norm", out.norm().item(), "ref norm", ref.norm().item())
print("kernel[:6]", out[:6].tolist())
print("ref[:6]   ", ref[:6].tolist())
