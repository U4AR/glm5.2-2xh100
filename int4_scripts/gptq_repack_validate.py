"""
Repack W4AFP8 layer-3 experts 0..7 -> kt GPTQ_INT4 format (AutoGPTQ: int32
qweight [in/8,out] packed 8x4bit along K low-first; scales [groups,out] fp16;
symmetric zero=8), write to a temp dir, then drive the REAL kt GPTQ_INT4 kernel
via NativeMoEWrapper and compare to a reference dequant+SwiGLU matmul.

env GPTQ_OFFSET=8 (default) stores q=v+8 (unsigned+zero8); =0 stores signed nibble.
"""
import os, json, numpy as np, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from safetensors import safe_open
from safetensors.torch import save_file

W4 = "/cache/nvme0/models/GLM-5.2-W4AFP8"
OUT = "/cache/nvme1/gptq_test_layer3"
L, HID, MOE, NE_TEST, K = 3, 6144, 2048, 8, 8
OFFSET = int(os.getenv("GPTQ_OFFSET", "8"))
os.makedirs(OUT, exist_ok=True)

idx = json.load(open(f"{W4}/model.safetensors.index.json"))["weight_map"]
def w4_signed(packed_u8):  # [out, in/2] -> signed int [out, in], lo-nibble first
    lo = (packed_u8 & 0xF).astype(np.int16); lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((packed_u8 >> 4) & 0xF).astype(np.int16); hi = np.where(hi >= 8, hi - 16, hi)
    o = np.empty((packed_u8.shape[0], packed_u8.shape[1] * 2), dtype=np.int16)
    o[:, 0::2] = lo; o[:, 1::2] = hi
    return o

def pack_gptq(signed_out_in, scale_out_groups):
    # signed_out_in: [out, in] in [-8,7]; -> qweight int32 [in//8, out], scales fp16 [groups, out]
    out, inn = signed_out_in.shape
    q = (signed_out_in + (8 if OFFSET == 8 else 0)).astype(np.uint32) & 0xF   # [out,in]
    qt = q.T.copy()                                # [in, out]
    qt = qt.reshape(inn // 8, 8, out)              # [in/8, 8, out]
    qw = np.zeros((inn // 8, out), dtype=np.uint32)
    for j in range(8):
        qw |= (qt[:, j, :] << (4 * j))
    qweight = torch.from_numpy(qw.astype(np.int32)).contiguous()
    scales = torch.from_numpy(scale_out_groups.T.copy()).to(torch.float16).contiguous()  # [groups,out]
    return qweight, scales

tensors = {}
for e in range(NE_TEST):
    base = f"model.layers.{L}.mlp.experts.{e}"
    sh = idx[f"{base}.gate_proj.weight"]
    with safe_open(f"{W4}/{sh}", framework="pt") as f:
        for p in ("gate", "up", "down"):
            pk = f.get_tensor(f"{base}.{p}_proj.weight").numpy().astype(np.uint8)
            sc = f.get_tensor(f"{base}.{p}_proj.weight_scale_inv").float().numpy()  # [out, groups]
            sgn = w4_signed(pk)  # [out, in]
            qw, scl = pack_gptq(sgn, sc)
            tensors[f"{base}.{p}_proj.qweight"] = qw
            tensors[f"{base}.{p}_proj.scales"] = scl
save_file(tensors, f"{OUT}/model.safetensors")
json.dump({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "deepseek_v3",
           "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128,
                                   "sym": True, "desc_act": False}},
          open(f"{OUT}/config.json", "w"))
print(f"repacked {NE_TEST} experts (OFFSET={OFFSET}) -> {OUT}")
print("gate qweight", tensors[f'model.layers.{L}.mlp.experts.0.gate_proj.qweight'].shape,
      "scales", tensors[f'model.layers.{L}.mlp.experts.0.gate_proj.scales'].shape)

# ---- run kt GPTQ_INT4 kernel ----
from kt_kernel.utils.amx import NativeMoEWrapper
mask = torch.zeros(NE_TEST, dtype=torch.bool)
w = NativeMoEWrapper(layer_idx=L, num_experts=NE_TEST, num_experts_per_tok=K, hidden_size=HID,
    moe_intermediate_size=MOE, gpu_experts_mask=mask, cpuinfer_threads=32, threadpool_count=1,
    weight_path=OUT, chunked_prefill_size=2048, method="GPTQ_INT4", numa_nodes=[0])
w.load_weights(torch.arange(NE_TEST, dtype=torch.long))
torch.manual_seed(0)
x = (torch.randn(1, HID, dtype=torch.bfloat16, device="cuda") * 0.1)
ids = torch.arange(K, dtype=torch.long, device="cuda").view(1, K)
tw = torch.full((1, K), 1.0 / K, dtype=torch.float32, device="cuda")
out = w.forward(x, ids, tw, torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
out = out.float().cpu()[0]

# ---- reference ----
def dq(pk, sc):
    s = w4_signed(pk).astype(np.float32)
    return s * np.repeat(sc, 128, axis=1)[:, :s.shape[1]]
xf = x.float().cpu().numpy()[0]; ref = np.zeros(HID, np.float32)
for e in range(K):
    base = f"model.layers.{L}.mlp.experts.{e}"; sh = idx[f"{base}.gate_proj.weight"]
    with safe_open(f"{W4}/{sh}", framework="pt") as f:
        g = dq(f.get_tensor(f"{base}.gate_proj.weight").numpy().astype(np.uint8), f.get_tensor(f"{base}.gate_proj.weight_scale_inv").float().numpy())
        u = dq(f.get_tensor(f"{base}.up_proj.weight").numpy().astype(np.uint8), f.get_tensor(f"{base}.up_proj.weight_scale_inv").float().numpy())
        d = dq(f.get_tensor(f"{base}.down_proj.weight").numpy().astype(np.uint8), f.get_tensor(f"{base}.down_proj.weight_scale_inv").float().numpy())
    gx = g @ xf; ux = u @ xf; act = (gx / (1 + np.exp(-gx))) * ux; ref += (1 / K) * (d @ act)
ref = torch.from_numpy(ref)
print(f"\n==> cos(kernel,ref)={torch.nn.functional.cosine_similarity(out,ref,dim=0).item():+.5f}")
print("kernel nan", torch.isnan(out).any().item(), "norm", out.norm().item(), "ref norm", ref.norm().item())
print("kernel[:5]", out[:5].tolist()); print("ref[:5]   ", ref[:5].tolist())
