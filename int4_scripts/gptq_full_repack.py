"""
Full repack: W4AFP8 routed experts -> kt GPTQ_INT4 format, one .safetensors shard
per MoE layer, into $GPTQ_EXPERTS_DIR (default ./weights/GLM-5.2-W4-GPTQ-experts).
Parallel over layers.
Validated convention (cos 1.0): AutoGPTQ qweight int32 [in/8,out] 8x4bit lo-first
along K, q=signed+8; scales [groups,out] fp16; group-128 symmetric.
Non-expert tensors are NOT repacked (sglang reads them from --model-path=W4AFP8).
"""
import os, json, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from concurrent.futures import ProcessPoolExecutor, as_completed
from _paths import W4, GPTQ_EXPERTS_DIR as OUT  # repo-relative; see int4_scripts/_paths.py

os.makedirs(OUT, exist_ok=True)
IDX = json.load(open(f"{W4}/model.safetensors.index.json"))["weight_map"]

def moe_layers():
    import re
    ls = set()
    for k in IDX:
        m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.0\.gate_proj\.weight$", k)
        if m:
            ls.add(int(m.group(1)))
    return sorted(ls)

def n_experts(L):
    e = 0
    while f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight" in IDX:
        e += 1
    return e

def w4_signed(pk):
    lo = (pk & 0xF).astype(np.int16); lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((pk >> 4) & 0xF).astype(np.int16); hi = np.where(hi >= 8, hi - 16, hi)
    o = np.empty((pk.shape[0], pk.shape[1] * 2), dtype=np.int16)
    o[:, 0::2] = lo; o[:, 1::2] = hi
    return o

def pack_gptq(sgn, sc):
    out, inn = sgn.shape
    q = (sgn + 8).astype(np.uint32) & 0xF
    qt = q.T.reshape(inn // 8, 8, out)
    qw = np.zeros((inn // 8, out), dtype=np.uint32)
    for j in range(8):
        qw |= (qt[:, j, :] << (4 * j))
    return torch.from_numpy(qw.astype(np.int32)).contiguous(), torch.from_numpy(sc.T.copy()).to(torch.float16).contiguous()

def repack_layer(L):
    ne = n_experts(L)
    # group W4 tensors by shard to minimize file opens
    tensors = {}
    handles = {}
    def get(name):
        sh = IDX[name]
        if sh not in handles:
            handles[sh] = safe_open(f"{W4}/{sh}", framework="pt")
        return handles[sh].get_tensor(name)
    for e in range(ne):
        base = f"model.layers.{L}.mlp.experts.{e}"
        for p in ("gate", "up", "down"):
            pk = get(f"{base}.{p}_proj.weight").numpy().astype(np.uint8)
            sc = get(f"{base}.{p}_proj.weight_scale_inv").float().numpy()
            qw, scl = pack_gptq(w4_signed(pk), sc)
            tensors[f"{base}.{p}_proj.qweight"] = qw
            tensors[f"{base}.{p}_proj.scales"] = scl
    save_file(tensors, f"{OUT}/experts-layer-{L:03d}.safetensors")
    return L, ne

if __name__ == "__main__":
    json.dump({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "deepseek_v3",
               "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128,
                                       "sym": True, "desc_act": False}},
              open(f"{OUT}/config.json", "w"))
    layers = moe_layers()
    print(f"MoE layers: {len(layers)} -> {layers[0]}..{layers[-1]}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(repack_layer, L): L for L in layers}
        for f in as_completed(futs):
            L, ne = f.result(); done += 1
            print(f"[{done}/{len(layers)}] layer {L}: {ne} experts repacked", flush=True)
    print("REPACK COMPLETE", flush=True)
