import os, json, numpy as np, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from safetensors import safe_open
import kt_kernel.experts_base as eb
from kt_kernel.utils.amx import NativeMoEWrapper

W4 = "/cache/nvme0/models/GLM-5.2-W4AFP8"
L, HID, MOE, NE, K = 3, 6144, 2048, 256, 8
ZERO = os.getenv("INPUT_ZERO", "0") == "1"
NTOK = int(os.getenv("NTOK", "1"))

mask = torch.zeros(NE, dtype=torch.bool)
w = NativeMoEWrapper(
    layer_idx=L, num_experts=NE, num_experts_per_tok=K,
    hidden_size=HID, moe_intermediate_size=MOE,
    gpu_experts_mask=mask, cpuinfer_threads=64, threadpool_count=2,
    weight_path=W4, chunked_prefill_size=2048, method="RAWINT4", numa_nodes=[0, 1],
)
w.load_weights(torch.arange(NE, dtype=torch.long))

torch.manual_seed(0)
if ZERO:
    x = torch.zeros(NTOK, HID, dtype=torch.bfloat16, device="cuda")
else:
    x = (torch.randn(NTOK, HID, dtype=torch.bfloat16, device="cuda") * 0.1)
topk_ids = torch.arange(K, dtype=torch.long, device="cuda").view(1, K).repeat(NTOK, 1)
topk_w = torch.full((NTOK, K), 1.0 / K, dtype=torch.float32, device="cuda")
stream = torch.cuda.current_stream().cuda_stream
out_gpu = w.forward(x, topk_ids, topk_w, stream)
torch.cuda.synchronize()

# read raw CPU output buffer too
buf = eb.KExpertsCPUBuffer.temp_buffer
output_cpu = buf[4][L % eb.KExpertsCPUBuffer.buffer_depth]
oc = output_cpu.float()
og = out_gpu.float().cpu()
print(f"\nINPUT_ZERO={ZERO} NTOK={NTOK}")
print(f"output_cpu: nan={torch.isnan(oc).any().item()} inf={torch.isinf(oc).any().item()} "
      f"norm={oc.norm().item():.4f} min={oc.min().item():.3e} max={oc.max().item():.3e} nan_frac={torch.isnan(oc).float().mean().item():.3f}")
print(f"output_gpu: nan={torch.isnan(og).any().item()} norm={og.norm().item():.4f}")
print("output_cpu[0,:8]", oc[0, :8].tolist())
