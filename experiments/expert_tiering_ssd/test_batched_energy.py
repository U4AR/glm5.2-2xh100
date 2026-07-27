import os, sys, torch
os.environ["KT_ENERGY"]="1"
sys.path.insert(0,".venv/lib/python3.12/site-packages")
from sglang.srt.layers.moe.kt_ep_wrapper import _kt_energy_of, _kt_energy_all
E,L,NG=256,75,96
class M:
    def __init__(s,seed):
        g=torch.Generator().manual_seed(seed)
        s._kt_energy_fast=torch.rand(E,generator=g)
        s._kt_energy_slow=torch.rand(E,generator=g)*10
        s.num_gpu_experts=NG
ms=[M(i) for i in range(L)]
per=torch.stack([_kt_energy_of(m,num_gpu=NG) for m in ms]).float()
bat=_kt_energy_all(ms)
d=(per-bat).abs().max().item()
print("max abs diff batched vs per-layer:", d)
assert d < 1e-5, "batched energy diverges from the per-layer reference"
print("ok  batched energy is numerically identical to the per-layer path")
