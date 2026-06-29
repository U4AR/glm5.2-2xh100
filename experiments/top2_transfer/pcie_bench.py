import torch, time

dev = torch.device('cuda:0')
# One GLM-5.2 int4 expert: gate+up+down = 3 * hidden(6144) * inter(2048) elems @ 0.5 byte (int4)
HIDDEN, INTER = 6144, 2048
elems = 3 * HIDDEN * INTER
expert_bytes = elems // 2  # int4 -> 0.5 byte/elem
print(f"int4 expert: {elems/1e6:.1f}M elems, {expert_bytes/1e6:.2f} MB")

def bench_h2d(nbytes, pinned, iters=50):
    cpu = torch.empty(nbytes, dtype=torch.uint8,
                      pin_memory=pinned, device='cpu')
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    # warmup
    for _ in range(5):
        gpu.copy_(cpu, non_blocking=pinned)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        gpu.copy_(cpu, non_blocking=pinned)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, nbytes / dt / 1e9  # s, GB/s

print("\n== Raw H2D bandwidth (pageable vs pinned) ==")
for mb in [1, 4, 16, 64, 256]:
    nb = mb * 1024 * 1024
    for pin in (False, True):
        dt, bw = bench_h2d(nb, pin)
        print(f"  {mb:4d} MB  pinned={pin!s:5}  {dt*1e3:7.3f} ms  {bw:6.1f} GB/s")

print("\n== One int4 expert (18.9MB) transfer latency ==")
dt, bw = bench_h2d(expert_bytes, True)
print(f"  pinned: {dt*1e3:.3f} ms  ({bw:.1f} GB/s)")
dt, bw = bench_h2d(expert_bytes, False)
print(f"  pageable: {dt*1e3:.3f} ms  ({bw:.1f} GB/s)")

print("\n== Per-token cost model (75 MoE layers) ==")
dt_exp, _ = bench_h2d(expert_bytes, True)
for experts_per_layer in [1.25, 2.0]:
    per_tok = dt_exp * experts_per_layer * 75
    print(f"  {experts_per_layer} experts/layer x75 = {per_tok*1e3:6.1f} ms/token  "
          f"-> ceiling {1/per_tok:5.1f} tok/s (transfer-only, serial)")

print("\n== Can transfer OVERLAP compute? (separate stream + concurrent GEMM) ==")
# Simulate: while transferring 2 experts, run a GPU MoE-ish GEMM on default stream
cpu = torch.empty(expert_bytes*2, dtype=torch.uint8, pin_memory=True)
gpu_dst = torch.empty(expert_bytes*2, dtype=torch.uint8, device=dev)
a = torch.randn(2048, 6144, device=dev, dtype=torch.bfloat16)
b = torch.randn(6144, 2048, device=dev, dtype=torch.bfloat16)
copy_stream = torch.cuda.Stream()
# compute-only baseline
for _ in range(5): _ = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50): _ = a @ b
torch.cuda.synchronize()
comp = (time.perf_counter()-t0)/50
# transfer-only baseline
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(50): gpu_dst.copy_(cpu, non_blocking=True)
torch.cuda.synchronize()
xfer=(time.perf_counter()-t0)/50
# overlapped
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(50):
    with torch.cuda.stream(copy_stream):
        gpu_dst.copy_(cpu, non_blocking=True)
    _ = a @ b
torch.cuda.synchronize()
both=(time.perf_counter()-t0)/50
print(f"  compute(2048x6144x2048 GEMM): {comp*1e3:.3f} ms")
print(f"  transfer(2 experts):          {xfer*1e3:.3f} ms")
print(f"  overlapped:                   {both*1e3:.3f} ms  (vs sum {(comp+xfer)*1e3:.3f}, max {max(comp,xfer)*1e3:.3f})")
