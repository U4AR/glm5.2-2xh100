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
CPUINFER = int(os.getenv("CPUINFER", "28"))
KT_THREADPOOL_COUNT = int(os.getenv("KT_THREADPOOL_COUNT", "2"))
MATRIX = os.getenv("KT_TEST_MATRIX") == "1"

# Matrix mode needs both CPU and GPU-marked logical IDs so one wrapper can test
# every CPU route count without reloading the layer.  NativeMoEWrapper computes
# only the false-mask (CPU) routes; the true-mask routes therefore contribute
# exactly zero to this CPU-kernel harness.
mask = torch.zeros(NE, dtype=torch.bool)
if MATRIX:
    mask[NE // 2:] = True
w = NativeMoEWrapper(
    layer_idx=L, num_experts=NE, num_experts_per_tok=K,
    hidden_size=HID, moe_intermediate_size=MOE,
    gpu_experts_mask=mask, cpuinfer_threads=CPUINFER, threadpool_count=KT_THREADPOOL_COUNT,
    weight_path=W4, chunked_prefill_size=2048, method="RAWINT4", numa_nodes=[0, 1],
)
w.load_weights(torch.arange(NE // 2 if MATRIX else NE, dtype=torch.long))


def run_kernel(x, topk_ids, topk_w):
    stream = torch.cuda.current_stream().cuda_stream
    result = w.forward(x, topk_ids, topk_w, stream)
    torch.cuda.synchronize()
    result = result.float().cpu()
    assert torch.isfinite(result).all(), "packed RAWINT4 kernel produced a non-finite value"
    return result


def matrix_input(m, kind, seed):
    if kind == "zero":
        return torch.zeros(m, HID, dtype=torch.bfloat16, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "high":
        # Large finite BF16 values with both signs exercise scale selection and
        # biased-u8 endpoints without introducing infinities.
        signs = torch.randint(0, 2, (m, HID), generator=gen, device="cuda")
        return torch.where(signs == 0, -64.0, 64.0).to(torch.bfloat16)
    return (torch.randn(m, HID, generator=gen, dtype=torch.bfloat16, device="cuda") * 0.1)


def matrix_routes(m, cpu_routes, repeated):
    ids = torch.empty((m, K), dtype=torch.long, device="cuda")
    for row in range(m):
        for slot in range(K):
            if slot < cpu_routes:
                ids[row, slot] = 0 if repeated else (row * K + slot) % (NE // 2)
            else:
                ids[row, slot] = NE // 2 if repeated else NE // 2 + (row * K + slot) % (NE // 2)
    weights = torch.full((m, K), 1.0 / K, dtype=torch.float32, device="cuda")
    return ids, weights


def run_matrix():
    outputs = {}
    # Four cases at every (M, route-count) point jointly cover zero/high inputs,
    # repeated/distinct routes, and two deterministic random seeds.
    variants = (
        ("random", 0, False),
        ("random", 17, True),
        ("zero", 0, False),
        ("high", 17, True),
    )
    if os.getenv("KT_TEST_DISTINCT_ONLY") == "1":
        variants = tuple(v for v in variants if not v[2])
    for m in (1, 2, 4, 8):
        for cpu_routes in (0, 1, 2, 4, 8):
            for kind, seed, repeated in variants:
                key = f"m{m}-cpu{cpu_routes}-{kind}-s{seed}-{'repeat' if repeated else 'distinct'}"
                x_case = matrix_input(m, kind, seed)
                ids_case, weights_case = matrix_routes(m, cpu_routes, repeated)
                outputs[key] = run_kernel(x_case, ids_case, weights_case)
    return outputs

torch.manual_seed(0)
x = (torch.randn(1, HID, dtype=torch.bfloat16, device="cuda") * 0.1)
topk_ids = torch.arange(K, dtype=torch.long, device="cuda").view(1, K)
topk_w = torch.full((1, K), 1.0 / K, dtype=torch.float32, device="cuda")
out = run_kernel(x, topk_ids, topk_w)[0]
matrix_outputs = run_matrix() if MATRIX else None

# Stage-by-stage CPU expert optimization parity.  Save with the legacy binary,
# then compare after each rebuilt runtime.  The tensor is stored before the
# floating-point reference work below so this check covers the actual packed
# RAWINT4 kernel output, not a rounded text sample.
baseline_path = os.getenv("KT_TEST_BASELINE", "")
if baseline_path:
    if os.getenv("KT_TEST_SAVE_BASELINE") == "1":
        payload = {"single": out, "matrix": matrix_outputs} if MATRIX else out
        torch.save(payload, baseline_path)
        print(f"saved bitwise baseline: {baseline_path}")
    else:
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        expected_single = baseline["single"] if MATRIX else baseline
        if not torch.equal(out, expected_single):
            different = torch.count_nonzero(out.view(torch.int32) != expected_single.view(torch.int32)).item()
            max_abs = torch.max(torch.abs(out - expected_single)).item()
            raise AssertionError(f"packed RAWINT4 single output differs: {different}/{out.numel()} values, max_abs={max_abs}")
        if MATRIX:
            expected_matrix = baseline["matrix"]
            assert set(matrix_outputs).issubset(expected_matrix), "matrix baseline case set differs"
            for key, actual in matrix_outputs.items():
                expected = expected_matrix[key]
                if not torch.equal(actual, expected):
                    different = torch.count_nonzero(actual.view(torch.int32) != expected.view(torch.int32)).item()
                    max_abs = torch.max(torch.abs(actual - expected)).item()
                    raise AssertionError(
                        f"packed RAWINT4 matrix case {key} differs: "
                        f"{different}/{actual.numel()} values, max_abs={max_abs}"
                    )
            print(f"bitwise matrix match: {len(matrix_outputs)} cases")
        print(f"bitwise baseline match: {baseline_path}")

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
assert torch.isfinite(out).all(), "packed RAWINT4 kernel produced a non-finite value"
assert cos >= 0.999, f"packed RAWINT4 reference cosine regressed: {cos}"
