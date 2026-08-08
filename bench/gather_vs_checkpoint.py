#!/usr/bin/env python3
"""Is the CUDA gather producing the bytes cutlass expects? Checkpoint decides.

The behavioural ladder (`bench/gather_truth.sh`) proved the gathered weights are
WRONG -- omitting a landed expert scores better than computing it from the gather
(accept 2.532 vs 2.273), and computing something can only beat computing nothing
if what you computed is right. This names which part is wrong.

The reference is the checkpoint, not kt's staging: those boots run
KT_GPU_PREFILL_THRESHOLD=0, so kt's `write_weights_to_buffer` never runs in
anger and may share the same defect. Comparing two possibly-broken readers is
what got us here.

The dump carries a RESIDENT expert R alongside the gathered non-resident E.
R came out of VRAM via sglang's ordinary weight loader, so reconstructing R from
the checkpoint and matching it byte for byte is what licenses this script to
judge E. If R does not match, the reader is wrong and the verdict on E is
withheld -- that check is not a formality, it is the whole reason one boot is
enough.

Layout being reconstructed (from W4AFp8MoEMethod + interleave_scales):
    w13   [2*moe, hidden/2]   gate rows then up rows, int4-packed
    w2    [hidden, moe/2]
    w13_s [G13/4, 2*moe*4] bf16,  G13 = hidden/group
    w2_s  [G2/4,  hidden*4] bf16,  G2  = moe/group
  interleave: dst[g, n*4 + a] = src[n, g*4 + a]
"""
import argparse, glob, json, os, sys
import torch


def interleave(s: torch.Tensor) -> torch.Tensor:
    """sglang's interleave_scales for a single expert: [N, G] -> [G/4, N*4]."""
    n, g = s.shape
    align = 4 if g % 4 == 0 else 1
    return s.reshape(n, g // align, align).permute(1, 0, 2).reshape(g // align, n * align).contiguous()


def load_ckpt_expert(model_dir, layer, expert):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.layers.{layer}.mlp.experts.{expert}."
    out = {}
    for short, key in (("gate_w", "gate_proj.weight"), ("gate_s", "gate_proj.weight_scale_inv"),
                       ("up_w", "up_proj.weight"), ("up_s", "up_proj.weight_scale_inv"),
                       ("down_w", "down_proj.weight"), ("down_s", "down_proj.weight_scale_inv")):
        full = pre + key
        if full not in idx:
            raise KeyError(f"missing {full}")
        with safe_open(os.path.join(model_dir, idx[full]), framework="pt") as f:
            out[short] = f.get_tensor(full)
    return out


def build_expected(ck, tp_rank, tp_size, moe, hidden, group=128):
    """Reconstruct this rank's cutlass tensors for one expert.

    gate/up are row-sharded along the intermediate dim; down is column-sharded
    along the same axis (its K).
    """
    gw, uw, dw = ck["gate_w"], ck["up_w"], ck["down_w"]
    gs, us, ds = ck["gate_s"], ck["up_s"], ck["down_s"]
    lo, hi = tp_rank * moe, (tp_rank + 1) * moe

    # int4-packed: 2 values per byte along the LAST dim, so the row count is the
    # intermediate dim for gate/up and the shard applies to rows.
    g_w, u_w = gw[lo:hi], uw[lo:hi]
    w13 = torch.cat([g_w, u_w], dim=0)

    # down_proj: [hidden, moe_full] packed along last dim -> shard columns, and
    # the packed byte width is half the element count.
    d_lo, d_hi = lo // 2, hi // 2
    w2 = dw[:, d_lo:d_hi]

    w13_s = interleave(torch.cat([gs[lo:hi], us[lo:hi]], dim=0).to(torch.bfloat16))
    g2_lo, g2_hi = lo // group, hi // group
    w2_s = interleave(ds[:, g2_lo:g2_hi].to(torch.bfloat16))
    return {"w13_weight": w13, "w2_weight": w2,
            "w13_weight_scale_inv": w13_s, "w2_weight_scale_inv": w2_s}


def cmp(tag, got, want):
    if got.shape != want.shape:
        print(f"    {tag:<24} SHAPE {tuple(got.shape)} vs {tuple(want.shape)}")
        return False
    g, w = got.reshape(-1), want.reshape(-1)
    if g.dtype in (torch.bfloat16, torch.float16, torch.float32):
        d = (g.float() - w.float()).abs()
        nbad = int((d > 0).sum())
        print(f"    {tag:<24} {'OK' if nbad==0 else 'MISMATCH'}  "
              f"{nbad}/{g.numel()} differ  max|d|={float(d.max()) if d.numel() else 0:.6g}")
    else:
        gb, wb = g.view(torch.uint8), w.view(torch.uint8)
        ne = gb != wb
        nbad = int(ne.sum())
        extra = ""
        if nbad:
            first = int(torch.nonzero(ne)[0])
            extra = f"  first@{first} got=0x{int(gb[first]):02x} want=0x{int(wb[first]):02x}"
            # Cheap hypotheses, so the verdict names a defect instead of a delta.
            swapped = ((gb >> 4) | (gb << 4))
            if bool((swapped == wb).all()):
                extra += "  [NIBBLES SWAPPED]"
            elif bool(((gb ^ 0x88) == wb).all()):
                extra += "  [SIGN/OFFSET-8 encoding]"
        print(f"    {tag:<24} {'OK' if nbad==0 else 'MISMATCH'}  "
              f"{nbad}/{gb.numel()} bytes differ{extra}")
    return nbad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="bench/profile_out/pf_dump")
    ap.add_argument("--model", default="weights/GLM-5.2-W4AFP8")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dump, "*.pt")))
    if not files:
        print(f"no dumps in {a.dump}")
        return 1
    rc = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        L, tp, tps = d["layer_idx"], d["tp_rank"], d["tp_size"] or 1
        moe, hidden = d["moe"], d["hidden"]
        print(f"\n=== {os.path.basename(f)}  layer={L} tp={tp}/{tps} "
              f"moe={moe} hidden={hidden} R={d['R']} E={d['E']} ===")

        for who, eid in (("resident R", d["R"]), ("gathered E", d["E"])):
            try:
                ck = load_ckpt_expert(a.model, L, eid)
                exp = build_expected(ck, tp, tps, moe, hidden)
            except Exception as e:
                print(f"  {who}: checkpoint read failed: {e}")
                rc = 1
                continue
            src = d["resident"] if who.startswith("resident") else d["gathered"]
            print(f"  {who} (expert {eid}):")
            ok = all(cmp(n, src[n], exp[n]) for n in exp)
            if who.startswith("resident"):
                if ok:
                    print("    -> reader VALIDATED; the verdict on E is meaningful")
                else:
                    print("    -> reader NOT validated (shard/layout assumption wrong).")
                    print("       WITHHOLDING the verdict on E: a mismatch below "
                          "would be this script's bug, not the gather's.")
                    rc = 2
                    break
    return rc


if __name__ == "__main__":
    sys.exit(main())
