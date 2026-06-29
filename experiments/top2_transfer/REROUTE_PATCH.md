# Reroute patch (applied to .venv copies during the experiment)
## sglang/srt/models/glm4_moe.py — added before class Glm4MoeSparseMoeBlock:
```python
# [EXPERIMENT top2-expert-transfer] env-gated GPU-resident reroute.
#   KT_REROUTE_GPU=1 turns it on. KT_REROUTE_KEEP=K keeps the top-K experts (by
#   router logit, on ANY device) and forces the remaining top_k-K slots to be
#   filled from GPU-resident experts only (logits of CPU-resident non-kept experts
#   are masked to -inf before topk). K=2 = user's "keep top-2, substitute tail-6
#   from GPU"; K=0 = pure GPU-only routing (CPU experts never selected). The kt
#   path already computes any kept CPU-resident experts in place, so this needs NO
#   weight transfer — it just shrinks the CPU critical path from ~5 to <=K experts.
_KT_REROUTE_GPU = os.environ.get("KT_REROUTE_GPU") == "1"
_KT_REROUTE_KEEP = int(os.environ.get("KT_REROUTE_KEEP", "2"))


def _kt_reroute_to_gpu(router_logits, quant_method):
    """Mask router logits so all but the top-K selected experts come from the
    GPU-resident set. Shape-static (CUDA-graph safe). No-op if mask unavailable."""
    mask = getattr(quant_method, "gpu_experts_mask_cuda", None)
    if mask is None:
        return router_logits
    allow = mask.to(torch.bool).view(1, -1).expand_as(router_logits).clone()
    if _KT_REROUTE_KEEP > 0:
        keep_idx = torch.topk(router_logits, _KT_REROUTE_KEEP, dim=-1).indices
        allow.scatter_(1, keep_idx, True)
    neg = torch.finfo(router_logits.dtype).min
    return torch.where(allow, router_logits, neg)
```
## call sites: after each `router_logits = self.gate(hidden_states)` in forward_normal[_dual_stream]:
```python
            if _KT_REROUTE_GPU:
                router_logits = _kt_reroute_to_gpu(
                    router_logits, getattr(self.experts, "quant_method", None))
```
## kt_ep_wrapper.py — KT_DUMP_TOPK trace tap before mask_and_remap_expert_ids (eager only)
