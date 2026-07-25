#!/usr/bin/env python3
"""Numerical test for the portable SM80+ GLM W4AFP8 Marlin MoE backend."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("KT_W4AFP8_GPU_BACKEND", "marlin_sm80")

import torch

from sglang.srt.server_args import set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(
    SimpleNamespace(enable_deterministic_inference=False)
)

from sglang.srt.layers.quantization.w4afp8 import (  # noqa: E402
    W4AFp8Config,
    W4AFp8MarlinMoEMethod,
    _raw_twos_complement_int4_to_gptq,
)


def pack_twos_complement(q: torch.Tensor) -> torch.Tensor:
    lo = (q[..., 0::2] & 0x0F).to(torch.uint8)
    hi = (q[..., 1::2] & 0x0F).to(torch.uint8) << 4
    return (lo | hi).view(torch.int8)


def expanded_scales(scales: torch.Tensor, group_size: int) -> torch.Tensor:
    return scales.repeat_interleave(group_size, dim=-1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        raise RuntimeError(f"Marlin requires SM80+, found SM{capability[0]}{capability[1]}")

    torch.manual_seed(11)
    device = torch.device("cuda:0")
    num_experts, num_tokens = 2, 6
    hidden_size = intermediate_size = 256
    group_size = 128

    # First validate the convention and packing order without invoking Marlin.
    boundary = torch.tensor(
        [[[-8, 7, -1, 0, 1, -7, 6, -2]]], dtype=torch.int8
    )
    gptq = _raw_twos_complement_int4_to_gptq(
        pack_twos_complement(boundary), size_k=8, size_n=1
    )
    biased_nibbles = []
    for byte in gptq.view(torch.uint8).flatten().tolist():
        biased_nibbles.extend((byte & 0x0F, byte >> 4))
    expected = (boundary.flatten().to(torch.int16) + 8).tolist()
    assert biased_nibbles == expected, (biased_nibbles, expected)

    method = W4AFp8MarlinMoEMethod(W4AFp8Config(group_size=group_size))
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    for name, parameter in list(layer.named_parameters()):
        setattr(
            layer,
            name,
            torch.nn.Parameter(parameter.to(device), requires_grad=False),
        )

    q13 = torch.randint(
        -8,
        8,
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.int8,
    )
    q2 = torch.randint(
        -8,
        8,
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.int8,
    )
    s13 = (
        torch.rand(
            (
                num_experts,
                2 * intermediate_size,
                hidden_size // group_size,
            ),
            device=device,
        )
        * 0.015
        + 0.002
    ).to(torch.float32)
    s2 = (
        torch.rand(
            (
                num_experts,
                hidden_size,
                intermediate_size // group_size,
            ),
            device=device,
        )
        * 0.015
        + 0.002
    ).to(torch.float32)

    layer.w13_weight.data.copy_(pack_twos_complement(q13))
    layer.w2_weight.data.copy_(pack_twos_complement(q2))
    layer.w13_weight_scale_inv.data.copy_(s13)
    layer.w2_weight_scale_inv.data.copy_(s2)
    method.process_weights_after_loading(layer)
    method.create_moe_runner(
        layer,
        SimpleNamespace(activation="silu", routed_scaling_factor=None),
    )

    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )
    topk_ids = torch.tensor([[0], [1], [0], [1], [-1], [0]], device=device)
    topk_weights = torch.tensor(
        [[1.0], [1.0], [0.75], [0.5], [0.9], [0.25]], device=device
    )
    router_logits = torch.zeros(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    dispatch = SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=(topk_weights, topk_ids, router_logits),
    )
    output = method.apply(layer, dispatch).hidden_states

    reference = []
    for token, expert in enumerate(topk_ids[:, 0].tolist()):
        if expert < 0:
            reference.append(torch.zeros(hidden_size, device=device))
            continue
        w13 = q13[expert].float() * expanded_scales(
            s13[expert], group_size
        )
        w2 = q2[expert].float() * expanded_scales(s2[expert], group_size)
        gate_up = hidden_states[token].float() @ w13.T
        activated = (
            torch.nn.functional.silu(gate_up[:intermediate_size])
            * gate_up[intermediate_size:]
        )
        reference.append(
            (activated @ w2.T) * topk_weights[token, 0]
        )
    reference_tensor = torch.stack(reference)

    valid = topk_ids[:, 0] >= 0
    cosine = torch.nn.functional.cosine_similarity(
        output[valid].float().reshape(1, -1),
        reference_tensor[valid].reshape(1, -1),
    ).item()
    assert torch.isfinite(output).all()
    assert output[~valid].abs().max().item() == 0.0
    assert cosine >= 0.999, cosine
    print(
        f"SM{capability[0]}{capability[1]} W4AFP8 Marlin: "
        f"cos={cosine:.7f}, finite=True, invalid_route_zero=True"
    )


if __name__ == "__main__":
    main()
