from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)
_LOGGED_W4AFP8_GPU_BACKENDS: set[str] = set()


class W4AFp8Config(QuantizationConfig):
    """Config class for MIXED_PRECISION W4AFp8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        is_checkpoint_w4afp8_serialized: bool = True,
        linear_activation_scheme: str = "dynamic",
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized
        if is_checkpoint_w4afp8_serialized:
            logger.warning("Detected w4afp8 checkpoint. Please note that")
        if moe_activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.linear_activation_scheme = linear_activation_scheme
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = [128, 128]
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM80+ can run the W4A16 Marlin fallback.  W4A8 CUTLASS remains the
        # default on Hopper (SM90+) and is selected in get_quant_method().
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4AFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method
        linear_activation_scheme = "dynamic"
        moe_activation_scheme = "static"
        weight_block_size = [128, 128]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            backend = os.getenv("KT_W4AFP8_GPU_BACKEND", "auto").strip().lower()
            aliases = {
                "cutlass": "cutlass_sm90",
                "sm90": "cutlass_sm90",
                "marlin": "marlin_sm80",
                "sm89": "marlin_sm80",
            }
            backend = aliases.get(backend, backend)
            if backend not in ("auto", "cutlass_sm90", "marlin_sm80"):
                raise ValueError(
                    "KT_W4AFP8_GPU_BACKEND must be auto, cutlass_sm90, or "
                    f"marlin_sm80; got {backend!r}"
                )
            capability = (
                torch.cuda.get_device_capability()
                if torch.cuda.is_available()
                else (0, 0)
            )
            if backend == "auto":
                backend = "cutlass_sm90" if capability[0] >= 9 else "marlin_sm80"
            if backend == "cutlass_sm90" and capability[0] < 9:
                raise RuntimeError(
                    "W4AFP8 CUTLASS requires SM90+; current capability is "
                    f"SM{capability[0]}{capability[1]}. Use "
                    "KT_W4AFP8_GPU_BACKEND=marlin_sm80 on Ampere/Ada."
                )
            if backend == "marlin_sm80":
                if capability[0] < 8:
                    raise RuntimeError(
                        "W4AFP8 Marlin requires SM80+; current capability is "
                        f"SM{capability[0]}{capability[1]}."
                    )
                if backend not in _LOGGED_W4AFP8_GPU_BACKENDS:
                    logger.info(
                        "W4AFP8 GPU backend: Marlin W4A16 (SM%d%d)",
                        capability[0],
                        capability[1],
                    )
                    _LOGGED_W4AFP8_GPU_BACKENDS.add(backend)
                return W4AFp8MarlinMoEMethod(self)
            if backend not in _LOGGED_W4AFP8_GPU_BACKENDS:
                logger.info(
                    "W4AFP8 GPU backend: CUTLASS W4A8 (SM%d%d)",
                    capability[0],
                    capability[1],
                )
                _LOGGED_W4AFP8_GPU_BACKENDS.add(backend)
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
    s_shape = scales.shape
    # Reshape to separate groups of 4
    alignment = 4 if s_shape[2] % 4 == 0 else 1
    scales_interleaved = scales.reshape(
        s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment
    )
    # Permute dimensions to interleave
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
    # Reshape back to original dimensions but with interleaved values
    scales_interleaved = scales_interleaved.reshape(
        s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment
    )
    return scales_interleaved.contiguous()


class W4AFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: W4AFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # kt-kernel's full-GPU-prefill path (kt_ep_wrapper SharedFullContext) calls
        # create_weights to build scratch GPU expert tensors and populates them via
        # .copy_() from its own CPU store, NOT via the model weight_loader. Tolerate a
        # missing loader there with a no-op; the normal model-load path always passes
        # a real weight_loader so this assert is preserved for that case.
        if "weight_loader" not in extra_weight_attrs:
            extra_weight_attrs["weight_loader"] = lambda *a, **k: None

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Input scales
        w13_input_scale = torch.nn.Parameter(
            torch.ones((num_experts, 2), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
        device = layer.w13_weight.device

        self.a_strides1 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

        return

    def process_weights_after_loading(self, layer: Module) -> None:
        dtype = torch.bfloat16
        device = layer.w2_weight.device

        # Idempotency guard: interleave_scales is shape-changing and NOT safe to
        # run twice. In the kt_ep_wrapper integration this can be invoked more
        # than once (KTEPWrapperMethod.process_weights_after_loading + Shared
        # FullContext paths); guard so it interleaves exactly once per layer.
        if getattr(layer, "_w4afp8_scales_interleaved", False):
            return

        # Interleave w13_weight_scale (gate_up_proj)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        w13_weight_scale = interleave_scales(w13_weight_scale)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_weight_scale = interleave_scales(w2_weight_scale)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)

        # Process input scales
        w13_input_scale_max = layer.w13_input_scale.max().to(torch.float32).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],
            dtype=torch.float32,
            device=device,
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        w2_input_scale_max = layer.w2_input_scale.max().to(torch.float32).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max], dtype=torch.float32, device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)
        layer._w4afp8_scales_interleaved = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # kt_ep_wrapper marks non-GPU (CPU-resident) experts with -1 in topk_ids.
        # cutlass_w4a8_moe only remaps -1 -> num_local_experts (its grouped-gemm
        # "skip" sentinel) when EP world size > 1; at ep_size=1 the -1 flows into
        # the reorder/preprocess kernels as an invalid index and corrupts the
        # output (NaN -> poisons the residual stream -> garbage tokens). Remap
        # here so masked experts are correctly skipped regardless of EP size.
        # No-op when there are no -1 entries (e.g. pure-GPU Phala path).
        num_local_experts = layer.w13_weight.shape[0]
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, num_local_experts),
            topk_ids,
        )

        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_ll(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPLLDispatchOutput,
    ) -> torch.Tensor:

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll

        hidden_states, _, topk_ids, _, masked_m, _ = dispatch_output

        output = cutlass_w4a8_moe_deepep_ll(
            hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_ids,
            masked_m,
            layer.quant_method.a_strides1,
            layer.quant_method.b_strides1,
            layer.quant_method.c_strides1,
            layer.quant_method.a_strides2,
            layer.quant_method.b_strides2,
            layer.quant_method.c_strides2,
            layer.quant_method.s_strides13,
            layer.quant_method.s_strides2,
            layer.quant_method.expert_offsets,
            layer.quant_method.problem_sizes1,
            layer.quant_method.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
        )

        return output

    def apply_deepep_normal(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe_deepep_normal,
        )

        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        num_tokens = hidden_states.shape[0]
        if num_tokens > 0:
            return cutlass_w4a8_moe_deepep_normal(
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk_weights,
                topk_idx,
                self.a_strides1,
                self.b_strides1,
                self.c_strides1,
                self.a_strides2,
                self.b_strides2,
                self.c_strides2,
                self.s_strides13,
                self.s_strides2,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                layer.w13_input_scale,
                layer.w2_input_scale,
            )
        else:
            return hidden_states


def _raw_twos_complement_int4_to_gptq(
    packed_weight: torch.Tensor,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """Convert GLM RAWINT4 [E, N, K/2] bytes to GPTQ [E, K/8, N].

    GLM stores signed two's-complement nibbles. Marlin's symmetric uint4b8
    scalar type decodes an unsigned nibble as ``nibble - 8``. XORing bit 3
    maps exactly between those conventions while preserving nibble order.
    """
    if packed_weight.dtype != torch.int8:
        raise TypeError(f"expected int8 packed weights, got {packed_weight.dtype}")
    if tuple(packed_weight.shape[1:]) != (size_n, size_k // 2):
        raise ValueError(
            "unexpected RAWINT4 shape "
            f"{tuple(packed_weight.shape)} for K={size_k}, N={size_n}"
        )
    if size_k % 8:
        raise ValueError(f"Marlin INT4 K must be divisible by 8, got {size_k}")

    num_experts = packed_weight.shape[0]
    biased_bytes = packed_weight.view(torch.uint8) ^ 0x88
    # Four adjacent RAWINT4 bytes contain eight consecutive K values. Make
    # those four bytes the little-endian payload of one GPTQ int32.
    return (
        biased_bytes.reshape(num_experts, size_n, size_k // 8, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.int32)
        .squeeze(-1)
    )


class W4AFp8MarlinMoEMethod(W4AFp8MoEMethod):
    """Portable SM80+ GPU backend for GLM W4AFP8 routed experts.

    Weights remain packed INT4 in VRAM. Activations are BF16 (W4A16) because
    the existing W4A8 grouped GEMM relies on Hopper TMA and cannot initialize
    on Ada SM89. The validated SM90 CUTLASS class above is deliberately left
    unchanged and remains the automatic Hopper selection.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_w4afp8_marlin_converted", False):
            return

        from sglang.srt.layers.quantization.gptq import (
            gptq_marlin_moe_repack,
        )
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_moe_permute_scales,
        )

        group_size = self.quant_config.group_size
        if group_size != 128:
            raise ValueError(
                f"GLM W4AFP8 Marlin requires group_size=128, got {group_size}"
            )

        num_experts = layer.w13_weight.shape[0]
        w13_n = layer.w13_weight.shape[1]
        w13_k = layer.w13_weight.shape[2] * 2
        w2_n = layer.w2_weight.shape[1]
        w2_k = layer.w2_weight.shape[2] * 2
        device = layer.w13_weight.device
        empty_perm = torch.empty(
            (num_experts, 0), dtype=torch.int32, device=device
        )

        w13_gptq = _raw_twos_complement_int4_to_gptq(
            layer.w13_weight, w13_k, w13_n
        )
        w2_gptq = _raw_twos_complement_int4_to_gptq(
            layer.w2_weight, w2_k, w2_n
        )
        w13_marlin = gptq_marlin_moe_repack(
            w13_gptq, empty_perm, w13_k, w13_n, 4
        )
        w2_marlin = gptq_marlin_moe_repack(
            w2_gptq, empty_perm, w2_k, w2_n, 4
        )

        # Checkpoint scales are [E, N, K/group]. Marlin first expects
        # [E, K/group, N], followed by its lane permutation.
        w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale_inv.transpose(1, 2)
            .contiguous()
            .to(torch.bfloat16),
            w13_k,
            w13_n,
            group_size,
        )
        w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale_inv.transpose(1, 2)
            .contiguous()
            .to(torch.bfloat16),
            w2_k,
            w2_n,
            group_size,
        )

        layer.w13_weight = Parameter(w13_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_scales, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scales, requires_grad=False)
        layer.w13_weight_g_idx = Parameter(empty_perm, requires_grad=False)
        layer.w2_weight_g_idx = Parameter(empty_perm.clone(), requires_grad=False)
        layer.w13_g_idx_sort_indices = Parameter(
            empty_perm.clone(), requires_grad=False
        )
        layer.w2_g_idx_sort_indices = Parameter(
            empty_perm.clone(), requires_grad=False
        )
        layer._w4afp8_marlin_converted = True
        # SharedFullContext uses this explicit marker when it restores fresh raw
        # parameters for another layer.
        layer._w4afp8_gpu_backend = "marlin_sm80"

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self.moe_runner_config.activation != "silu":
            raise ValueError("W4AFP8 Marlin supports only SiLU activation")

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, router_logits = dispatch_output.topk_output

        # KT marks CPU-routed slots as -1. Keep the Marlin call valid and make
        # those slots contribute exactly zero, independent of SGLang version.
        num_local_experts = layer.w13_weight.shape[0]
        invalid = (topk_ids < 0) | (topk_ids >= num_local_experts)
        topk_ids = torch.where(invalid, torch.zeros_like(topk_ids), topk_ids)
        topk_weights = torch.where(
            invalid, torch.zeros_like(topk_weights), topk_weights
        )

        output = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            gating_output=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=4,
            is_k_full=True,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)
