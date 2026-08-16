"""verl Megatron actor integration helpers.

This module keeps the Megatron/MCore batch construction and restore helpers.
Production monkey-patching is owned by ``prefix_sharing.setup`` patch sets;
this module no longer exposes standalone patch installer classes.

Both paths share the same core logic (plan -> trim -> layout -> state).

``VerlMCoreBatchAdapter`` is framework-light and testable locally. It turns a
verl-style micro-batch payload into prefix-sharing metadata plus trimmed
inputs/labels/masks, and it assembles restored logprobs after forward.
"""

from __future__ import annotations

from typing import Any

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.parallel_info import get_megatron_parallel_info
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_utils import collect_kept_position_rows
from prefix_sharing.integrations.verl_utils import extract_seq_from_dense_tensor
from prefix_sharing.integrations.verl_utils import extract_seq_from_nested_tensor
from prefix_sharing.integrations.verl_utils import is_nested_tensor
from prefix_sharing.integrations.verl_utils import trim_redundant_prefix_in_nested_tensor
from prefix_sharing.integrations.verl_utils import trim_plain_batch_thd
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config


def build_prefix_sharing_micro_batch_verl080(
    engine_self: Any,
    batch: Any,
    ps_config: PrefixSharingConfig,
) -> tuple[Any, PrefixSharingRuntimeState | None]:
    """verl 0.8.0 engine 架构下的 prefix-sharing micro-batch 构建。

    MCore/THD 路径：NestedTensor 裁剪后按 kept 区段展开为 packed layout，
    2D 路径：物理裁剪 input_ids/position_ids，通过 attention_mask 标记 valid。

    参数 ps_config 已由调用方通过 PrefixSharingConfig.from_raw() 解析完成，
    不需要再次 from_raw。

    核心原则：2D + attention_mask 为主路径，NestedTensor 路径仅在
    GPU + use_remove_padding=True 时作为可选优化。
    NPU 不支持 torch.nested，所有 NPU 场景都走 2D 路径。
    """
    # ── PATH 1: prefix sharing disabled ──
    if not ps_config.enable_prefix_sharing:
        print("[PS][prepare] PATH 1: prefix sharing disabled")
        return batch, None

    # ── 阶段 1: 配置校验 ──
    use_remove_padding = getattr(engine_self.engine_config, "use_remove_padding", True)
    ps_config.validate_for_engine(use_remove_padding=use_remove_padding)

    # ── 阶段 2: 拒绝不支持的特性 ──
    try:
        from verl.utils import tensordict_utils as tu
        use_fused = tu.get_non_tensor_data(batch, "use_fused_kernels", default=False)
    except Exception:
        use_fused = False
    if use_fused:
        raise RuntimeError("prefix sharing phase 1 requires fused kernels disabled")
    if getattr(engine_self.engine_config, "dynamic_context_parallel", False):
        raise RuntimeError("prefix sharing phase 1 does not support dynamic context parallel")

    # ── 阶段 3: 从 batch 提取序列 ──
    # NestedTensor → 从 offsets/values 提取；kept_position_rows 由 trim 直接返回。
    # Plain 2D → 从 attention_mask.nonzero() 提取，并保留 valid_indices
    # 供 trim 后的 collect_kept_position_rows 使用。
    input_ids = batch["input_ids"]
    is_nested_input = is_nested_tensor(input_ids)
    attention_mask_bool_for_layout = None

    if is_nested_input:
        sequences = extract_seq_from_nested_tensor(input_ids)
    else:
        # plain 2D tensor（需要 attention_mask）
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            print("[PS][prepare] PATH 4: plain 2D batch without attention_mask")
            return batch, None
        attention_mask_bool = attention_mask.to(bool)
        valid_indices = [
            attention_mask_bool[row].nonzero(as_tuple=False).flatten()
            for row in range(input_ids.shape[0])
        ]
        sequences = extract_seq_from_dense_tensor(input_ids, valid_indices)

    # ── 阶段 4: 前缀共享规划 ──
    plan = PrefixSharingPlanner(ps_config).plan(sequences)
    if not plan.has_sharing:
        print("[PS][prepare] no prefix sharing detected")
        return batch, None

    # ── 阶段 5: 物理裁剪 batch ──
    #   NestedTensor path: 裁剪 input_ids/position_ids/loss_mask 以匹配 kept 区段。
    #   2D path: 只改 attention_mask（Megatron 从 mask 动态重算 packed），
    #   v080 THD 路径用 preprocess_thd_engine(input_ids) 直接处理数据，
    #   不看 attention_mask。必须物理裁剪 input_ids/position_ids。
    if is_nested_input:
        trimmed_batch, kept_position_rows = trim_redundant_prefix_in_nested_tensor(batch, plan)
    else:
        trimmed_batch = trim_plain_batch_thd(batch, plan, valid_indices)
        kept_position_rows = collect_kept_position_rows(
            trimmed_batch, plan, is_nested_input,
            valid_indices=valid_indices,
        )

    # ── 阶段 6: 构建 layout ──
    parallel_info = get_megatron_parallel_info()
    align_size = (
        parallel_info.tp_size * parallel_info.cp_size * 2
        if parallel_info.cp_size > 1
        else parallel_info.tp_size
    )
    packed_layout = PackedBatchLayout.from_kept_position_rows(
        kept_position_rows,
        align_size=int(align_size),
    )

    # ── 阶段 7: 构建 state ──
    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=get_backend_instance(ps_config),
        packed_batch_layout=packed_layout,
        parallel_info=parallel_info,
    )

    print(
        f"[PS][prepare] PATH 6: sharing detected, plan={plan}, layout={packed_layout}"
    )

    return trimmed_batch, state

