"""verl080 prefix-sharing restore 端到端集成测试。

验证 forward_step patch + vocab_logprobs patch + restore_via_2d_unfold_verl080
在真实 verl 0.8.0 + Megatron Core 0.16.1 环境下的接线正确性与精度一致性。

本地（无 verl/Megatron/GPU）自动 skip。在目标环境（NPU/GPU + verl080）运行。

测试覆盖：
1. **packed_index 一致性**：PackedBatchLayout.packed_index 与 preprocess_thd_engine
   实际生成的 packed 1D 布局一致（含 TP alignment）
2. **restore 后 NestedTensor 满足 no_padding_2_padding 校验**：重组后 offsets
   反映完整 [prefix|suffix] 长度，sequence_offsets[-1] == values.shape[0]
3. **精度一致性**：PS 模式下 reuser 完整序列的 logp/entropy 与 baseline
   （无 PS 独立 forward）数值一致（atol/rtol 容差内）
"""

from __future__ import annotations

import pytest

# 端到端测试需要完整 verl + Megatron + GPU/NPU 环境，本地自动 skip。
verl = pytest.importorskip("verl")
megatron = pytest.importorskip("megatron")
torch = pytest.importorskip("torch")

from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch_verl080
from prefix_sharing.integrations.verl_utils import restore_via_2d_unfold_verl080


# ═══════════════════════════════════════
# 1. packed_index 与 preprocess_thd_engine 一致性
# ═══════════════════════════════════════


def test_packed_index_matches_preprocess_thd_engine():
    """验证 PackedBatchLayout.packed_index 与 verl080 preprocess_thd_engine 的
    packed 1D 布局一致。

    构造一个有 prefix sharing 的 micro-batch，物理裁剪后送入
    preprocess_thd_engine 生成实际 cu_seqlens，与 PackedBatchLayout 的
    cu_seqlens 比对。provider prefix-last 的 provider_1d_pos 必须指向
    preprocess_thd_engine 布局里正确的 logits 位置。

    需要：真实 engine_self（MegatronEngineWithLMHead 实例）、GPU 设备。
    在 CI 的 GPU 流水线中执行。
    """
    # TODO: 接入真实 engine fixture 后填充
    # 1. 构造 batch（provider + reuser，有公共前缀）
    # 2. build_prefix_sharing_micro_batch_verl080 → trimmed_batch + state
    # 3. preprocess_thd_engine(trimmed_batch["input_ids"]) → actual_cu_seqlens
    # 4. assert state.packed_batch_layout.cu_seqlens == actual_cu_seqlens
    # 5. 对每个 prefix-last index：
    #    assert index.provider_1d_pos == actual_cu_seqlens[provider] + prefix_len - 1
    pytest.skip("requires verl080 engine + GPU fixture")


# ═══════════════════════════════════════
# 2. restore 后 NestedTensor 长度校验
# ═══════════════════════════════════════


def test_restore_nested_tensor_satisfies_no_padding_2_padding_assert():
    """验证 restore 后 NestedTensor 满足 no_padding_2_padding 的长度断言：
    sequence_offsets[-1] == values.shape[0]。

    restore 前裁剪后 NestedTensor 长度 < 原始总长度（reuser prefix 被裁）；
    restore 后恢复完整 [prefix|suffix]，长度 == sum(original_lengths)。
    """
    # TODO: 接入真实 forward fixture 后填充
    # 1. forward_step(trimmed_batch) → output_dict (NestedTensor, 裁剪后)
    # 2. restore_via_2d_unfold_verl080(output_dict, ...) → restored
    # 3. assert restored["log_probs"].values().shape[0] == sum(plan.original_lengths)
    # 4. 模拟 no_padding_2_padding 的长度校验通过
    pytest.skip("requires verl080 forward_step + GPU fixture")


# ═══════════════════════════════════════
# 3. 精度一致性（PS vs baseline）
# ═══════════════════════════════════════


def test_ps_logprobs_match_baseline_within_tolerance():
    """PS 模式下 reuser 完整序列 logp 与 baseline（独立 forward）数值一致。

    精度红线：logp/loss/梯度语义与 baseline 一致。KV injection 让 reuser
    suffix 区段看到 provider prefix 的 KV，因此 suffix logp 与 baseline 一致；
    interior/prefix-last 经 restore 后也应一致。

    容差：logp atol=1e-4（TP 下 cross_entropy 数值精度），rtol=1e-3。
    """
    # TODO: 接入真实 engine fixture 后填充
    # 1. 构造 provider + reuser batch
    # 2. baseline: 关闭 PS，独立 forward 每条序列 → baseline_logp
    # 3. PS: 开启 PS，forward_step + restore → ps_logp
    # 4. 对 reuser 完整序列：
    #    assert torch.allclose(ps_logp, baseline_logp, atol=1e-4, rtol=1e-3)
    pytest.skip("requires verl080 engine + GPU fixture")


def test_three_logprob_paths_all_restored():
    """验证三路独立 forward（训练 log_prob / old_log_prob / ref_log_prob）都正确 restore。

    vocab_logprobs patch 装上后覆盖所有 vocab_parallel_log_probs_from_logits 调用，
    三路 forward 天然都被 patch。每路的 reuser prefix 区段都必须 restore，
    否则 KL penalty / ratio / advantage 链路数值错误。
    """
    # TODO: 接入 actor + reference policy fixture 后填充
    pytest.skip("requires actor + ref policy + GPU fixture")
