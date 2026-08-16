"""restore_via_2d_unfold_verl080 单元测试。

验证 v080 restore 包装的核心逻辑：
- 无 context / 非 NestedTensor 时 early return
- _unfold_trimmed_nested_to_2d + _fold_2d_to_nested 的 round-trip 对称性
- interior 从 provider 复制 logp/entropy
- prefix-last 用 saved logits + label_value 重算 logp，entropy 从 provider 复制
- 压回后 NestedTensor offsets 恢复为完整 [prefix | suffix] 长度

完整 PS 精度对比（vs baseline forward）属于 integrated_test，需要真实
verl/Megatron/GPU 环境，本文件只覆盖本地可验证的张量逻辑。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_utils import (
    _fold_2d_to_nested,
    _unfold_trimmed_nested_to_2d,
    restore_via_2d_unfold_verl080,
)


def _make_state(sequences, min_prefix_len=3):
    planner = PrefixSharingPlanner(
        PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix_len)
    )
    plan = planner.plan(sequences, forward_id=10, micro_batch_id=20)
    return PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q),
        parallel_info=MegatronParallelInfo(),
    )


def _mock_vocab_log_probs_fn(logits, labels):
    """简易 log_softmax gather，模拟 vocab_parallel_log_probs_from_logits。

    label 取模防越界（测试用小 vocab 维度，token id 较大）。
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    labels = labels.long() % logits.size(-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


# ═══════════════════════════════════════
# Early return 路径
# ═══════════════════════════════════════


def test_no_context_returns_output_unchanged():
    """contextvar 无 context 时直接返回，不做任何处理。"""
    output = {"log_probs": torch.tensor([1.0, 2.0])}
    result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
    assert result is output


def test_non_nested_log_probs_returns_unchanged():
    """log_probs 不是 NestedTensor 时 early return（可能是 v070 2D 路径误入）。"""
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]])
    original = torch.zeros(2, 5)
    with prefix_sharing_runtime_context(state):
        output = {"log_probs": original.clone()}
        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
    assert result is output
    assert torch.equal(result["log_probs"], original)


def test_no_sharing_returns_output_unchanged():
    """plan 无 reuser（has_sharing=False）时 early return，输出不变。"""
    # 两条无公共前缀的序列 → 无 reuse → has_sharing=False
    state = _make_state([[1, 2, 3, 10, 11], [4, 5, 6, 7, 8]])
    nested = torch.nested.nested_tensor(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])], layout=torch.jagged
    )
    output = {"log_probs": nested}
    with prefix_sharing_runtime_context(state) as ctx:
        assert not ctx.prefix_sharing_plan.has_sharing
        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
        assert result is output


# ═══════════════════════════════════════
# Round-trip 对称性
# ═══════════════════════════════════════


def test_unfold_fold_roundtrip_preserves_values():
    """裁剪后 NestedTensor → 2D left-pad → 压回，值与形态正确。

    row0 provider: len=5（完整），right-pad 到 L_max=6
    row1 reuser:   suffix len=3，left-pad prefix_len=3 个 0
    """
    row0 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    row1 = torch.tensor([1.1, 1.2, 1.3])  # reuser suffix only
    nested = torch.nested.nested_tensor([row0, row1], layout=torch.jagged)

    original_lengths = [5, 6]
    input_keep_ranges = [(0, 5), (3, 6)]
    L_max = 6

    logp_2d, entropy_2d = _unfold_trimmed_nested_to_2d(
        nested, None, original_lengths, input_keep_ranges, L_max, 2,
    )
    assert entropy_2d is None
    assert logp_2d.shape == (2, 6)
    # provider row: 原值 + right-pad 0
    assert torch.allclose(logp_2d[0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]))
    # reuser row: left-pad 3 个 0 + suffix
    assert torch.allclose(logp_2d[1], torch.tensor([0.0, 0.0, 0.0, 1.1, 1.2, 1.3]))

    folded = _fold_2d_to_nested(logp_2d, original_lengths)
    assert folded.offsets().tolist() == [0, 5, 11]
    assert torch.allclose(folded.values()[0:5], row0)
    # reuser 行压回后是完整 6 长度（含 left-pad 的 prefix 区段）
    assert torch.allclose(folded.values()[5:11], logp_2d[1, 0:6])


def test_unfold_handles_pure_reuser_zero_length_row():
    """纯 reuser（suffix_len=0）裁剪后 NestedTensor 行长度为 0。

    展开为全 padding 行；restore 后 interior 填充，prefix-last 仍可能为 padding
    （与 baseline 不算 next-token 语义一致）。
    """
    # suffix_len=0 的行：suffix_data 为空
    row0 = torch.tensor([0.5, 0.6, 0.7])  # provider
    empty_row = torch.tensor([])  # reuser suffix 为空
    nested = torch.nested.nested_tensor([row0, empty_row], layout=torch.jagged)

    original_lengths = [3, 3]  # provider=3, reuser orig=3 (prefix3+suffix0)
    input_keep_ranges = [(0, 3), (3, 3)]  # reuser keep[3:3]=空
    L_max = 3

    logp_2d, _ = _unfold_trimmed_nested_to_2d(
        nested, None, original_lengths, input_keep_ranges, L_max, 2,
    )
    assert logp_2d.shape == (2, 3)
    # reuser 行全 0（left-pad 3 个 0，无 suffix）
    assert torch.allclose(logp_2d[1], torch.zeros(3))


# ═══════════════════════════════════════
# Restore 行为（interior 复制 + prefix-last 重算）
# ═══════════════════════════════════════


def test_restore_copies_interior_and_recomputes_prefix_last():
    """完整 restore：2 interior 复制 + 1 prefix-last 重算。

    sequences: [[1,2,3,10,11], [1,2,3,20,21,22]]
      provider=row0 [1,2,3,10,11] len5
      reuser=row1  [1,2,3,20,21,22] len6, prefix_len=3, suffix=[20,21,22]
      restore: interior @ target_2d_pos=0,1 ; prefix-last @ target_2d_pos=2 (label=20)
    """
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])
    plan = state.prefix_sharing_plan
    assert plan.original_lengths == [5, 6]
    assert plan.input_keep_ranges[1] == (3, 6)

    provider_logp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
    reuser_suffix_logp = torch.tensor([-1.1, -1.2, -1.3])
    nested_logp = torch.nested.nested_tensor(
        [provider_logp, reuser_suffix_logp], layout=torch.jagged
    )
    output = {"log_probs": nested_logp}

    with prefix_sharing_runtime_context(state) as ctx:
        # 仅 1 条 prefix-last（interior 由 restore 侧 bulk 切片处理，不建索引）
        assert len(ctx.prefix_last_restore_indices) == 1
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        assert prefix_last_idx.target_2d_pos == 2
        assert prefix_last_idx.label_value == 20  # input_ids[1][3]

        # 预填 saved logits（vocab 维=4），模拟 vocab_logprobs patch 保存的结果
        saved_logits = torch.tensor([[0.5, 0.3, 0.1, 0.1]])  # [1, 4]
        ctx.prefix_last_logits_saved[
            (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ] = saved_logits

        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)

    restored = result["log_probs"]
    offsets = restored.offsets()
    assert offsets.tolist() == [0, 5, 11]  # provider 5 + reuser 完整 6

    values = restored.values()
    # provider 原样
    assert torch.allclose(values[0:5], provider_logp)

    reuser = values[5:11]
    # interior pos0,1 从 provider 复制
    assert torch.allclose(reuser[0], provider_logp[0])
    assert torch.allclose(reuser[1], provider_logp[1])
    # prefix-last pos2 重算：log_softmax(saved)[label=20%4=0]
    expected_plast = torch.log_softmax(saved_logits.float(), dim=-1)[0, 0]
    assert torch.allclose(reuser[2], expected_plast)
    # suffix 原样保留
    assert torch.allclose(reuser[3:6], reuser_suffix_logp)


def test_restore_with_entropy_copies_both_logp_and_entropy():
    """entropy 同步 restore：interior 和 prefix-last 都从 provider 复制（不重算）。"""
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])

    provider_logp = torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5])
    reuser_logp = torch.tensor([-1.1, -1.2, -1.3])
    provider_ent = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    reuser_ent = torch.tensor([1.5, 1.6, 1.7])

    nested_logp = torch.nested.nested_tensor(
        [provider_logp, reuser_logp], layout=torch.jagged
    )
    nested_ent = torch.nested.nested_tensor(
        [provider_ent, reuser_ent], layout=torch.jagged
    )
    output = {"log_probs": nested_logp, "entropy": nested_ent}

    with prefix_sharing_runtime_context(state) as ctx:
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        ctx.prefix_last_logits_saved[
            (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ] = torch.tensor([[0.5, 0.3, 0.1, 0.1]])

        result = restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)

    ent_values = result["entropy"].values()
    ent_offsets = result["entropy"].offsets()
    assert ent_offsets.tolist() == [0, 5, 11]
    reuser_ent = ent_values[5:11]
    # interior + prefix-last 的 entropy 都从 provider 复制
    assert torch.allclose(reuser_ent[0], provider_ent[0])  # interior pos0
    assert torch.allclose(reuser_ent[1], provider_ent[1])  # interior pos1
    assert torch.allclose(reuser_ent[2], provider_ent[2])  # prefix-last pos2（复制非重算）
    # suffix entropy 原样
    assert torch.allclose(reuser_ent[3:6], torch.tensor([1.5, 1.6, 1.7]))


def test_restore_clears_saved_logits_is_callers_responsibility():
    """包装函数不清空 prefix_last_logits_saved（由 forward_step patch 调用后 clear）。

    验证 restore 后 saved dict 仍非空（清理职责在调用方）。
    """
    state = _make_state([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])
    nested_logp = torch.nested.nested_tensor(
        [torch.tensor([-0.1, -0.2, -0.3, -0.4, -0.5]),
         torch.tensor([-1.1, -1.2, -1.3])],
        layout=torch.jagged,
    )
    output = {"log_probs": nested_logp}

    with prefix_sharing_runtime_context(state) as ctx:
        prefix_last_idx = ctx.prefix_last_restore_indices[0]
        key = (prefix_last_idx.reuse_idx_in_batch, prefix_last_idx.target_2d_pos)
        ctx.prefix_last_logits_saved[key] = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        restore_via_2d_unfold_verl080(output, _mock_vocab_log_probs_fn)
        # 包装函数不清空，调用方负责
        assert key in ctx.prefix_last_logits_saved
