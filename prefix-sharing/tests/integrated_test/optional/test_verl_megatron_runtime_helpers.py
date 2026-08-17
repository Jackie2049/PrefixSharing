"""Shared core tests: restore_reuser_prefix_columns_2d + prefix_attention integration.

These tests verify the shared prefix-sharing core logic used by both verl070
and verl080 paths. The batch fixture is constructed inline.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.context import current_prefix_sharing_context, prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_runtime import prefix_attention
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo
from prefix_sharing.integrations.verl_utils import restore_reuser_prefix_columns_2d


def _make_state() -> tuple[PrefixSharingRuntimeState, list]:
    """Build a trivial 2-row state with prefix sharing (prefix_len=3).

    Returns:
        (state, prefix_last_restore_indices) so the caller can inspect indices.
    """
    config = PrefixSharingConfig.from_raw({"enable_prefix_sharing": True, "min_prefix_len": 3})
    sequences = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]]
    plan = PrefixSharingPlanner(config).plan(sequences)
    kept_position_rows = [torch.arange(5), torch.arange(2, 5)]
    layout = PackedBatchLayout.from_kept_position_rows(kept_position_rows, align_size=1)
    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        attention_backend=None,
        packed_batch_layout=layout,
        parallel_info=MegatronParallelInfo(
            global_rank=0, tp_rank=0, tp_size=1,
            cp_rank=0, cp_size=1,
            pp_rank=0, pp_size=1,
            is_pipeline_first_stage=True, is_pipeline_last_stage=True,
        ),
    )
    return state


def test_restore_reuser_prefix_columns_2d_prefix_last_keeps_autograd():
    # input: [[1,2,3,10,11], [1,2,3,20,21]] → prefix_len=3, prompt_len=3.
    # Only the prefix-last spec is indexed (interior is bulk-sliced in the
    # restore). Reuser's first suffix token at target_2d_pos=2 differs from
    # provider's, so its logprob must be recomputed from saved provider logits.
    state = _make_state()

    def gather_fn(provider_logits, reuse_label):
        return torch.gather(
            torch.log_softmax(provider_logits, dim=-1),
            dim=-1,
            index=reuse_label.unsqueeze(-1),
        ).squeeze(-1)

    with prefix_sharing_runtime_context(state) as ctx:
        # 1 prefix-last spec (interior bulk-sliced, not indexed)
        assert len(ctx.prefix_last_restore_indices) == 1
        index = ctx.prefix_last_restore_indices[0]  # prefix-last spec

        # Simulate 2D postprocess: output dict with [B, L] log_probs.
        log_probs_2d = torch.zeros(2, 5)
        output = {"log_probs": log_probs_2d}

        # Saved provider packed logits for prefix-last recompute.
        saved_logits = torch.randn(1, 32, requires_grad=True)
        ctx.prefix_last_logits_saved[(index.reuse_idx_in_batch, index.target_2d_pos)] = saved_logits

        output = restore_reuser_prefix_columns_2d(output, gather_fn)
        assert ctx.stats.actual_restore_count == ctx.stats.expected_restore_count == 1

        restored_val = output["log_probs"][index.reuse_idx_in_batch, index.target_2d_pos]

    # Gradient must flow through saved_logits.
    restored_val.backward()
    assert saved_logits.grad is not None
    assert saved_logits.grad.abs().sum() > 0


def test_attention_hook_rejects_sp_local_shard_token_length():
    state = _make_state()
    attention_module = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True), layer_number=1)
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    query = torch.randn(6, 1, 2)
    key = torch.randn(6, 1, 2)
    value = torch.randn(6, 1, 2)

    with prefix_sharing_runtime_context(state):
        with pytest.raises(RuntimeError, match="SP-local shard"):
            prefix_attention(
                attention_module,
                query,
                key,
                value,
                attention_mask=None,
                rotary_pos_emb=(object(), object()),
                packed_seq_params=packed_seq_params,
            )
