"""Tests for the hybrid FL-forward + FA-backward backend.

CPU-only tests verify factory wiring and backend metadata.
GPU tests (skipped when flash-attn / flex_attention / CUDA are unavailable)
check forward/backward numerical equivalence and packed-KV retention.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.kv_gather import get_kv_gather_index
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.ps_flex_fwd_flash_bwd_gpu import (
    PSFlexFwdFlashBwdBackend,
    PSFlexFwdFlashBwdFunction,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _make_plan(batch_sizes, prefix_lens):
    """Build a PrefixSharingPlan with controlled provider/reuser layout."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    sequences = []
    next_token = 100
    provider_seqs = {}
    for i, (size, p) in enumerate(zip(batch_sizes, prefix_lens)):
        if p == 0:
            seq = list(range(next_token, next_token + size))
            next_token += size
            provider_seqs[i] = seq
            sequences.append(seq)
        else:
            provider_idx = max(j for j in range(i) if prefix_lens[j] == 0)
            provider_seq = provider_seqs[provider_idx]
            suffix = list(range(next_token, next_token + size - p))
            next_token += size - p
            sequences.append(provider_seq[:p] + suffix)
    return planner.plan(sequences)


def _make_layout(plan, align_size=1):
    rows = [torch.zeros(length, dtype=torch.long) for length in plan.kept_lengths_q]
    return PackedBatchLayout.from_kept_position_rows(rows, align_size=align_size)


def _random_qkv(total, num_heads=4, head_dim=64, dtype=torch.bfloat16, seed=42, requires_grad=False):
    torch.manual_seed(seed)
    q = torch.randn(total, num_heads, head_dim, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(total, num_heads, head_dim, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(total, num_heads, head_dim, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


# ------------------------------------------------------------------
# CPU wiring tests
# ------------------------------------------------------------------

def test_factory_returns_hybrid_backend():
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="ps_flex_fwd_flash_bwd_gpu")
    backend = get_backend_instance(config)
    assert isinstance(backend, PSFlexFwdFlashBwdBackend)


def test_backend_declares_no_kv_expansion():
    assert PSFlexFwdFlashBwdBackend.capabilities.requires_kv_expansion is False
    assert PSFlexFwdFlashBwdBackend.capabilities.supports_cuda is True


def test_config_validate_accepts_hybrid_backend():
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="ps_flex_fwd_flash_bwd_gpu")
    # validate will import flex_attention; if not present it raises RuntimeError,
    # which we allow as an environment-specific skip rather than a config error.
    try:
        config.validate(model_config={"model_type": "text_only_causal_lm"}, integrate_mode="verl_fsdp")
    except RuntimeError as exc:
        if "flex_attention" in str(exc):
            pytest.skip("torch.nn.attention.flex_attention not available")
        raise


def test_build_kv_raises():
    plan = _make_plan([6, 5], [0, 3])
    backend = PSFlexFwdFlashBwdBackend()
    q, k, v = _random_qkv(sum(plan.kept_lengths_q))
    with pytest.raises(RuntimeError, match="does not use build_kv"):
        backend.build_kv(k, v, None, plan, layer_id=0)


# ------------------------------------------------------------------
# GPU numerical tests
# ------------------------------------------------------------------

@pytest.fixture
def gpu_available():
    return torch.cuda.is_available()


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _has_flex_attention():
    try:
        from torch.nn.attention import flex_attention  # noqa: F401
        return True
    except Exception:
        return False


def _can_run_gpu_tests():
    if not torch.cuda.is_available():
        return False
    return _has_flash_attn() and _has_flex_attention()


_gpu_skip = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


_flex_skip = pytest.mark.skipif(
    not _has_flex_attention(),
    reason="torch.nn.attention.flex_attention not available",
)


_flash_skip = pytest.mark.skipif(
    not _has_flash_attn(),
    reason="flash-attn not installed",
)


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_attention_runs_without_build_kv():
    """Smoke test: hybrid backend attention runs and returns packed-shape output."""
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    backend = PSFlexFwdFlashBwdBackend()
    out = backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out.shape == q.shape
    assert out.requires_grad is True
    assert out.grad_fn is not None


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_backward_flows_to_packed_kv():
    """Gradients must reach the original packed K/V."""
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    backend = PSFlexFwdFlashBwdBackend()
    out = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_saved_tensors_are_packed_shape():
    """Autograd must retain packed K/V, not expanded K/V."""
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    backend = PSFlexFwdFlashBwdBackend()
    out = backend.attention(q, k, v, plan, packed_batch_layout=layout)

    saved = out.grad_fn.saved_tensors
    assert len(saved) == 8  # q, packed_k, packed_v, out, lse, index, cu_seqlens_q, cu_seqlens_kv
    saved_k = saved[1]
    saved_v = saved[2]
    assert saved_k.shape == k.shape
    assert saved_v.shape == v.shape
    assert saved_k.shape[0] == layout.total_padded_length


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_forward_matches_flex_attention_backend():
    """Hybrid forward (flex) must match the pure flex_attention backend forward."""
    from prefix_sharing.backends.flex_atten_gpu import GpuFlexAttentionBackend

    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    flex_backend = GpuFlexAttentionBackend()
    hybrid_backend = PSFlexFwdFlashBwdBackend()

    out_flex = flex_backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_hybrid = hybrid_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert torch.allclose(out_flex, out_hybrid, atol=1e-5, rtol=1e-3)


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_gradient_matches_flash_attention_gather_backend():
    """Hybrid gradients must match the FA gather-path gradients."""
    from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend

    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)

    def run(backend_cls):
        _q, _k, _v = q.cuda(), k.cuda(), v.cuda()
        backend = backend_cls()
        if getattr(backend.capabilities, "requires_kv_expansion", True):
            from prefix_sharing.core.prefix_store import PrefixAttentionStore
            _k, _v = backend.build_kv(
                _k, _v, PrefixAttentionStore(), plan,
                packed_batch_layout=layout, layer_id=0,
            )
        out = backend.attention(_q, _k, _v, plan, packed_batch_layout=layout)
        loss = out.sum()
        loss.backward()
        return _q.grad, _k.grad, _v.grad

    dq_fa, dk_fa, dv_fa = run(GpuFlashAttentionBackend)
    dq_hy, dk_hy, dv_hy = run(PSFlexFwdFlashBwdBackend)

    assert torch.allclose(dq_hy, dq_fa, atol=1e-5, rtol=1e-3)
    assert torch.allclose(dk_hy, dk_fa, atol=1e-4, rtol=1e-3)
    assert torch.allclose(dv_hy, dv_fa, atol=1e-4, rtol=1e-3)


@_gpu_skip
@pytest.mark.skipif(not _can_run_gpu_tests(), reason="flash-attn or flex_attention unavailable")
def test_block_mask_not_saved():
    """BlockMask is a non-tensor object and must not appear in saved_tensors."""
    plan = _make_plan([8, 7, 6], [0, 3, 5])
    layout = _make_layout(plan)
    q, k, v = _random_qkv(layout.total_padded_length, requires_grad=True)
    q, k, v = q.cuda(), k.cuda(), v.cuda()

    backend = PSFlexFwdFlashBwdBackend()
    out = backend.attention(q, k, v, plan, packed_batch_layout=layout)

    saved = out.grad_fn.saved_tensors
    for tensor in saved:
        assert isinstance(tensor, torch.Tensor)
    # No BlockMask in saved tensors because ctx.save_for_backward only accepts tensors.
