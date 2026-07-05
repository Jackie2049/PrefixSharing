from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    build_prefix_sharing_micro_batch_fsdp,
    forward_prefix_sharing_fsdp_micro_batch,
    restore_prefix_sharing_outputs_2d,
)
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.setup.patches.verl080_fsdp.forward_step import patch_fsdp_forward_step


def _mock_log_probs_fn(logits, labels):
    logp = torch.log_softmax(logits.float(), dim=-1)
    labels = labels.long() % logits.size(-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _entropy_from_logits(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


class _TinyHFStyleModel(torch.nn.Module):
    def __init__(self, *, vocab_size=32, hidden_heads=2, head_dim=4):
        super().__init__()
        self.hidden_heads = hidden_heads
        self.head_dim = head_dim
        hidden = hidden_heads * head_dim
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, prefix_sharing_runtime=None):
        del attention_mask, position_ids, use_cache
        hidden = self.embed(input_ids)
        query = self.q_proj(hidden).reshape(*hidden.shape[:2], self.hidden_heads, self.head_dim)
        key = self.k_proj(hidden).reshape(*hidden.shape[:2], self.hidden_heads, self.head_dim)
        value = self.v_proj(hidden).reshape(*hidden.shape[:2], self.hidden_heads, self.head_dim)
        if prefix_sharing_runtime is None:
            attn_output = _baseline_attention(query, key, value)
        else:
            attn_output = prefix_sharing_runtime.forward(None, query, key, value)
        flat_output = self.o_proj(attn_output.reshape(*hidden.shape[:2], -1))
        logits = self.lm_head(flat_output)
        return type("Output", (), {"logits": logits, "attention_output": flat_output})()


class _EngineConfig:
    def __init__(self, prefix_sharing_config, **kwargs):
        self.prefix_sharing_config = prefix_sharing_config
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeFSDPEngine:
    def __init__(self, module, engine_config):
        self.module = module
        self.engine_config = engine_config

    def get_data_parallel_group(self):
        return None


class _FakeNativeFSDPEngine(_FakeFSDPEngine):
    def __init__(self, module, engine_config):
        super().__init__(module, engine_config)
        self._autocast_dtype = torch.float32

    def prepare_model_inputs(self, micro_batch):
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        if hasattr(input_ids, "values"):
            model_input_ids = input_ids.values().unsqueeze(0)
            model_position_ids = position_ids.values().unsqueeze(0)
            offsets = input_ids.offsets()
            labels = []
            for row in range(offsets.numel() - 1):
                row_values = input_ids.values()[offsets[row]:offsets[row + 1]]
                row_labels = torch.roll(row_values, shifts=-1, dims=0)
                if row_labels.numel() > 0:
                    row_labels[-1] = 0
                labels.append(row_labels)
            flat_labels = torch.cat(labels, dim=0)
            return {
                "input_ids": model_input_ids,
                "attention_mask": None,
                "position_ids": model_position_ids,
            }, {"labels": flat_labels, "offsets": offsets}
        return {
            "input_ids": input_ids,
            "attention_mask": micro_batch.get("attention_mask"),
            "position_ids": position_ids,
        }, {"labels": micro_batch["labels"]}

    def prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        del logits_processor_func
        logits = output.logits
        if logits.dim() == 3 and logits.shape[0] == 1:
            flat_logits = logits.squeeze(0)
            labels = output_args["labels"]
            log_probs = _mock_log_probs_fn(flat_logits, labels)
            offsets = output_args["offsets"]
            rows = [
                log_probs[offsets[row]:offsets[row + 1]]
                for row in range(offsets.numel() - 1)
            ]
            values = torch.cat(rows, dim=0)
            return {"log_probs": torch.nested.nested_tensor_from_jagged(values, offsets)}
        return {"log_probs": _mock_log_probs_fn(logits, output_args["labels"]), "logits": logits}


def _baseline_attention(query, key, value):
    scale = query.shape[-1] ** -0.5
    scores = torch.einsum("blhd,bmhd->blmh", query, key) * scale
    length = query.shape[1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=query.device)).unsqueeze(-1)
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=2)
    return torch.einsum("blmh,bmhd->blhd", probs, value)


def test_build_prefix_sharing_micro_batch_fsdp_returns_trimmed_batch_and_runtime_state():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [
                [2, 3, 10, 11, -100, -100],
                [2, 3, 20, 21, 22, -100],
            ],
            dtype=torch.long,
        ),
        "loss_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
    }

    trimmed_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)

    assert runtime_state is not None
    assert isinstance(runtime_state, PrefixSharingRuntimeState)
    plan = runtime_state.prefix_sharing_plan
    assert plan.has_sharing
    assert plan.provider_index == [0, 0]
    assert plan.prefix_lens == [0, 3]
    assert plan.input_keep_ranges == [(0, 5), (3, 6)]
    assert runtime_state.packed_batch_layout.valid_lengths == [5, 3]
    assert runtime_state.packed_batch_layout.padded_lengths == [5, 3]
    assert runtime_state.packed_batch_layout.cu_seqlens == [0, 5, 8]
    assert runtime_state.packed_batch_layout.packed_position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4, 5]

    assert trimmed_batch is not batch
    assert torch.equal(trimmed_batch["attention_mask"][0], batch["attention_mask"][0])
    assert trimmed_batch["attention_mask"][1].tolist() == [False, False, False, True, True, True]
    assert trimmed_batch["loss_mask"][1].tolist() == [False, False, False, True, True, False]
    assert torch.equal(trimmed_batch["position_ids"][1, 3:6], torch.tensor([3, 4, 5]))


def test_build_prefix_sharing_micro_batch_fsdp_returns_none_when_no_sharing():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
    }

    returned_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)

    assert returned_batch is batch
    assert runtime_state is None


def test_build_prefix_sharing_micro_batch_fsdp_trims_nested_remove_padding_batch():
    if not hasattr(torch, "nested"):
        pytest.skip("torch.nested is unavailable")
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.nested.nested_tensor(
            [
                torch.tensor([1, 2, 3, 10, 11], dtype=torch.long),
                torch.tensor([1, 2, 3, 20, 21, 22], dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
        "position_ids": torch.nested.nested_tensor(
            [
                torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
        "loss_mask": torch.nested.nested_tensor(
            [
                torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
                torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool),
            ],
            layout=torch.jagged,
        ),
    }

    trimmed_batch, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)

    assert runtime_state is not None
    plan = runtime_state.prefix_sharing_plan
    assert plan.input_keep_ranges == [(0, 5), (3, 6)]
    trimmed_offsets = trimmed_batch["input_ids"].offsets()
    trimmed_values = trimmed_batch["input_ids"].values()
    assert trimmed_offsets.tolist() == [0, 5, 8]
    assert trimmed_values.tolist() == [1, 2, 3, 10, 11, 20, 21, 22]
    assert runtime_state.packed_batch_layout.valid_lengths == [5, 3]
    assert runtime_state.packed_batch_layout.packed_position_ids.tolist() == [0, 1, 2, 3, 4, 3, 4, 5]


def test_restore_prefix_sharing_outputs_2d_restores_interior_last_logits_entropy_and_attention_output():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
    }
    _, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)
    assert runtime_state is not None

    vocab = 5
    hidden = 4
    output = {
        "log_probs": torch.tensor(
            [
                [-0.1, -0.2, -0.3, -0.4, -0.5, 0.0],
                [0.0, 0.0, 0.0, -1.3, -1.4, -1.5],
            ]
        ),
        "entropy": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
                [0.0, 0.0, 0.0, 1.3, 1.4, 1.5],
            ]
        ),
        "logits": torch.arange(2 * 6 * vocab, dtype=torch.float32).reshape(2, 6, vocab),
        "attention_output": torch.arange(2 * 6 * hidden, dtype=torch.float32).reshape(2, 6, hidden),
    }
    original_reuser_suffix_logits = output["logits"][1, 3:].clone()
    original_reuser_suffix_attention = output["attention_output"][1, 3:].clone()

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        restore_index = ctx.prefix_last_restore_indices[0]
        saved_logits = output["logits"][restore_index.provider_idx_in_batch, restore_index.target_2d_pos:restore_index.target_2d_pos + 1]
        ctx.prefix_last_logits_saved[(restore_index.reuse_idx_in_batch, restore_index.target_2d_pos)] = saved_logits
        restored = restore_prefix_sharing_outputs_2d(output, _mock_log_probs_fn)

    # interior prefix logp/entropy copied from provider.
    assert torch.allclose(restored["log_probs"][1, 0:2], restored["log_probs"][0, 0:2])
    assert torch.allclose(restored["entropy"][1, 0:3], restored["entropy"][0, 0:3])

    # prefix-last logp recomputed using provider logits and reuser first suffix label (token 20 -> 0 mod vocab).
    expected = torch.log_softmax(saved_logits.float(), dim=-1)[0, 20 % vocab]
    assert torch.allclose(restored["log_probs"][1, 2], expected)

    # logits and attention output for the whole prefix copied from provider.
    assert torch.allclose(restored["logits"][1, 0:3], restored["logits"][0, 0:3])
    assert torch.allclose(restored["attention_output"][1, 0:3], restored["attention_output"][0, 0:3])

    # suffix part stays untouched.
    assert torch.allclose(restored["logits"][1, 3:], original_reuser_suffix_logits)
    assert torch.allclose(restored["attention_output"][1, 3:], original_reuser_suffix_attention)


def test_prefix_sharing_fsdp_attention_runtime_scatter_dense_outputs():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
    }
    _, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)
    assert runtime_state is not None

    torch.manual_seed(1)
    query = torch.randn(2, 6, 2, 4)
    key = torch.randn(2, 6, 2, 4)
    value = torch.randn(2, 6, 2, 4)

    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=7)
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        dense_output = runtime.forward(None, query, key, value)
        assert ctx.stats.layers[7].reuse_hit_count == 1

    assert dense_output.shape == query.shape
    # Provider valid tokens are computed; provider padding remains zero.
    assert not torch.allclose(dense_output[0, 0:5], torch.zeros_like(dense_output[0, 0:5]))
    assert torch.allclose(dense_output[0, 5], torch.zeros_like(dense_output[0, 5]))
    # Reuser prefix is not computed on Q path and is restored later.
    assert torch.allclose(dense_output[1, 0:3], torch.zeros_like(dense_output[1, 0:3]))
    # Reuser suffix is computed.
    assert not torch.allclose(dense_output[1, 3:6], torch.zeros_like(dense_output[1, 3:6]))


def test_forward_prefix_sharing_fsdp_micro_batch_matches_tiny_hf_model_baseline():
    torch.manual_seed(2026)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 20, 21],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
    }
    labels = torch.roll(batch["input_ids"], shifts=-1, dims=1)
    labels[:, -1] = 0
    batch["labels"] = labels

    model = _TinyHFStyleModel(vocab_size=32)
    baseline = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    baseline_logits = baseline.logits
    baseline_log_probs = _mock_log_probs_fn(baseline_logits, labels)
    baseline_entropy = _entropy_from_logits(baseline_logits)

    prefix_output = forward_prefix_sharing_fsdp_micro_batch(
        batch,
        model,
        config,
        calculate_entropy=True,
        log_probs_fn=_mock_log_probs_fn,
        entropy_fn=_entropy_from_logits,
    )

    assert torch.allclose(prefix_output["logits"], baseline_logits, atol=1e-5)
    assert torch.allclose(prefix_output["log_probs"], baseline_log_probs, atol=1e-5)
    assert torch.allclose(prefix_output["entropy"], baseline_entropy, atol=1e-5)
    assert torch.allclose(prefix_output["attention_output"], baseline.attention_output, atol=1e-5)


def test_forward_prefix_sharing_fsdp_micro_batch_keeps_provider_prefix_grad_path():
    torch.manual_seed(2027)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 20, 21],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
    }
    labels = torch.roll(batch["input_ids"], shifts=-1, dims=1)
    labels[:, -1] = 0
    batch["labels"] = labels

    model = _TinyHFStyleModel(vocab_size=32)
    output = forward_prefix_sharing_fsdp_micro_batch(
        batch,
        model,
        config,
        log_probs_fn=_mock_log_probs_fn,
    )
    loss = -output["log_probs"][1, 3:5].sum()
    loss.backward()

    provider_prefix_ids = batch["input_ids"][0, 0:3]
    grad = model.embed.weight.grad
    assert grad is not None
    assert grad[provider_prefix_ids].abs().sum() > 0


def test_verl_fsdp_attention_patch_installs_and_rolls_back():
    transformers_modeling_utils = pytest.importorskip("transformers.modeling_utils")
    registry = transformers_modeling_utils.ALL_ATTENTION_FUNCTIONS
    if "eager" not in registry:
        pytest.skip("transformers eager attention function is unavailable")

    original = registry["eager"]
    handle = None
    try:
        handle = __import__(
            "prefix_sharing.integrations.verl_fsdp",
            fromlist=["VerlFSDPIntegration"],
        ).VerlFSDPIntegration._install_transformers_attention_patch()
        assert handle.active
        assert registry["eager"] is not original
    finally:
        if handle is not None:
            handle.disable()
    assert registry["eager"] is original


def test_verl080_fsdp_forward_step_patch_runs_prefix_sharing_path():
    torch.manual_seed(2030)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 20, 21],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
    }
    labels = torch.roll(batch["input_ids"], shifts=-1, dims=1)
    labels[:, -1] = 0
    batch["labels"] = labels

    def original_forward_step(self, micro_batch, loss_function, forward_only):
        raise AssertionError("original forward_step should not run when prefix sharing is enabled")

    def loss_function(model_output, data, dp_group):
        del data, dp_group
        loss = -model_output["log_probs"][1, 3:5].sum()
        return loss, {"loss_tokens": 2}

    patched = patch_fsdp_forward_step(original_forward_step)
    engine = _FakeFSDPEngine(
        _TinyHFStyleModel(vocab_size=32),
        _EngineConfig({"enable_prefix_sharing": True, "min_prefix_len": 3}),
    )

    loss, output = patched(engine, batch, loss_function, forward_only=False)

    assert loss.requires_grad
    assert output["metrics"] == {"loss_tokens": 2}
    assert "log_probs" in output["model_output"]
    assert "logits" in output["model_output"]


def test_verl080_fsdp_forward_step_patch_falls_back_when_disabled():
    def original_forward_step(self, micro_batch, loss_function, forward_only):
        return "loss", {"model_output": {"fallback": True}}

    patched = patch_fsdp_forward_step(original_forward_step)
    engine = _FakeFSDPEngine(
        _TinyHFStyleModel(vocab_size=32),
        _EngineConfig({"enable_prefix_sharing": False}),
    )

    result = patched(engine, {}, None, forward_only=True)

    assert result == ("loss", {"model_output": {"fallback": True}})


def test_verl080_fsdp_forward_step_patch_allows_remove_padding_config_without_engine_prepare():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
    }

    def original_forward_step(self, micro_batch, loss_function, forward_only):
        raise AssertionError("original forward_step should not run")

    patched = patch_fsdp_forward_step(original_forward_step)
    engine = _FakeFSDPEngine(
        _TinyHFStyleModel(vocab_size=32),
        _EngineConfig(
            {"enable_prefix_sharing": True, "min_prefix_len": 2},
            use_remove_padding=True,
        ),
    )

    loss, output = patched(engine, batch, None, forward_only=True)

    assert output["loss"] == pytest.approx(float(loss.detach().item()))


def test_verl080_fsdp_forward_step_patch_runs_native_nested_prepare_outputs_path():
    if not hasattr(torch, "nested"):
        pytest.skip("torch.nested is unavailable")
    torch.manual_seed(2032)
    batch = {
        "input_ids": torch.nested.nested_tensor(
            [
                torch.tensor([1, 2, 3, 10, 11], dtype=torch.long),
                torch.tensor([1, 2, 3, 20, 21, 22], dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
        "position_ids": torch.nested.nested_tensor(
            [
                torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
    }

    def original_forward_step(self, micro_batch, loss_function, forward_only):
        raise AssertionError("original forward_step should not run when prefix sharing is enabled")

    def loss_function(model_output, data, dp_group):
        del data, dp_group
        values = model_output["log_probs"].values()
        return -values.sum(), {"restored_tokens": int(values.numel())}

    patched = patch_fsdp_forward_step(original_forward_step)
    engine = _FakeNativeFSDPEngine(
        _TinyHFStyleModel(vocab_size=32),
        _EngineConfig(
            {"enable_prefix_sharing": True, "min_prefix_len": 3},
            use_remove_padding=True,
        ),
    )

    loss, output = patched(engine, batch, loss_function, forward_only=False)

    log_probs = output["model_output"]["log_probs"]
    assert hasattr(log_probs, "offsets")
    assert log_probs.offsets().tolist() == [0, 5, 11]
    assert output["metrics"] == {"restored_tokens": 11}
    assert loss.requires_grad


def test_prefix_sharing_fsdp_attention_runtime_supports_packed_single_batch_shape():
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11],
                [1, 2, 3, 20, 21],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
    }
    _, runtime_state = build_prefix_sharing_micro_batch_fsdp(batch, config)
    assert runtime_state is not None

    torch.manual_seed(2031)
    total_kept = runtime_state.packed_batch_layout.total_padded_length
    query = torch.randn(1, total_kept, 4, 4)
    key = torch.randn(1, total_kept, 2, 4)
    value = torch.randn(1, total_kept, 2, 4)

    runtime = PrefixSharingFSDPAttentionRuntime(layer_id=11)
    with prefix_sharing_runtime_context(runtime_state) as ctx:
        output = runtime.forward(None, query, key, value)

    assert output.shape == query.shape
    assert ctx.stats.layers[11].reuse_hit_count == 1
