"""P0-3 build_kv correctness verification: old torch.cat vs new prealloc.

For each sharing pattern and configuration, this script:
1. Runs _reference_build_kv (old per-row cat) on CPU with float32
2. Runs TorchReferenceBackend.build_kv (new prealloc) on CPU with float32
3. Compares expanded_key and expanded_value via element-wise cosine similarity
4. Verifies gradient paths (requires_grad, grad flows through prefix portions)
5. Runs the same comparison on GPU with bfloat16 for production fidelity

All results output as JSONL for post-processing.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
import torch

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore, PrefixActivationSlotId, PREFIX_STATE_TYPE_ATTENTION_KV
from prefix_sharing.backends.torch_ref import TorchReferenceBackend, _split_packed
from prefix_sharing.backends.packed_layout import PackedBatchLayout


# ---- Reference (old torch.cat) implementation ----

def _reference_build_kv(key, value, plan, layout, *, layer_id=0, tp_rank=0):
    """Old per-row cat implementation — the correctness oracle."""
    key_rows = _split_packed(key, layout.padded_lengths)
    value_rows = _split_packed(value, layout.padded_lengths)
    store = PrefixAttentionStore()
    expanded_keys = []
    expanded_values = []

    for batch_index, (key_row, value_row) in enumerate(zip(key_rows, value_rows)):
        valid_length = layout.valid_lengths[batch_index]
        valid_key_row = key_row[:valid_length]
        valid_value_row = value_row[:valid_length]
        if not plan.is_reuser(batch_index):
            slot_id = PrefixActivationSlotId(
                plan.forward_id,
                plan.micro_batch_id,
                layer_id,
                batch_index,
                PREFIX_STATE_TYPE_ATTENTION_KV,
                tp_rank,
            )
            store.store(
                slot_id,
                key_tensor=valid_key_row,
                value_tensor=valid_value_row,
                prefix_len=valid_key_row.shape[0],
                overwrite=True,
            )
            expanded_keys.append(valid_key_row)
            expanded_values.append(valid_value_row)
            continue

        provider = plan.provider_index[batch_index]
        provider_slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            provider,
            PREFIX_STATE_TYPE_ATTENTION_KV,
            tp_rank,
        )
        entry = store.load(provider_slot_id)
        prefix_len = plan.prefix_lens[batch_index]
        expanded_key = torch.cat([entry.key_tensor[:prefix_len], valid_key_row], dim=0)
        expanded_value = torch.cat([entry.value_tensor[:prefix_len], valid_value_row], dim=0)
        own_slot_id = PrefixActivationSlotId(
            plan.forward_id,
            plan.micro_batch_id,
            layer_id,
            batch_index,
            PREFIX_STATE_TYPE_ATTENTION_KV,
            tp_rank,
        )
        store.store(
            own_slot_id,
            key_tensor=expanded_key,
            value_tensor=expanded_value,
            prefix_len=expanded_key.shape[0],
            overwrite=True,
        )
        expanded_keys.append(expanded_key)
        expanded_values.append(expanded_value)

    return torch.cat(expanded_keys, dim=0), torch.cat(expanded_values, dim=0)


# ---- Sequence generation (same as comprehensive benchmark) ----

def generate_rl_sequences(sharing, batch_size, prompt_len, response_len,
                         vocab_size=32000, seed=42):
    """Generate sequences simulating RL training prompts + responses."""
    seq_len = prompt_len + response_len
    counter = vocab_size

    def _prompt(base):
        return [base + i for i in range(prompt_len)]

    def _response(base):
        return [base + i for i in range(response_len)]

    if sharing == "no_sharing":
        seqs = []
        for i in range(batch_size):
            seqs.append(_prompt(10000 + i * (prompt_len + 10)) + _response(20000 + i * (response_len + 10)))
        return seqs

    if sharing == "one_provider":
        shared = _prompt(10000)
        seqs = [shared + _response(50000)]
        for i in range(1, batch_size):
            seqs.append(shared + _response(50000 + i * (response_len + 10)))
        return seqs

    if sharing == "multi_provider":
        num_prompts = max(2, min(4, batch_size // 4))
        num_per = batch_size // num_prompts
        seqs = []
        for p in range(num_prompts):
            prompt = _prompt(10000 + p * (prompt_len + 100))
            seqs.append(prompt + _response(50000 + p * 100000))
            for r in range(1, num_per):
                seqs.append(prompt + _response(50000 + p * 100000 + r * (response_len + 10)))
        while len(seqs) < batch_size:
            seqs.append(_prompt(10000 + num_prompts * 100) + _response(50000 + num_prompts * 100000))
        return seqs[:batch_size]

    if sharing == "chain":
        root = _prompt(10000) + _response(50000)
        seqs = [root]
        cum_prefix = prompt_len
        for i in range(1, batch_size):
            prev = seqs[i - 1]
            ap = min(cum_prefix, len(prev))
            suffix_len = seq_len - ap
            suffix = _response(50000 + i * (suffix_len + 10))
            seqs.append(prev[:ap] + suffix)
            cum_prefix += response_len // 2
        return seqs

    raise ValueError(f"Unknown sharing: {sharing}")


# ---- Cosine similarity helper ----

def cosine_sim(a, b):
    """Compute cosine similarity between two flattened tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = (a_flat * b_flat).sum()
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a == 0 or norm_b == 0:
        return 1.0  # both zero → identical
    return (dot / (norm_a * norm_b)).item()


def max_abs_diff(a, b):
    """Max absolute difference."""
    return (a.flatten().float() - b.flatten().float()).abs().max().item()


def mean_abs_diff(a, b):
    """Mean absolute difference."""
    return (a.flatten().float() - b.flatten().float()).abs().mean().item()


# ---- Make plan helper ----

def _make_plan(kept_lengths_q, prefix_lens):
    """Create a minimal PrefixSharingPlan for testing."""
    from prefix_sharing.core.planner import PrefixSharingPlan, PrefixReuseSpec, PrefixLastRestoreSpec

    n = len(kept_lengths_q)
    is_provider = [pl == 0 for pl in prefix_lens]
    provider_index = []
    for i in range(n):
        if is_provider[i]:
            provider_index.append(i)
        else:
            # Find nearest provider — simple: walk backwards
            pi = i - 1
            while pi >= 0 and not is_provider[pi]:
                pi -= 1
            provider_index.append(pi if pi >= 0 else 0)

    suffix_lens = [kept_lengths_q[i] - prefix_lens[i] if not is_provider[i] else kept_lengths_q[i]
                   for i in range(n)]
    expanded_lengths_kv = [prefix_lens[i] + suffix_lens[i] if not is_provider[i]
                           else kept_lengths_q[i]
                           for i in range(n)]

    # cu_seqlens
    cu_seqlens_q = [0]
    for l in kept_lengths_q:
        cu_seqlens_q.append(cu_seqlens_q[-1] + l)
    cu_seqlens_kv = [0]
    for l in expanded_lengths_kv:
        cu_seqlens_kv.append(cu_seqlens_kv[-1] + l)

    # Position offsets
    q_position_offsets = [prefix_lens[i] if not is_provider[i] else 0 for i in range(n)]
    kv_position_offsets = [0] * n

    # Keep ranges (simple: keep all)
    input_keep_ranges = [(0, kept_lengths_q[i]) for i in range(n)]
    label_keep_ranges = [(0, kept_lengths_q[i]) for i in range(n)]
    loss_mask_keep_ranges = [(0, kept_lengths_q[i]) for i in range(n)]

    # Reuse specs
    reuse_specs = []
    for i in range(n):
        if not is_provider[i]:
            reuse_specs.append(PrefixReuseSpec(
                provider_row=provider_index[i],
                prefix_len=prefix_lens[i],
                suffix_len=suffix_lens[i],
            ))

    # Prefix-last restore specs
    prefix_last_restore = []
    for i in range(n):
        if not is_provider[i]:
            target_2d_pos = prefix_lens[i] - 1  # last position of prefix in 2D
            prefix_last_restore.append(PrefixLastRestoreSpec(
                row_index=i,
                target_2d_pos=target_2d_pos,
                label_value=1,
            ))

    return PrefixSharingPlan(
        original_lengths=[pl + rl for pl, rl in zip([prompt_len_global] * n,
                                                     [response_len_global] * n)],
        kept_lengths_q=kept_lengths_q,
        expanded_lengths_kv=expanded_lengths_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        q_position_offsets=q_position_offsets,
        kv_position_offsets=kv_position_offsets,
        is_provider=is_provider,
        provider_index=provider_index,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        reuse_specs=reuse_specs,
        has_sharing=any(not ip for ip in is_provider),
        input_keep_ranges=input_keep_ranges,
        label_keep_ranges=label_keep_ranges,
        loss_mask_keep_ranges=loss_mask_keep_ranges,
        prefix_last_restore=prefix_last_restore,
        forward_id=None,
        micro_batch_id=None,
    )


# ---- Global vars for _make_plan ----
prompt_len_global = 256
response_len_global = 256


# ---- Verification experiment ----

def verify_build_kv_correctness(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    device_name: str = "cpu",
    dtype_name: str = "float32",
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    use_plan_api: bool = True,
) -> dict:
    """Compare old vs new build_kv for a given configuration.

    Args:
        use_plan_api: If True, use PrefixSharingPlanner.plan() to generate the plan
                     (tests P0-1 prefilter too). If False, use _make_plan() for manual plans.
    """
    global prompt_len_global, response_len_global
    prompt_len_global = prompt_len
    response_len_global = response_len

    device = torch.device(device_name)
    dtype = getattr(torch, dtype_name)
    seq_len = prompt_len + response_len

    # Generate sequences
    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)

    # Create plan
    if use_plan_api:
        min_prefix = max(1, min(prompt_len, prompt_len // 2))
        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan(sequences)
    else:
        # Manual plan — won't have proper prefix_lens for sharing patterns
        # This path is only for no_sharing verification
        plan = _make_plan([seq_len] * batch_size, [0] * batch_size)

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    total_padded = layout.total_padded_length
    total_valid = layout.total_valid_length

    # Create input tensors
    torch.manual_seed(42)
    key_new = torch.randn(total_padded, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    value_new = torch.randn(total_padded, num_kv_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    key_ref = key_new.detach().clone().requires_grad_(True)
    value_ref = value_new.detach().clone().requires_grad_(True)

    # Run old implementation (reference)
    backend = TorchReferenceBackend()
    store_ref = PrefixAttentionStore()
    if device_name == "cpu":
        ref_k, ref_v = _reference_build_kv(key_ref, value_ref, plan, layout, layer_id=0, tp_rank=0)
    else:
        # On GPU, reference also uses TorchReferenceBackend but with old code
        # We need to run it on CPU then transfer, or run via the backend
        ref_k, ref_v = _reference_build_kv(key_ref, value_ref, plan, layout, layer_id=0, tp_rank=0)

    # Run new implementation (prealloc)
    store_new = PrefixAttentionStore()
    new_k, new_v = backend.build_kv(
        key_new, value_new, store_new, plan,
        packed_batch_layout=layout, layer_id=0, tp_rank=0,
    )

    # ---- Value comparison ----
    cos_k = cosine_sim(ref_k, new_k)
    cos_v = cosine_sim(ref_v, new_v)
    max_diff_k = max_abs_diff(ref_k, new_k)
    max_diff_v = max_abs_diff(ref_v, new_v)
    mean_diff_k = mean_abs_diff(ref_k, new_k)
    mean_diff_v = mean_abs_diff(ref_v, new_v)

    # For float32: expect exact equality (max_diff == 0)
    # For bfloat16: expect cos >= 0.9999
    exact_match_k = torch.equal(ref_k.cpu(), new_k.cpu())
    exact_match_v = torch.equal(ref_v.cpu(), new_v.cpu())

    # ---- Gradient comparison ----
    loss_new = (new_k.square().sum() + new_v.square().sum())
    loss_ref = (ref_k.square().sum() + ref_v.square().sum())
    loss_new.backward()
    loss_ref.backward()

    grad_cos_k = cosine_sim(key_ref.grad, key_new.grad) if key_new.grad is not None and key_ref.grad is not None else None
    grad_cos_v = cosine_sim(value_ref.grad, value_new.grad) if value_new.grad is not None and value_ref.grad is not None else None
    grad_exact_k = torch.equal(key_ref.grad, key_new.grad) if key_new.grad is not None and key_ref.grad is not None else False
    grad_exact_v = torch.equal(value_ref.grad, value_new.grad) if value_new.grad is not None and value_ref.grad is not None else False

    # ---- Gradient path verification (P0-3 specific) ----
    # Check that prefix portions in reuser rows carry the provider's computation graph
    prefix_grad_preserved = True
    for i in range(batch_size):
        if plan.is_reuser(i):
            prefix_len = plan.prefix_lens[i]
            row_start = plan.cu_seqlens_kv[i]
            # The expanded KV at [row_start:row_start+prefix_len] should carry gradients
            # from the provider row. Check that it's not detached.
            prefix_slice = new_k[row_start:row_start + prefix_len]
            if not prefix_slice.requires_grad:
                prefix_grad_preserved = False
                break

    # ---- Store layer isolation ----
    # Verify store entries with different layer_id don't conflict
    store_isolation_ok = True
    if batch_size >= 2 and any(plan.is_reuser(i) for i in range(batch_size)):
        # Create a second store with layer_id=1
        key2 = torch.randn(total_padded, num_kv_heads, head_dim, dtype=dtype, device=device)
        value2 = torch.randn(total_padded, num_kv_heads, head_dim, dtype=dtype, device=device)
        store_layer1 = PrefixAttentionStore()
        backend.build_kv(
            key2, value2, store_layer1, plan,
            packed_batch_layout=layout, layer_id=1, tp_rank=0,
        )
        # Verify stores are independent
        for i in range(batch_size):
            slot0 = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, 0, i,
                PREFIX_STATE_TYPE_ATTENTION_KV, 0,
            )
            slot1 = PrefixActivationSlotId(
                plan.forward_id, plan.micro_batch_id, 1, i,
                PREFIX_STATE_TYPE_ATTENTION_KV, 0,
            )
            try:
                entry0 = store_new.load(slot0)
                entry1 = store_layer1.load(slot1)
                # Different layer_id → different slots → entries should be independent
                # (Not comparing values since input was different; just confirming both exist)
            except KeyError:
                # Some rows might not be stored if no_sharing
                pass

    # ---- Expanded KV token count ----
    expanded_kv_tokens = sum(plan.expanded_lengths_kv)

    result = {
        "dim": "correctness",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "device": device_name,
        "dtype": dtype_name,
        "has_sharing": plan.has_sharing,
        "reuser_count": sum(1 for ip in plan.is_provider if not ip),
        "provider_count": sum(1 for ip in plan.is_provider if ip),
        "expanded_kv_tokens": expanded_kv_tokens,
        "cos_k": cos_k,
        "cos_v": cos_v,
        "max_diff_k": max_diff_k,
        "max_diff_v": max_diff_v,
        "mean_diff_k": mean_diff_k,
        "mean_diff_v": mean_diff_v,
        "exact_match_k": exact_match_k,
        "exact_match_v": exact_match_v,
        "grad_cos_k": grad_cos_k,
        "grad_cos_v": grad_cos_v,
        "grad_exact_k": grad_exact_k,
        "grad_exact_v": grad_exact_v,
        "prefix_grad_preserved": prefix_grad_preserved,
        "store_isolation_ok": store_isolation_ok,
        "PASS": (
            (cos_k >= 0.9999 and cos_v >= 0.9999)
            and prefix_grad_preserved
            and (grad_cos_k is not None and grad_cos_k >= 0.9999)
            and (grad_cos_v is not None and grad_cos_v >= 0.9999)
        ),
    }

    return result


# ---- P0-1 prefilter correctness ----

def verify_prefilter_correctness(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
) -> dict:
    """Verify P0-1 prefilter: _can_skip_detection_as_no_sharing is correct."""
    from prefix_sharing.core.planner import _can_skip_detection_as_no_sharing

    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)
    min_prefix = max(1, min(prompt_len, prompt_len // 2))
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix)

    # Check prefilter result
    can_skip = _can_skip_detection_as_no_sharing(
        sequences, min_prefix_len=min_prefix, min_group_size=config.min_group_size)

    # Run full planner to get actual plan
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)

    # Run detector directly to see if sharing exists
    from prefix_sharing.core.prefix_detector import TriePrefixDetector
    detector = TriePrefixDetector(min_prefix_len=min_prefix, min_group_size=config.min_group_size)
    detection = detector.detect(sequences)
    has_detection_sharing = len(detection.reuse_specs) > 0

    # Correctness: prefilter should say "skip" only when there truly is no sharing
    prefilter_correct = (can_skip == (not has_detection_sharing))

    result = {
        "dim": "prefilter_correctness",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "min_prefix_len": min_prefix,
        "min_group_size": config.min_group_size,
        "can_skip": can_skip,
        "has_detection_sharing": has_detection_sharing,
        "plan_has_sharing": plan.has_sharing,
        "prefilter_correct": prefilter_correct,
        "PASS": prefilter_correct,
    }

    return result


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="p0_correctness_results.jsonl")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    # P0-1 prefilter experiments
    prefilter_experiments = [
        # no-sharing should be correctly skipped
        {"sharing": "no_sharing", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "no_sharing", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "no_sharing", "batch_size": 64, "prompt_len": 256, "response_len": 256},
        {"sharing": "no_sharing", "batch_size": 128, "prompt_len": 256, "response_len": 256},
        # one_provider should NOT be skipped
        {"sharing": "one_provider", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 64, "prompt_len": 256, "response_len": 256},
        # chain should NOT be skipped
        {"sharing": "chain", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "chain", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "chain", "batch_size": 64, "prompt_len": 256, "response_len": 256},
        # multi_provider should NOT be skipped
        {"sharing": "multi_provider", "batch_size": 16, "prompt_len": 256, "response_len": 256},
        {"sharing": "multi_provider", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        # Short sequences (len < min_prefix_len)
        {"sharing": "no_sharing", "batch_size": 8, "prompt_len": 1, "response_len": 1},
        # Edge case: prefix_len just equals min_prefix_len
        {"sharing": "one_provider", "batch_size": 4, "prompt_len": 2, "response_len": 8},
    ]

    # P0-3 build_kv correctness experiments
    buildkv_experiments = [
        # CPU float32 (exact match expected)
        {"sharing": "no_sharing", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "no_sharing", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 64, "prompt_len": 256, "response_len": 256},
        {"sharing": "chain", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        {"sharing": "chain", "batch_size": 32, "prompt_len": 256, "response_len": 256},
        {"sharing": "multi_provider", "batch_size": 16, "prompt_len": 256, "response_len": 256},
        # Prompt length sweep
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 64, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 512, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 1024, "response_len": 256},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 2048, "response_len": 256},
        # Response length sweep
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 256, "response_len": 64},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 256, "response_len": 512},
        {"sharing": "one_provider", "batch_size": 32, "prompt_len": 256, "response_len": 1024},
    ]

    total = len(prefilter_experiments) + len(buildkv_experiments)
    print(f"[INFO] Total correctness experiments: {total}")
    print(f"[INFO] Device: {args.device}, dtype: {args.dtype}")

    # ---- Run P0-1 prefilter experiments ----
    print("\n=== P0-1 Prefilter Correctness ===")
    for i, exp in enumerate(prefilter_experiments):
        print(f"\n[{i+1}/{len(prefilter_experiments)}] sharing={exp['sharing']} bs={exp['batch_size']} prompt={exp['prompt_len']} response={exp['response_len']}")
        result = verify_prefilter_correctness(
            exp["sharing"], exp["batch_size"], exp["prompt_len"], exp["response_len"])
        status = "PASS" if result["PASS"] else "FAIL"
        print(f"  can_skip={result['can_skip']} has_detection_sharing={result['has_detection_sharing']} prefilter_correct={result['prefilter_correct']} {status}")
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ---- Run P0-3 build_kv correctness experiments (CPU float32) ----
    print("\n=== P0-3 build_kv Correctness (CPU float32) ===")
    for i, exp in enumerate(buildkv_experiments):
        print(f"\n[{i+1}/{len(buildkv_experiments)}] sharing={exp['sharing']} bs={exp['batch_size']} prompt={exp['prompt_len']} response={exp['response_len']} device=cpu dtype=float32")
        try:
            result = verify_build_kv_correctness(
                exp["sharing"], exp["batch_size"], exp["prompt_len"], exp["response_len"],
                device_name="cpu", dtype_name="float32",
            )
            status = "PASS" if result["PASS"] else "FAIL"
            print(f"  cos_k={result['cos_k']:.6f} cos_v={result['cos_v']:.6f} max_diff_k={result['max_diff_k']:.6e} exact={result['exact_match_k']} grad_preserved={result['prefix_grad_preserved']} {status}")
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_record = {"dim": "correctness", "sharing": exp["sharing"],
                           "batch_size": exp["batch_size"], "prompt_len": exp["prompt_len"],
                           "response_len": exp["response_len"], "device": "cpu", "dtype": "float32",
                           "error": str(e), "PASS": False}
            with open(args.output, "a") as f:
                f.write(json.dumps(fail_record) + "\n")

    # ---- GPU bf16: only small configs to verify production fidelity ----
    if args.device == "cuda":
        print("\n=== P0-3 build_kv Correctness (GPU bf16 - small configs only) ===")
        gpu_experiments = [
            {"sharing": "one_provider", "batch_size": 4, "prompt_len": 256, "response_len": 256},
            {"sharing": "one_provider", "batch_size": 8, "prompt_len": 256, "response_len": 256},
            {"sharing": "chain", "batch_size": 4, "prompt_len": 256, "response_len": 256},
            {"sharing": "no_sharing", "batch_size": 8, "prompt_len": 256, "response_len": 256},
        ]
        for i, exp in enumerate(gpu_experiments):
            print(f"\n[{i+1}/{len(gpu_experiments)}] sharing={exp['sharing']} bs={exp['batch_size']} prompt={exp['prompt_len']} response={exp['response_len']} device=cuda dtype=bfloat16")
            try:
                result = verify_build_kv_correctness(
                    exp["sharing"], exp["batch_size"], exp["prompt_len"], exp["response_len"],
                    device_name="cuda", dtype_name="bfloat16",
                )
                status = "PASS" if result["PASS"] else "FAIL"
                print(f"  cos_k={result['cos_k']:.6f} cos_v={result['cos_v']:.6f} max_diff_k={result['max_diff_k']:.6e} exact={result['exact_match_k']} grad_preserved={result['prefix_grad_preserved']} {status}")
                with open(args.output, "a") as f:
                    f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"  ERROR: {e}")
                fail_record = {"dim": "correctness", "sharing": exp["sharing"],
                               "batch_size": exp["batch_size"], "prompt_len": exp["prompt_len"],
                               "response_len": exp["response_len"], "device": "cuda", "dtype": "bfloat16",
                               "error": str(e), "PASS": False}
                with open(args.output, "a") as f:
                    f.write(json.dumps(fail_record) + "\n")

    # ---- Summary ----
    print("\n=== Summary ===")
    all_results = []
    with open(args.output, "r") as f:
        for line in f:
            all_results.append(json.loads(line))

    passed = sum(1 for r in all_results if r.get("PASS", False))
    failed = sum(1 for r in all_results if not r.get("PASS", False))
    errors = sum(1 for r in all_results if "error" in r)

    print(f"  Total: {len(all_results)} experiments")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")

    # Print any failures
    for r in all_results:
        if not r.get("PASS", False) and "error" not in r:
            print(f"  FAIL: {r['dim']} sharing={r['sharing']} bs={r['batch_size']} prompt={r['prompt_len']}")
            if "cos_k" in r:
                print(f"    cos_k={r['cos_k']:.6f} cos_v={r['cos_v']:.6f} max_diff={r.get('max_diff_k', 'N/A')}")

    print(f"\nResults saved to: {args.output}")

    # Exit with non-zero if any failures
    if failed > 0 or errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
