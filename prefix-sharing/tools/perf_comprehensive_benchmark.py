"""PrefixSharing comprehensive performance benchmark.

Covers multiple dimensions:
- prompt_len (shared prefix): 64, 128, 256, 512, 1024, 2048
- response_len (unique suffix): 64, 128, 256, 512, 1024
- batch_size: 4, 8, 16, 32, 64, 128
- sharing pattern: no_sharing, one_provider, multi_provider, chain
- model config: Qwen2.5-0.5B style (14Q/2KV/64D), Qwen3-0.6B style (16Q/8KV/128D)
- backend: flash_atten_gpu, torch_ref

Each experiment records p50/p90/p99 with CUDA synchronize.
Results output as JSONL records.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import torch

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_detector import TriePrefixDetector
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend


# ---- Model configs ----

MODEL_CONFIGS = {
    "qwen2.5-0.5b": {"num_heads": 14, "num_kv_heads": 2, "head_dim": 64, "hidden_dim": 896},
    "qwen3-0.6b": {"num_heads": 16, "num_kv_heads": 8, "head_dim": 128, "hidden_dim": 1024},
}


# ---- Sequence generation ----

def generate_rl_sequences(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    vocab_size: int = 32000,
    seed: int = 42,
) -> list[list[int]]:
    """Generate sequences simulating RL training prompts + responses.

    In RL training:
    - prompt = shared prefix (system + user message)
    - response = unique suffix (model-generated response)
    - seq_len = prompt_len + response_len
    - provider has the full sequence, reusers share the prompt prefix

    Args:
        sharing: "no_sharing" | "one_provider" | "multi_provider" | "chain"
        batch_size: total number of sequences in the batch
        prompt_len: length of shared prefix tokens
        response_len: length of unique suffix tokens per sample
    """
    seq_len = prompt_len + response_len
    counter = vocab_size
    rng_seed = seed

    def _next_token():
        nonlocal counter
        counter += 1
        return counter

    def _make_prompt(base_seed):
        """Create a unique prompt sequence."""
        tokens = []
        v = base_seed
        for _ in range(prompt_len):
            tokens.append(v)
            v += 1
        return tokens

    def _make_response(base_seed):
        """Create a unique response sequence."""
        tokens = []
        v = base_seed
        for _ in range(response_len):
            tokens.append(v)
            v += 1
        return tokens

    if sharing == "no_sharing":
        # All different prompts, detector finds nothing
        sequences = []
        for i in range(batch_size):
            prompt = _make_prompt(10000 + i * (prompt_len + 10))
            response = _make_response(20000 + i * (response_len + 10))
            sequences.append(prompt + response)
        return sequences

    if sharing == "one_provider":
        # 1 provider + (batch_size - 1) reusers sharing the same prompt
        shared_prompt = _make_prompt(10000)
        provider_seq = shared_prompt + _make_response(50000)
        sequences = [provider_seq]
        for i in range(1, batch_size):
            response = _make_response(50000 + i * (response_len + 10))
            sequences.append(shared_prompt + response)
        return sequences

    if sharing == "multi_provider":
        # Multiple different prompts, each with multiple responses
        # e.g. 4 prompts × (batch_size/4) responses each
        num_prompts = max(2, min(4, batch_size // 4))
        num_responses_per_prompt = batch_size // num_prompts
        # If not evenly divisible, add remaining to last prompt
        sequences = []
        for p_idx in range(num_prompts):
            prompt = _make_prompt(10000 + p_idx * (prompt_len + 100))
            # Provider for this prompt group
            provider_seq = prompt + _make_response(50000 + p_idx * 100000)
            sequences.append(provider_seq)
            # Reusers for this prompt group
            for r_idx in range(1, num_responses_per_prompt):
                response = _make_response(50000 + p_idx * 100000 + r_idx * (response_len + 10))
                sequences.append(prompt + response)
        # Pad to batch_size if needed
        while len(sequences) < batch_size:
            prompt = _make_prompt(10000 + num_prompts * (prompt_len + 100))
            response = _make_response(50000 + num_prompts * 100000)
            sequences.append(prompt + response)
        return sequences[:batch_size]

    if sharing == "chain":
        # Chain reuse: first is root, each subsequent shares an extending prefix
        root_prompt = _make_prompt(10000)
        root_response = _make_response(50000)
        sequences = [root_prompt + root_response]
        cumulative_prefix = prompt_len
        for i in range(1, batch_size):
            # Each reuser shares an increasingly longer prefix
            prev_seq = sequences[i - 1]
            # Chain extends: reuser shares (cumulative_prefix) tokens of previous
            actual_prefix_len = min(cumulative_prefix, len(prev_seq))
            shared_prefix = prev_seq[:actual_prefix_len]
            suffix_len = seq_len - actual_prefix_len
            suffix = _make_response(50000 + i * (suffix_len + 10))
            sequences.append(shared_prefix + suffix)
            cumulative_prefix += response_len // 2  # Extend for next
        return sequences

    raise ValueError(f"Unknown sharing pattern: {sharing}")


# ---- Statistics helpers ----

def percentile(lst, p):
    s = sorted(lst)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ---- CPU Overhead Measurement ----

def measure_cpu_overhead(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    num_runs: int = 50,
) -> dict:
    """Measure CPU overhead: detector, planner, trim, nonzero, tolist."""
    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)
    seq_len = prompt_len + response_len
    min_prefix = max(1, min(prompt_len, prompt_len // 2))
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix)

    # Build tensors on GPU for nonzero/tolist timing
    max_len = max(len(s) for s in sequences)
    padded = [s + [0] * (max_len - len(s)) for s in sequences]
    ids_tensor = torch.tensor(padded, dtype=torch.long, device="cuda")
    attn_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device="cuda")

    nonzero_times = []
    tolist_times = []
    detector_times = []
    plan_times = []
    trim_times = []
    peak_py_mbs = []

    for _ in range(num_runs):
        detector = TriePrefixDetector(min_prefix_len=min_prefix, min_group_size=config.min_group_size)
        planner = PrefixSharingPlanner(config, detector=detector)

        # nonzero
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = attn_mask.nonzero()
        torch.cuda.synchronize()
        nonzero_times.append((time.perf_counter() - t0) * 1000)

        # tolist (CPU side)
        t0 = time.perf_counter()
        _ = ids_tensor.detach().cpu().tolist()
        tolist_times.append((time.perf_counter() - t0) * 1000)

        # detector
        tracemalloc.start()
        t0 = time.perf_counter()
        detection = detector.detect(sequences)
        detector_times.append((time.perf_counter() - t0) * 1000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # plan (includes detector re-run inside planner.plan)
        tracemalloc.start()
        t0 = time.perf_counter()
        plan = planner.plan(sequences)
        plan_times.append((time.perf_counter() - t0) * 1000)
        _, peak2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_py_mbs.append(max(peak, peak2) / 1024 / 1024)

        # trim + layout
        t0 = time.perf_counter()
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        trim_times.append((time.perf_counter() - t0) * 1000)

    reused = sum(plan.original_lengths) - sum(plan.kept_lengths_q)
    providers = sum(plan.is_provider)
    reusers = sum(1 for i in range(plan.batch_size) if plan.is_reuser(i))

    # Plan object estimate
    py_obj = plan.batch_size * 14 + len(plan.reuse_specs) * 3 + len(plan.prefix_last_restore) * 5 + 2 * (plan.batch_size + 1)

    return {
        "dim": "cpu",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "total_valid_tokens": sum(plan.original_lengths),
        "reused_tokens": reused,
        "provider_count": providers,
        "reuser_count": reusers,
        "reused_ratio": reused / max(1, sum(plan.original_lengths)),
        "nonzero_ms_p50": percentile(nonzero_times, 50),
        "nonzero_ms_p90": percentile(nonzero_times, 90),
        "nonzero_ms_p99": percentile(nonzero_times, 99),
        "tolist_ms_p50": percentile(tolist_times, 50),
        "tolist_ms_p90": percentile(tolist_times, 90),
        "tolist_ms_p99": percentile(tolist_times, 99),
        "detector_ms_p50": percentile(detector_times, 50),
        "detector_ms_p90": percentile(detector_times, 90),
        "detector_ms_p99": percentile(detector_times, 99),
        "plan_ms_p50": percentile(plan_times, 50),
        "plan_ms_p90": percentile(plan_times, 90),
        "plan_ms_p99": percentile(plan_times, 99),
        "trim_layout_ms_p50": percentile(trim_times, 50),
        "py_objects": py_obj,
        "peak_python_mb": percentile(peak_py_mbs, 50),
    }


# ---- Device Overhead Measurement ----

def measure_device_overhead(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    backend_name: str = "flash_atten_gpu",
    model_name: str = "qwen3-0.6b",
    num_runs: int = 20,
) -> dict:
    """Measure GPU device overhead: build_kv, FA prepare/kernel/post."""
    mc = MODEL_CONFIGS[model_name]
    num_heads = mc["num_heads"]
    num_kv_heads = mc["num_kv_heads"]
    head_dim = mc["head_dim"]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = prompt_len + response_len

    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)
    min_prefix = max(1, min(prompt_len, prompt_len // 2))
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend=backend_name, min_prefix_len=min_prefix)
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend_obj = GpuFlashAttentionBackend() if backend_name == "flash_atten_gpu" else TorchReferenceBackend()

    total_q = sum(plan.kept_lengths_q)
    total_kv_input = layout.total_valid_length

    torch.manual_seed(42)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k_in = torch.randn(total_kv_input, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_in = torch.randn(total_kv_input, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    # Warmup
    for _ in range(3):
        store = PrefixAttentionStore()
        k_exp, v_exp = backend_obj.build_kv(k_in, v_in, store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=0)
        out = backend_obj.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
        del out, k_exp, v_exp, store
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    build_kv_times = []
    fa_prepare_times = []
    fa_kernel_times = []
    fa_post_times = []
    total_times = []

    for _ in range(num_runs):
        store = PrefixAttentionStore()

        # build_kv
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k_exp, v_exp = backend_obj.build_kv(k_in, v_in, store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bk_ms = (t1 - t0) * 1000
        build_kv_times.append(bk_ms)

        expanded_kv_tokens = sum(plan.expanded_lengths_kv)
        expanded_kv_mb = (k_exp.nelement() + v_exp.nelement()) * k_exp.element_size() / 1024 / 1024

        if backend_name == "flash_atten_gpu":
            # FA prepare
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            q_p, k_p, v_p, cu_q, cu_kv, mx_q, mx_kv, pad_lay = backend_obj._prepare_flash_inputs(
                q, k_exp, v_exp, plan, packed_batch_layout=layout
            )
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            prep_ms = (t3 - t2) * 1000
            fa_prepare_times.append(prep_ms)

            # FA kernel
            from flash_attn import flash_attn_varlen_func
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            out = flash_attn_varlen_func(q_p, k_p, v_p, cu_q, cu_kv, mx_q, mx_kv, causal=True)
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            kern_ms = (t5 - t4) * 1000
            fa_kernel_times.append(kern_ms)

            # FA post (repad)
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            if pad_lay is not None:
                out = backend_obj._repad_output(out, pad_lay)
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            post_ms = (t7 - t6) * 1000
            fa_post_times.append(post_ms)

            total_ms = bk_ms + prep_ms + kern_ms + post_ms
        else:
            # TorchRef: measure total attention
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            out = backend_obj.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            attn_ms = (t3 - t2) * 1000
            fa_prepare_times.append(0)
            fa_kernel_times.append(attn_ms)
            fa_post_times.append(0)
            total_ms = bk_ms + attn_ms

        total_times.append(total_ms)
        del out, k_exp, v_exp, store
        gc.collect()
        torch.cuda.empty_cache()

    bk_p50 = percentile(build_kv_times, 50)
    tot_p50 = percentile(total_times, 50)
    tot_p90 = percentile(total_times, 90)

    return {
        "dim": "device",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "backend": backend_name,
        "model": model_name,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "total_q_tokens": total_q,
        "total_kv_input_tokens": total_kv_input,
        "expanded_kv_tokens": expanded_kv_tokens,
        "expanded_kv_mb": expanded_kv_mb,
        "build_kv_ms_p50": bk_p50,
        "build_kv_ms_p90": percentile(build_kv_times, 90),
        "build_kv_pct": bk_p50 / tot_p50 * 100 if tot_p50 > 0 else 0,
        "fa_prepare_ms_p50": percentile(fa_prepare_times, 50),
        "fa_kernel_ms_p50": percentile(fa_kernel_times, 50),
        "fa_post_ms_p50": percentile(fa_post_times, 50),
        "total_attention_ms_p50": tot_p50,
        "total_attention_ms_p90": tot_p90,
        "total_attention_ms_p99": percentile(total_times, 99),
    }


# ---- Memory Overhead Measurement ----

def measure_memory_overhead(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    backend_name: str = "flash_atten_gpu",
    model_name: str = "qwen3-0.6b",
) -> dict:
    """Measure HBM peak with PS enabled vs disabled baseline."""
    mc = MODEL_CONFIGS[model_name]
    num_heads, num_kv_heads, head_dim = mc["num_heads"], mc["num_kv_heads"], mc["head_dim"]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = prompt_len + response_len

    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)
    min_prefix = max(1, min(prompt_len, prompt_len // 2))
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend=backend_name, min_prefix_len=min_prefix)
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    backend_obj = GpuFlashAttentionBackend() if backend_name == "flash_atten_gpu" else TorchReferenceBackend()

    torch.manual_seed(42)
    total_q = sum(plan.kept_lengths_q)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k_in = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_in = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    # --- PS enabled ---
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    store = PrefixAttentionStore()
    k_exp, v_exp = backend_obj.build_kv(k_in, v_in, store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=0)
    expanded_kv_mb = (k_exp.nelement() + v_exp.nelement()) * k_exp.element_size() / 1024 / 1024
    out = backend_obj.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
    torch.cuda.synchronize()
    peak_enabled = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    del out, k_exp, v_exp, store; gc.collect(); torch.cuda.empty_cache()

    # --- PS disabled baseline (no sharing: all full sequences) ---
    total_baseline_q = sum(plan.original_lengths)
    total_baseline_kv = sum(plan.original_lengths)
    torch.cuda.reset_peak_memory_stats(device)
    q_b = torch.randn(total_baseline_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k_b = torch.randn(total_baseline_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_b = torch.randn(total_baseline_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    if backend_name == "flash_atten_gpu":
        from flash_attn import flash_attn_varlen_func
        cu_base = torch.tensor([0] + list(plan.original_lengths), dtype=torch.int32, device=device)
        mx_base = max(plan.original_lengths)
        out_b = flash_attn_varlen_func(q_b, k_b, v_b, cu_base, cu_base, mx_base, mx_base, causal=True)
    else:
        # TorchRef baseline: simple per-row attention without sharing
        q_rows = torch.split(q_b, list(plan.original_lengths))
        k_rows = torch.split(k_b, list(plan.original_lengths))
        v_rows = torch.split(v_b, list(plan.original_lengths))
        outs = []
        for qr, kr, vr in zip(q_rows, k_rows, v_rows):
            import torch.nn.functional as F
            scale = 1.0 / math.sqrt(head_dim)
            q4 = qr.transpose(0,1).unsqueeze(0).contiguous()
            k4 = kr.transpose(0,1).unsqueeze(0).contiguous()
            v4 = vr.transpose(0,1).unsqueeze(0).contiguous()
            o = F.scaled_dot_product_attention(q4, k4, v4, scale=scale, is_causal=True)
            outs.append(o.squeeze(0).transpose(0,1))
        out_b = torch.cat(outs, dim=0)

    torch.cuda.synchronize()
    peak_disabled = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    del q_b, k_b, v_b, out_b; gc.collect(); torch.cuda.empty_cache()

    hbm_saving = peak_disabled - peak_enabled
    hbm_saving_pct = hbm_saving / peak_disabled * 100 if peak_disabled > 0 else 0

    return {
        "dim": "memory",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "backend": backend_name,
        "model": model_name,
        "peak_hbm_enabled_mb": peak_enabled,
        "peak_hbm_disabled_mb": peak_disabled,
        "expanded_kv_mb": expanded_kv_mb,
        "hbm_saving_mb": hbm_saving,
        "hbm_saving_pct": hbm_saving_pct,
        "total_q_tokens_ps": total_q,
        "total_q_tokens_baseline": total_baseline_q,
        "q_reduction_pct": (1 - total_q / total_baseline_q) * 100 if total_baseline_q > 0 else 0,
    }


# ---- Experiment Grid ----

def build_experiments(phase: str) -> list[dict]:
    """Build experiment configurations for each phase."""

    experiments = []

    if phase == "cpu" or phase == "all":
        # CPU overhead: sweep batch_size, prompt_len, response_len, sharing
        prompt_lens = [64, 128, 256, 512, 1024, 2048]
        response_lens = [64, 128, 256, 512, 1024]
        batch_sizes = [4, 8, 16, 32, 64, 128]
        sharings = ["no_sharing", "one_provider", "multi_provider", "chain"]

        # Scenario-based: medium prompt/response with batch sweep
        for bs in batch_sizes:
            for sh in sharings:
                experiments.append({"dim": "cpu", "sharing": sh, "batch_size": bs,
                                    "prompt_len": 256, "response_len": 256})

        # Prompt length sweep (fixed bs=32, response=256)
        for pl in prompt_lens:
            for sh in ["one_provider", "chain", "no_sharing"]:
                experiments.append({"dim": "cpu", "sharing": sh, "batch_size": 32,
                                    "prompt_len": pl, "response_len": 256})

        # Response length sweep (fixed bs=32, prompt=256)
        for rl in response_lens:
            for sh in ["one_provider", "chain"]:
                experiments.append({"dim": "cpu", "sharing": sh, "batch_size": 32,
                                    "prompt_len": 256, "response_len": rl})

        # Extreme: long prompt + short response (common RL scenario)
        for bs in [16, 32, 64]:
            experiments.append({"dim": "cpu", "sharing": "one_provider", "batch_size": bs,
                                "prompt_len": 1024, "response_len": 128})
            experiments.append({"dim": "cpu", "sharing": "one_provider", "batch_size": bs,
                                "prompt_len": 2048, "response_len": 256})

    if phase == "device" or phase == "all":
        # Device overhead: focus on key scenarios with both backends and model configs
        prompt_lens = [64, 128, 256, 512, 1024, 2048]
        response_lens = [64, 128, 256, 512, 1024]
        batch_sizes = [4, 8, 16, 32, 64]
        sharings = ["one_provider", "chain", "multi_provider"]
        backends = ["flash_atten_gpu", "torch_ref"]
        models = ["qwen3-0.6b", "qwen2.5-0.5b"]

        # Core sweep: batch_size × sharing × backend × model (prompt=256, response=256)
        for bs in batch_sizes:
            for sh in sharings:
                for be in backends:
                    for mo in models:
                        experiments.append({"dim": "device", "sharing": sh, "batch_size": bs,
                                            "prompt_len": 256, "response_len": 256,
                                            "backend": be, "model": mo})

        # Prompt length sweep (bs=32, response=256, FA GPU, qwen3)
        for pl in prompt_lens:
            experiments.append({"dim": "device", "sharing": "one_provider", "batch_size": 32,
                                "prompt_len": pl, "response_len": 256,
                                "backend": "flash_atten_gpu", "model": "qwen3-0.6b"})
            # Also torch_ref for comparison at key sizes
            if pl in [128, 512, 2048]:
                experiments.append({"dim": "device", "sharing": "one_provider", "batch_size": 32,
                                    "prompt_len": pl, "response_len": 256,
                                    "backend": "torch_ref", "model": "qwen3-0.6b"})

        # Response length sweep (bs=32, prompt=256, FA GPU, qwen3)
        for rl in response_lens:
            experiments.append({"dim": "device", "sharing": "one_provider", "batch_size": 32,
                                "prompt_len": 256, "response_len": rl,
                                "backend": "flash_atten_gpu", "model": "qwen3-0.6b"})

        # Extreme: long prompt + short response
        for bs in [16, 32, 64]:
            for be in ["flash_atten_gpu"]:
                experiments.append({"dim": "device", "sharing": "one_provider", "batch_size": bs,
                                    "prompt_len": 1024, "response_len": 128,
                                    "backend": be, "model": "qwen3-0.6b"})
                experiments.append({"dim": "device", "sharing": "one_provider", "batch_size": bs,
                                    "prompt_len": 2048, "response_len": 256,
                                    "backend": be, "model": "qwen3-0.6b"})

        # Model config comparison: qwen2.5-0.5b (14Q/2KV/64D) GQA extreme
        for sh in ["one_provider", "chain"]:
            experiments.append({"dim": "device", "sharing": sh, "batch_size": 32,
                                "prompt_len": 256, "response_len": 256,
                                "backend": "flash_atten_gpu", "model": "qwen2.5-0.5b"})

    if phase == "memory" or phase == "all":
        # Memory: key scenarios with both backends and model configs
        batch_sizes = [8, 16, 32, 64]
        prompt_response_pairs = [(64, 64), (128, 128), (256, 256), (512, 512),
                                  (1024, 128), (1024, 512), (2048, 256)]
        sharings = ["one_provider", "chain"]
        backends = ["flash_atten_gpu", "torch_ref"]
        models = ["qwen3-0.6b", "qwen2.5-0.5b"]

        for bs in batch_sizes:
            for pl, rl in prompt_response_pairs:
                for sh in sharings:
                    for be in backends:
                        for mo in models:
                            experiments.append({"dim": "memory", "sharing": sh, "batch_size": bs,
                                                "prompt_len": pl, "response_len": rl,
                                                "backend": be, "model": mo})

    # Deduplicate
    seen = set()
    unique = []
    for e in experiments:
        key = tuple(sorted(e.items()))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["cpu", "device", "memory", "all"])
    parser.add_argument("--output", default="perf_comprehensive_results.jsonl")
    parser.add_argument("--cpu-runs", default=50, type=int)
    parser.add_argument("--device-runs", default=20, type=int)
    args = parser.parse_args()

    experiments = build_experiments(args.phase)
    print(f"[INFO] Total experiments: {len(experiments)}")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    results = []
    for i, exp in enumerate(experiments):
        dim = exp["dim"]
        sharing = exp["sharing"]
        bs = exp["batch_size"]
        pl = exp["prompt_len"]
        rl = exp["response_len"]
        be = exp.get("backend", "flash_atten_gpu")
        mo = exp.get("model", "qwen3-0.6b")
        seq_len = pl + rl

        print(f"\n[{i+1}/{len(experiments)}] {dim}: sharing={sharing} bs={bs} prompt={pl} response={rl} seq={seq_len} backend={be} model={mo}")

        try:
            if dim == "cpu":
                result = measure_cpu_overhead(sharing, bs, pl, rl, num_runs=args.cpu_runs)
            elif dim == "device":
                result = measure_device_overhead(sharing, bs, pl, rl, backend_name=be, model_name=mo, num_runs=args.device_runs)
            elif dim == "memory":
                result = measure_memory_overhead(sharing, bs, pl, rl, backend_name=be, model_name=mo)
            else:
                continue

            # Print summary
            if dim == "cpu":
                print(f"  detector={result['detector_ms_p50']:.2f}ms plan={result['plan_ms_p50']:.2f}ms trim={result['trim_layout_ms_p50']:.2f}ms")
                print(f"  nonzero={result['nonzero_ms_p50']:.2f}ms tolist={result['tolist_ms_p50']:.2f}ms reused={result['reused_tokens']}({result['reused_ratio']:.1%})")
            elif dim == "device":
                print(f"  build_kv={result['build_kv_ms_p50']:.2f}ms({result['build_kv_pct']:.1f}%) fa_prep={result['fa_prepare_ms_p50']:.2f}ms fa_kern={result['fa_kernel_ms_p50']:.2f}ms fa_post={result['fa_post_ms_p50']:.2f}ms total={result['total_attention_ms_p50']:.2f}ms")
            elif dim == "memory":
                print(f"  hbm_enabled={result['peak_hbm_enabled_mb']:.1f}MB hbm_disabled={result['peak_hbm_disabled_mb']:.1f}MB saving={result['hbm_saving_mb']:.1f}MB({result['hbm_saving_pct']:.1f}%) q_reduction={result['q_reduction_pct']:.1f}%")

            results.append(result)
            with open(args.output, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Record failure
            fail_record = {"dim": dim, "sharing": sharing, "batch_size": bs,
                           "prompt_len": pl, "response_len": rl, "seq_len": seq_len,
                           "backend": be, "model": mo, "error": str(e)}
            with open(args.output, "a") as f:
                f.write(json.dumps(fail_record) + "\n")

    print(f"\n=== Complete. {len(results)} successful experiments. Results in {args.output} ===")


if __name__ == "__main__":
    main()
