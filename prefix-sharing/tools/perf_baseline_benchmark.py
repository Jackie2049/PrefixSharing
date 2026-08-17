"""PrefixSharing performance baseline benchmark.

This script runs structured performance experiments on real GPU hardware,
measuring CPU overhead (detector/planner), device overhead (build_kv, FA
attention), memory overhead (HBM), and I/O overhead.

Results are output as JSONL records, matching the result template format
from docs/developer-docs/impr-perf.md Section 1.4.

Usage:
    python perf_baseline_benchmark.py [--backend torch_ref|flash_atten_gpu] \
        [--sync 0|1] [--output results.jsonl]

Environment:
    Requires CUDA GPU + flash-attn package for GPU FA backend.
    Runs on any conda env with prefix-sharing installed.
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
from dataclasses import dataclass
from typing import Any, Sequence

import torch

# ---- Import prefix-sharing modules ----
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_detector import TriePrefixDetector
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend


# ---- Data generation ----

def generate_sequences(
    case: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    vocab_size: int = 32000,
    seed: int = 42,
) -> list[list[int]]:
    """Generate synthetic token sequences for benchmark cases.

    All sequences are guaranteed to have exactly seq_len tokens.

    Cases:
        - "disabled": all unique, no prefix sharing possible
        - "no_sharing": all different prefixes, detector finds nothing
        - "one_provider": 1 provider + (batch_size-1) reusers sharing prefix_len tokens
        - "chain": sequential chain reuse (each reuser extends previous prefix)
    """
    rng = _Rng(vocab_size, seed)

    if case == "disabled":
        # Completely unique sequences, no possible sharing
        sequences = []
        for i in range(batch_size):
            start = vocab_size + i * seq_len
            sequences.append(list(range(start, start + seq_len)))
        return sequences

    if case == "no_sharing":
        # Each sequence has unique prefix (first 3 tokens differ)
        # so detector cannot find sharing with min_prefix_len >= 4
        sequences = []
        for i in range(batch_size):
            unique_prefix = [vocab_size + i * 1000 + j for j in range(3)]
            body = [rng.next() for _ in range(seq_len - 3)]
            sequences.append(unique_prefix + body)
        return sequences

    if case == "one_provider":
        # 1 provider with full sequence, rest share prefix_len tokens
        provider_seq = [rng.next() for _ in range(seq_len)]
        sequences = [provider_seq]
        for i in range(1, batch_size):
            shared_prefix = provider_seq[:prefix_len]
            suffix = [rng.next() for _ in range(seq_len - prefix_len)]
            sequences.append(shared_prefix + suffix)
        return sequences

    if case == "chain":
        # Chain reuse: seq[0] is root provider, each subsequent extends
        root = [rng.next() for _ in range(seq_len)]
        sequences = [root]
        cumulative_prefix = prefix_len
        for i in range(1, batch_size):
            # Each reuser shares increasingly longer prefix with previous
            provider = sequences[i - 1]
            actual_prefix = min(cumulative_prefix, len(provider))
            shared_prefix = provider[:actual_prefix]
            suffix = [rng.next() for _ in range(seq_len - actual_prefix)]
            sequences.append(shared_prefix + suffix)
            cumulative_prefix += prefix_len // 2  # Extend prefix for next
        return sequences

    raise ValueError(f"Unknown case: {case}")


class _Rng:
    """Simple deterministic token generator."""
    def __init__(self, start: int, seed: int):
        self._counter = start
        torch.manual_seed(seed)

    def next(self) -> int:
        self._counter += 1
        return self._counter


# ---- CPU Overhead Measurement ----

@dataclass
class CpuOverheadResult:
    device: str
    pipeline: str
    case: str
    batch_size: int
    seq_len: int
    total_valid_tokens: int
    reused_tokens: int
    provider_count: int
    reuser_count: int
    nonzero_ms_p50: float
    nonzero_ms_p90: float
    tolist_ms_p50: float
    tolist_ms_p90: float
    detector_ms_p50: float
    detector_ms_p90: float
    plan_construct_ms_p50: float
    plan_construct_ms_p90: float
    trim_layout_ms_p50: float
    trim_layout_ms_p90: float
    py_objects_estimate: int
    peak_python_mb: float
    conclusion: str


def measure_cpu_overhead(
    case: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    num_runs: int = 50,
    sync: bool = False,
) -> CpuOverheadResult:
    """Measure CPU overhead of detector/planner/trim pipeline.

    Steps measured:
    1. attention_mask.nonzero() (simulated)
    2. .detach().cpu().tolist() (simulated)
    3. TriePrefixDetector.detect()
    4. PrefixSharingPlanner.plan_from_detection() / plan construction
    5. trim + layout build
    """
    sequences = generate_sequences(case, batch_size, seq_len, prefix_len)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=max(1, prefix_len // 4))
    detector = TriePrefixDetector(min_prefix_len=config.min_prefix_len, min_group_size=config.min_group_size)

    nonzero_times = []
    tolist_times = []
    detector_times = []
    plan_construct_times = []
    trim_layout_times = []

    # Generate device-side tensors for nonzero/tolist simulation
    # Pad sequences to same length for 2D tensor
    max_len = max(len(s) for s in sequences)
    padded_sequences = [s + [0] * (max_len - len(s)) for s in sequences]
    input_ids_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    if torch.cuda.is_available():
        input_ids_tensor = input_ids_tensor.cuda()
    attention_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    if torch.cuda.is_available():
        attention_mask = attention_mask.cuda()

    for run_idx in range(num_runs):
        # --- nonzero timing ---
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        nonzero_result = attention_mask.nonzero()
        t1 = time.perf_counter()
        nonzero_times.append((t1 - t0) * 1000)

        # --- tolist timing ---
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_list = input_ids_tensor.detach().tolist()
        t1 = time.perf_counter()
        tolist_times.append((t1 - t0) * 1000)

        # --- detector timing ---
        tracemalloc.start()
        t0 = time.perf_counter()
        detection = detector.detect(sequences)
        t1 = time.perf_counter()
        _, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        detector_times.append((t1 - t0) * 1000)

        # --- plan construct timing ---
        planner = PrefixSharingPlanner(config, detector=detector)
        tracemalloc.start()
        t0 = time.perf_counter()
        plan = planner.plan(sequences)
        t1 = time.perf_counter()
        _, peak2_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        plan_construct_times.append((t1 - t0) * 1000)

        # --- trim + layout timing ---
        tracemalloc.start()
        t0 = time.perf_counter()
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        t1 = time.perf_counter()
        _, peak3_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trim_layout_times.append((t1 - t0) * 1000)

        # Reset detector for next run
        detector = TriePrefixDetector(min_prefix_len=config.min_prefix_len, min_group_size=config.min_group_size)

    def p50(lst):
        s = sorted(lst)
        return s[len(s) // 2]

    def p90(lst):
        s = sorted(lst)
        return s[int(len(s) * 0.9)]

    # Estimate Python objects in plan
    py_objects = (
        plan.batch_size * 14  # per-row list fields
        + len(plan.reuse_specs) * 3  # each spec has 3 fields
        + len(plan.prefix_last_restore) * 5  # each restore spec
        + 2 * (plan.batch_size + 1)  # cu_seqlens
    )

    reused_tokens = plan.original_tokens - sum(plan.kept_lengths_q) if hasattr(plan, 'original_tokens') else sum(plan.original_lengths) - sum(plan.kept_lengths_q)

    return CpuOverheadResult(
        device="gpu_4090",
        pipeline="standalone",
        case=case,
        batch_size=batch_size,
        seq_len=seq_len,
        total_valid_tokens=sum(plan.original_lengths),
        reused_tokens=reused_tokens,
        provider_count=sum(plan.is_provider),
        reuser_count=sum(1 for i in range(plan.batch_size) if plan.is_reuser(i)),
        nonzero_ms_p50=p50(nonzero_times),
        nonzero_ms_p90=p90(nonzero_times),
        tolist_ms_p50=p50(tolist_times),
        tolist_ms_p90=p90(tolist_times),
        detector_ms_p50=p50(detector_times),
        detector_ms_p90=p90(detector_times),
        plan_construct_ms_p50=p50(plan_construct_times),
        plan_construct_ms_p90=p90(plan_construct_times),
        trim_layout_ms_p50=p50(trim_layout_times),
        trim_layout_ms_p90=p90(trim_layout_times),
        py_objects_estimate=py_objects,
        peak_python_mb=max(peak_py, peak2_py, peak3_py) / 1024 / 1024,
        conclusion="",
    )


# ---- Device Overhead Measurement ----

@dataclass
class DeviceOverheadResult:
    device: str
    pipeline: str
    backend: str
    case: str
    total_attention_ms_p50: float
    total_attention_ms_p90: float
    rope_ms: float
    build_kv_ms: float
    build_kv_pct: float
    fa_prepare_ms: float
    fa_kernel_ms: float
    fa_post_ms: float
    restore_proj_ms: float
    expanded_kv_tokens: int
    conclusion: str


def measure_device_overhead(
    case: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    backend_name: str = "flash_atten_gpu",
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_runs: int = 50,
    sync: bool = True,
) -> DeviceOverheadResult:
    """Measure device overhead of build_kv + FA attention on GPU.

    Steps measured (all with CUDA synchronize):
    1. build_kv() — split, store/load, cat, final cat
    2. FA prepare — _prepare_flash_inputs (strip padding, build cu_seqlens)
    3. FA kernel — flash_attn_varlen_func
    4. FA post — _repad_output
    """
    device = torch.device("cuda")

    sequences = generate_sequences(case, batch_size, seq_len, prefix_len)
    config = PrefixSharingConfig(
        enable_prefix_sharing=True,
        backend=backend_name,
        min_prefix_len=max(1, prefix_len // 4),
    )
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    # Create backend
    if backend_name == "flash_atten_gpu":
        backend = GpuFlashAttentionBackend()
    else:
        backend = TorchReferenceBackend()

    # Generate random QKV tensors on GPU
    total_q = sum(plan.kept_lengths_q)
    total_kv = sum(plan.expanded_lengths_kv)

    torch.manual_seed(42)
    dtype = torch.bfloat16
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    # For build_kv, we need K/V in packed format matching layout
    k_input = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_input = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    # Warmup
    store = PrefixAttentionStore()
    for _ in range(3):
        k_exp, v_exp = backend.build_kv(
            k_input, v_input, store, plan,
            packed_batch_layout=layout,
            layer_id=0, tp_rank=0,
        )
        if backend_name == "flash_atten_gpu":
            out = backend.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
        else:
            out = backend.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
        del out, k_exp, v_exp
        store.close()
        store = PrefixAttentionStore()
        torch.cuda.synchronize()

    build_kv_times = []
    fa_prepare_times = []
    fa_kernel_times = []
    fa_post_times = []
    total_times = []

    for run_idx in range(num_runs):
        store = PrefixAttentionStore()

        # --- build_kv timing ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k_exp, v_exp = backend.build_kv(
            k_input, v_input, store, plan,
            packed_batch_layout=layout,
            layer_id=0, tp_rank=0,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        build_kv_ms = (t1 - t0) * 1000
        build_kv_times.append(build_kv_ms)

        # --- FA attention timing ---
        if backend_name == "flash_atten_gpu":
            # Prepare step
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            # Manually call _prepare_flash_inputs for timing
            q_prep, k_prep, v_prep, cu_q, cu_kv, max_q, max_kv, pad_layout = (
                backend._prepare_flash_inputs(q, k_exp, v_exp, plan, packed_batch_layout=layout)
            )
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            fa_prepare_ms = (t3 - t2) * 1000
            fa_prepare_times.append(fa_prepare_ms)

            # Kernel step
            from flash_attn import flash_attn_varlen_func
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            out = flash_attn_varlen_func(
                q_prep, k_prep, v_prep, cu_q, cu_kv,
                max_q, max_kv,
                causal=True,
            )
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            fa_kernel_ms = (t5 - t4) * 1000
            fa_kernel_times.append(fa_kernel_ms)

            # Post step (repad)
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            if pad_layout is not None:
                out = backend._repad_output(out, pad_layout)
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            fa_post_ms = (t7 - t6) * 1000
            fa_post_times.append(fa_post_ms)

            total_ms = build_kv_ms + fa_prepare_ms + fa_kernel_ms + fa_post_ms
            total_times.append(total_ms)
        else:
            # TorchRef: just measure total attention time
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            out = backend.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            fa_prepare_times.append(0)
            fa_kernel_times.append((t3 - t2) * 1000)
            fa_post_times.append(0)
            total_times.append(build_kv_ms + (t3 - t2) * 1000)

        del out, k_exp, v_exp, store
        gc.collect()
        torch.cuda.empty_cache()

    def p50(lst):
        s = sorted(lst)
        return s[len(s) // 2]

    def p90(lst):
        s = sorted(lst)
        return s[int(len(s) * 0.9)]

    build_kv_p50 = p50(build_kv_times)
    total_p50 = p50(total_times)

    return DeviceOverheadResult(
        device="gpu_4090",
        pipeline="standalone",
        backend=backend_name,
        case=case,
        total_attention_ms_p50=total_p50,
        total_attention_ms_p90=p90(total_times),
        rope_ms=0,  # Not measured in this standalone benchmark
        build_kv_ms=build_kv_p50,
        build_kv_pct=(build_kv_p50 / total_p50 * 100) if total_p50 > 0 else 0,
        fa_prepare_ms=p50(fa_prepare_times) if fa_prepare_times else 0,
        fa_kernel_ms=p50(fa_kernel_times),
        fa_post_ms=p50(fa_post_times) if fa_post_times else 0,
        restore_proj_ms=0,  # Not measured in this standalone benchmark
        expanded_kv_tokens=sum(plan.expanded_lengths_kv),
        conclusion="",
    )


# ---- Memory Overhead Measurement ----

@dataclass
class MemoryOverheadResult:
    device: str
    pipeline: str
    backend: str
    case: str
    peak_hbm_enabled_mb: float
    peak_hbm_disabled_mb: float
    expanded_kv_mb: float
    fa_mask_pad_mb: float
    dense_scatter_mb: float
    conclusion: str


def measure_memory_overhead(
    case: str,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    backend_name: str = "flash_atten_gpu",
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> MemoryOverheadResult:
    """Measure HBM memory overhead of prefix-sharing enabled vs disabled.

    Records:
    - peak HBM with PS enabled
    - peak HBM with PS disabled (baseline forward)
    - expanded KV tensor bytes
    - FA mask/pad bytes (GPU FA: negligible, but recorded)
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    sequences = generate_sequences(case, batch_size, seq_len, prefix_len)
    config = PrefixSharingConfig(
        enable_prefix_sharing=True,
        backend=backend_name,
        min_prefix_len=max(1, prefix_len // 4),
    )
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    if backend_name == "flash_atten_gpu":
        backend_obj = GpuFlashAttentionBackend()
    else:
        backend_obj = TorchReferenceBackend()

    torch.manual_seed(42)
    total_q = sum(plan.kept_lengths_q)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k_input = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_input = torch.randn(layout.total_valid_length, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    # --- Enabled measurement ---
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    store = PrefixAttentionStore()
    k_exp, v_exp = backend_obj.build_kv(
        k_input, v_input, store, plan,
        packed_batch_layout=layout,
        layer_id=0, tp_rank=0,
    )
    expanded_kv_mb = (k_exp.nelement() + v_exp.nelement()) * k_exp.element_size() / 1024 / 1024

    out = backend_obj.attention(q, k_exp, v_exp, plan, packed_batch_layout=layout)
    torch.cuda.synchronize()

    peak_enabled = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del out, k_exp, v_exp, store
    gc.collect()
    torch.cuda.empty_cache()

    # --- Disabled measurement (baseline: same total_q but no sharing) ---
    # Baseline: same input size but without prefix sharing
    # This means all sequences compute fully (no KV reuse)
    torch.cuda.reset_peak_memory_stats(device)

    total_baseline_q = sum(plan.original_lengths)
    total_baseline_kv = sum(plan.original_lengths)
    q_base = torch.randn(total_baseline_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k_base = torch.randn(total_baseline_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v_base = torch.randn(total_baseline_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02

    # Simple forward: no prefix-sharing, just varlen FA
    if backend_name == "flash_atten_gpu":
        from flash_attn import flash_attn_varlen_func
        cu_seqlens_base = torch.tensor(
            [0] + list(plan.original_lengths), dtype=torch.int32, device=device
        )
        max_seqlen_base = max(plan.original_lengths)
        out_base = flash_attn_varlen_func(
            q_base, k_base, v_base,
            cu_seqlens_base, cu_seqlens_base,
            max_seqlen_base, max_seqlen_base,
            causal=True,
        )
    else:
        # TorchRef: just do attention without sharing
        out_base = q_base  # placeholder, not meaningful for memory

    torch.cuda.synchronize()
    peak_disabled = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del q_base, k_base, v_base, out_base
    gc.collect()
    torch.cuda.empty_cache()

    return MemoryOverheadResult(
        device="gpu_4090",
        pipeline="standalone",
        backend=backend_name,
        case=case,
        peak_hbm_enabled_mb=peak_enabled,
        peak_hbm_disabled_mb=peak_disabled,
        expanded_kv_mb=expanded_kv_mb,
        fa_mask_pad_mb=0,  # GPU FA doesn't use dense masks
        dense_scatter_mb=0,  # Not applicable in standalone test
        conclusion="",
    )


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="PrefixSharing performance baseline benchmark")
    parser.add_argument("--backend", default="flash_atten_gpu",
                        choices=["torch_ref", "flash_atten_gpu"],
                        help="Attention backend to test")
    parser.add_argument("--sync", default=1, type=int,
                        help="Enable CUDA synchronize for accurate timing (0=off, 1=on)")
    parser.add_argument("--output", default="perf_baseline_results.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--num-runs", default=50, type=int,
                        help="Number of runs per experiment for p50/p90/p99")
    args = parser.parse_args()

    sync = bool(args.sync)
    backend_name = args.backend

    # Experiment configurations matching impr-perf.md PoC parameters
    experiments = [
        # CPU Overhead experiments
        {"dim": "cpu", "case": "no_sharing", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "cpu", "case": "no_sharing", "batch_size": 32, "seq_len": 512, "prefix_len": 384},
        {"dim": "cpu", "case": "one_provider", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "cpu", "case": "one_provider", "batch_size": 32, "seq_len": 512, "prefix_len": 384},
        {"dim": "cpu", "case": "chain", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "cpu", "case": "chain", "batch_size": 32, "seq_len": 512, "prefix_len": 384},

        # Device Overhead experiments
        {"dim": "device", "case": "one_provider", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "device", "case": "one_provider", "batch_size": 32, "seq_len": 512, "prefix_len": 384},
        {"dim": "device", "case": "chain", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "device", "case": "chain", "batch_size": 32, "seq_len": 512, "prefix_len": 384},

        # Memory Overhead experiments
        {"dim": "memory", "case": "one_provider", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "memory", "case": "one_provider", "batch_size": 32, "seq_len": 512, "prefix_len": 384},
        {"dim": "memory", "case": "chain", "batch_size": 8,  "seq_len": 256, "prefix_len": 128},
        {"dim": "memory", "case": "chain", "batch_size": 32, "seq_len": 512, "prefix_len": 384},
    ]

    results = []
    for exp in experiments:
        dim = exp["dim"]
        case = exp["case"]
        bs = exp["batch_size"]
        sl = exp["seq_len"]
        pl = exp["prefix_len"]

        print(f"\n=== {dim}: case={case} batch_size={bs} seq_len={sl} prefix_len={pl} backend={backend_name} ===")

        if dim == "cpu":
            result = measure_cpu_overhead(case, bs, sl, pl, num_runs=args.num_runs, sync=sync)
            print(f"  detector p50={result.detector_ms_p50:.3f}ms p90={result.detector_ms_p90:.3f}ms")
            print(f"  plan_construct p50={result.plan_construct_ms_p50:.3f}ms p90={result.plan_construct_ms_p90:.3f}ms")
            print(f"  trim_layout p50={result.trim_layout_ms_p50:.3f}ms p90={result.trim_layout_ms_p90:.3f}ms")
            print(f"  nonzero p50={result.nonzero_ms_p50:.3f}ms tolist p50={result.tolist_ms_p50:.3f}ms")
            print(f"  py_objects={result.py_objects_estimate} peak_py={result.peak_python_mb:.2f}MB")
            print(f"  reused_tokens={result.reused_tokens} providers={result.provider_count} reusers={result.reuser_count}")
        elif dim == "device":
            result = measure_device_overhead(
                case, bs, sl, pl, backend_name=backend_name,
                num_runs=args.num_runs, sync=sync,
            )
            print(f"  build_kv p50={result.build_kv_ms:.3f}ms ({result.build_kv_pct:.1f}%)")
            print(f"  fa_prepare p50={result.fa_prepare_ms:.3f}ms")
            print(f"  fa_kernel p50={result.fa_kernel_ms:.3f}ms")
            print(f"  fa_post p50={result.fa_post_ms:.3f}ms")
            print(f"  total p50={result.total_attention_ms_p50:.3f}ms p90={result.total_attention_ms_p90:.3f}ms")
            print(f"  expanded_kv_tokens={result.expanded_kv_tokens}")
        elif dim == "memory":
            result = measure_memory_overhead(
                case, bs, sl, pl, backend_name=backend_name,
            )
            print(f"  peak_hbm_enabled={result.peak_hbm_enabled_mb:.2f}MB")
            print(f"  peak_hbm_disabled={result.peak_hbm_disabled_mb:.2f}MB")
            print(f"  expanded_kv={result.expanded_kv_mb:.2f}MB")

        # Add conclusion
        if dim == "cpu":
            total_prepare = result.detector_ms_p50 + result.plan_construct_ms_p50 + result.trim_layout_ms_p50
            if case == "no_sharing" and total_prepare > 5:
                result.conclusion = "no-sharing path expensive and yields 0 benefit; prefilter P0 confirmed"
            elif result.detector_ms_p50 / total_prepare > 0.5:
                result.conclusion = "detector dominates prepare; core detector optimization P0"
            elif result.plan_construct_ms_p50 / total_prepare > 0.3:
                result.conclusion = "plan construction significant; compact representation P0/P1"
            else:
                result.conclusion = "prepare overhead acceptable at this scale"
        elif dim == "device":
            if result.build_kv_pct > 20:
                result.conclusion = f"build_kv accounts for {result.build_kv_pct:.0f}% of attention; prealloc P0 confirmed"
            elif result.fa_prepare_ms > result.fa_kernel_ms * 0.5:
                result.conclusion = "FA prepare approaches or exceeds kernel time; input preparation optimization P0"
            else:
                result.conclusion = f"FA kernel dominates; build_kv ratio {result.build_kv_pct:.0f}% acceptable"
        elif dim == "memory":
            if result.peak_hbm_enabled_mb >= result.peak_hbm_disabled_mb * 0.95:
                result.conclusion = "PS enabled HBM close to baseline; expanded KV or scatter offsets gains"
            else:
                result.conclusion = f"PS enabled saves {result.peak_hbm_disabled_mb - result.peak_hbm_enabled_mb:.1f}MB HBM"

        results.append(result)
        record = {k: v for k, v in result.__dict__.items()}
        with open(args.output, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\n=== All experiments complete. Results written to {args.output} ===")
    print(f"=== Total experiments: {len(results)} ===")


if __name__ == "__main__":
    main()
