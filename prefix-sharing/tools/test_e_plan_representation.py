"""Test E: Plan Representation Independent Verification.

Purpose: Disentangle detector cost from plan construction cost, and
verify whether the existing prefilter (_can_skip_detection_as_no_sharing)
is effective at skipping no-sharing batches.

Measures:
- prefilter_ms: time spent in _can_skip_detection_as_no_sharing
- detector_ms: time spent in TriePrefixDetector.detect()
- plan_from_detection_ms: time spent in PrefixSharingPlanner.plan_from_detection()
- plan_no_sharing_ms: time spent in PrefixSharingPlanner._plan_no_sharing()
- plan_list_field_count: total number of list elements in the plan
- plan_estimated_py_objects: estimated Python objects in plan
- plan_tracemalloc_peak_mb: tracemalloc peak during plan construction

Fills Test E template from impr-perf.md Section 1.4.2.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner, _can_skip_detection_as_no_sharing
from prefix_sharing.core.prefix_detector import TriePrefixDetector, PrefixDetectionResult


def percentile(lst, p):
    s = sorted(lst)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def generate_rl_sequences(sharing, batch_size, prompt_len, response_len,
                         vocab_size=32000, seed=42):
    """Generate sequences for benchmark (same as comprehensive benchmark)."""
    seq_len = prompt_len + response_len
    counter = vocab_size

    def _next():
        nonlocal counter
        counter += 1
        return counter

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


def count_plan_objects(plan):
    """Estimate total Python object count in a PrefixSharingPlan."""
    total = 0
    # Per-row list fields (14 lists of length batch_size)
    list_fields = [
        plan.original_lengths, plan.group_ids, plan.is_provider,
        plan.provider_index, plan.prefix_lens, plan.suffix_lens,
        plan.kept_lengths_q, plan.expanded_lengths_kv,
        plan.q_position_offsets, plan.kv_position_offsets,
        plan.input_keep_ranges, plan.label_keep_ranges,
        plan.loss_mask_keep_ranges,
    ]
    for field in list_fields:
        total += len(field)  # each element is one Python object

    # cu_seqlens (batch_size + 1 elements each, 2 lists)
    total += len(plan.cu_seqlens_q) + len(plan.cu_seqlens_kv)

    # reuse_specs (each is a frozen dataclass with 3 fields)
    total += len(plan.reuse_specs) * 4  # dataclass object + 3 fields

    # prefix_last_restore specs (each with 5 fields)
    total += len(plan.prefix_last_restore) * 6  # dataclass + 5 fields

    return total


def count_plan_list_elements(plan):
    """Count total number of elements across all list fields in plan."""
    total = 0
    list_fields = [
        plan.original_lengths, plan.group_ids, plan.is_provider,
        plan.provider_index, plan.prefix_lens, plan.suffix_lens,
        plan.kept_lengths_q, plan.expanded_lengths_kv,
        plan.q_position_offsets, plan.kv_position_offsets,
        plan.input_keep_ranges, plan.label_keep_ranges,
        plan.loss_mask_keep_ranges,
    ]
    for field in list_fields:
        total += len(field)
    total += len(plan.cu_seqlens_q) + len(plan.cu_seqlens_kv)
    return total


def measure_test_e(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    num_runs: int = 50,
) -> dict:
    """Run Test E: split detector vs plan_from_detection timing."""
    sequences = generate_rl_sequences(sharing, batch_size, prompt_len, response_len)
    seq_len = prompt_len + response_len
    min_prefix = max(1, min(prompt_len, prompt_len // 2))
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix)

    prefilter_times = []
    detector_times = []
    plan_from_detection_times = []
    plan_no_sharing_times = []
    full_plan_times = []
    peak_mbs = []
    skipped_count = 0

    for _ in range(num_runs):
        planner = PrefixSharingPlanner(config)

        # Step 1: prefilter check
        tracemalloc.start()
        t0 = time.perf_counter()
        can_skip = _can_skip_detection_as_no_sharing(
            sequences, min_prefix_len=min_prefix, min_group_size=config.min_group_size)
        t1 = time.perf_counter()
        _, peak1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        prefilter_times.append((t1 - t0) * 1000)

        if can_skip:
            skipped_count += 1
            # Step 2a: _plan_no_sharing path
            tracemalloc.start()
            t0 = time.perf_counter()
            plan = planner._plan_no_sharing(sequences)
            t1 = time.perf_counter()
            _, peak2 = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            plan_no_sharing_times.append((t1 - t0) * 1000)
            detector_times.append(0)
            plan_from_detection_times.append(0)
            full_plan_times.append(prefilter_times[-1] + plan_no_sharing_times[-1])
            peak_mbs.append(max(peak1, peak2) / 1024 / 1024)
        else:
            # Step 2b: full detector
            detector = TriePrefixDetector(min_prefix_len=min_prefix, min_group_size=config.min_group_size)
            tracemalloc.start()
            t0 = time.perf_counter()
            detection = detector.detect(sequences)
            t1 = time.perf_counter()
            _, peak_det = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            detector_times.append((t1 - t0) * 1000)

            # Step 3: plan_from_detection
            tracemalloc.start()
            t0 = time.perf_counter()
            plan = planner.plan_from_detection(sequences, detection)
            t1 = time.perf_counter()
            _, peak_plan = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            plan_from_detection_times.append((t1 - t0) * 1000)
            plan_no_sharing_times.append(0)
            full_plan_times.append(prefilter_times[-1] + detector_times[-1] + plan_from_detection_times[-1])
            peak_mbs.append(max(peak1, peak_det, peak_plan) / 1024 / 1024)

    py_objects = count_plan_objects(plan)
    list_elements = count_plan_list_elements(plan)
    reused = sum(plan.original_lengths) - sum(plan.kept_lengths_q)

    return {
        "dim": "test_e",
        "sharing": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "min_prefix_len": min_prefix,
        "prefilter_ms_p50": percentile(prefilter_times, 50),
        "prefilter_ms_p90": percentile(prefilter_times, 90),
        "prefilter_skip_pct": skipped_count / num_runs * 100,
        "detector_ms_p50": percentile(detector_times, 50) if any(t > 0 for t in detector_times) else 0,
        "detector_ms_p90": percentile([t for t in detector_times if t > 0], 90) if any(t > 0 for t in detector_times) else 0,
        "plan_from_detection_ms_p50": percentile(plan_from_detection_times, 50) if any(t > 0 for t in plan_from_detection_times) else 0,
        "plan_from_detection_ms_p90": percentile([t for t in plan_from_detection_times if t > 0], 90) if any(t > 0 for t in plan_from_detection_times) else 0,
        "plan_no_sharing_ms_p50": percentile(plan_no_sharing_times, 50) if any(t > 0 for t in plan_no_sharing_times) else 0,
        "plan_no_sharing_ms_p90": percentile([t for t in plan_no_sharing_times if t > 0], 90) if any(t > 0 for t in plan_no_sharing_times) else 0,
        "full_plan_ms_p50": percentile(full_plan_times, 50),
        "full_plan_ms_p90": percentile(full_plan_times, 90),
        "plan_py_objects": py_objects,
        "plan_list_elements": list_elements,
        "plan_tracemalloc_peak_mb": percentile(peak_mbs, 50),
        "reused_tokens": reused,
        "reused_ratio": reused / max(1, sum(plan.original_lengths)),
        "has_sharing": plan.has_sharing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", default=50, type=int)
    parser.add_argument("--output", default="test_e_results.jsonl")
    args = parser.parse_args()

    # Test E experiment grid: bs=32/64/128, prompt=256/1024/2048
    experiments = []
    batch_sizes = [8, 16, 32, 64, 128]
    prompt_lens = [64, 128, 256, 512, 1024, 2048]
    sharings = ["no_sharing", "one_provider", "chain"]

    for bs in batch_sizes:
        for pl in [256]:
            for sh in sharings:
                experiments.append({"sharing": sh, "batch_size": bs, "prompt_len": pl, "response_len": 256})

    for pl in prompt_lens:
        for sh in sharings:
            experiments.append({"sharing": sh, "batch_size": 32, "prompt_len": pl, "response_len": 256})

    for rl in [64, 128, 256, 512, 1024]:
        for sh in ["one_provider", "chain"]:
            experiments.append({"sharing": sh, "batch_size": 32, "prompt_len": 256, "response_len": rl})

    # Deduplicate
    seen = set()
    unique = []
    for e in experiments:
        key = tuple(sorted(e.items()))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    print(f"[INFO] Total Test E experiments: {len(unique)}")

    for i, exp in enumerate(unique):
        sh = exp["sharing"]
        bs = exp["batch_size"]
        pl = exp["prompt_len"]
        rl = exp["response_len"]
        print(f"\n[{i+1}/{len(unique)}] sharing={sh} bs={bs} prompt={pl} response={rl}")

        result = measure_test_e(sh, bs, pl, rl, num_runs=args.num_runs)

        print(f"  prefilter={result['prefilter_ms_p50']:.2f}ms skip={result['prefilter_skip_pct']:.0f}%")
        print(f"  detector={result['detector_ms_p50']:.2f}ms plan_from_det={result['plan_from_detection_ms_p50']:.2f}ms plan_no_share={result['plan_no_sharing_ms_p50']:.2f}ms")
        print(f"  full_plan={result['full_plan_ms_p50']:.2f}ms py_obj={result['plan_py_objects']} list_elem={result['plan_list_elements']} peak={result['plan_tracemalloc_peak_mb']:.2f}MB")
        print(f"  reused={result['reused_tokens']}({result['reused_ratio']:.0%}) has_sharing={result['has_sharing']}")

        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n=== Complete. Results in {args.output} ===")


if __name__ == "__main__":
    main()
