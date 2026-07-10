"""Test E: Plan Representation Independent Benefit Verification.

验证目的：拆清 detector 与 plan construction 各自占比，判定 list/dataclass
紧凑化是否值得。

方法：
1. 独立计时 TriePrefixDetector.detect() vs PrefixSharingPlanner.plan_from_detection()
2. 记录 plan_from_detection 内：list 字段数量/元素总数、dataclass 构造耗时、tracemalloc peak
3. 覆盖 bs=32/64/128, prompt=256/1024/2048, sharing=one_provider/no_sharing/chain
4. 50 runs 取 p50/p90/p99

判定规则（来自 impr-perf.md §1.4 测试 E）:
- if plan_from_detection_ms 占 planner 总耗时低 → plan representation 降级
- if 紧凑表示有稳定收益且不牺牲可读性 → 进入开发计划
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from collections import Counter
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_detector import TriePrefixDetector


# ---- Sequence generation ----

def generate_sequences(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
) -> list[list[int]]:
    """Generate sequences simulating RL training prompts + responses."""
    seq_len = prompt_len + response_len
    counter = [20000]  # mutable box for closure

    def _next():
        counter[0] += 1
        return counter[0]

    def _make_prompt(base):
        return [base + j for j in range(prompt_len)]

    def _make_response(base):
        return [base + j for j in range(response_len)]

    if sharing == "no_sharing":
        sequences = []
        for i in range(batch_size):
            seq = _make_prompt(30000 + i * 5000) + _make_response(50000 + i * 5000)
            sequences.append(seq)
        return sequences

    if sharing == "one_provider":
        shared_prompt = _make_prompt(30000)
        provider_seq = shared_prompt + _make_response(50000)
        sequences = [provider_seq]
        for i in range(1, batch_size):
            sequences.append(shared_prompt + _make_response(50000 + i * 500))
        return sequences

    if sharing == "three_provider":
        # Multiple provider groups: 3 prompts × (batch/3) responses
        num_groups = min(3, batch_size // 3)
        group_size = batch_size // num_groups
        sequences = []
        for g in range(num_groups):
            prompt = _make_prompt(30000 + g * 10000)
            provider_seq = prompt + _make_response(50000 + g * 100000)
            sequences.append(provider_seq)
            for r in range(1, group_size):
                sequences.append(prompt + _make_response(50000 + g * 100000 + r * 500))
        while len(sequences) < batch_size:
            sequences.append(_make_prompt(90000 + len(sequences) * 100) + _make_response(50000))
        return sequences[:batch_size]

    if sharing == "chain":
        root_prompt = _make_prompt(30000)
        root_response = _make_response(50000)
        sequences = [root_prompt + root_response]
        cumulative_prefix = prompt_len
        for i in range(1, batch_size):
            prev_seq = sequences[i - 1]
            actual_prefix_len = min(cumulative_prefix, len(prev_seq))
            shared = prev_seq[:actual_prefix_len]
            suffix = [50000 + i * 500 + j for j in range(response_len)]
            while len(shared) + len(suffix) < seq_len:
                suffix.append(50000 + i * 500 + seq_len)
            sequences.append(shared + suffix[:seq_len - len(shared)])
            cumulative_prefix += response_len // 4
        return sequences

    raise ValueError(f"Unknown sharing: {sharing}")


# ---- Test E runner ----

def run_test_e(
    sharing: str,
    batch_size: int,
    prompt_len: int,
    response_len: int,
    num_runs: int = 50,
) -> dict:
    """Run Test E for one experiment configuration.

    Returns dict with all fields needed for the result template.
    """
    sequences = generate_sequences(sharing, batch_size, prompt_len, response_len)
    seq_len = prompt_len + response_len
    min_prefix = max(1, min(prompt_len, prompt_len // 4))
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=min_prefix)

    detector_times = []
    pfd_times = []  # plan_from_detection
    planner_times = []
    list_fields_total = []
    list_elements_total = []
    pfd_peak_mbs = []
    py_objects_est = []

    for run_idx in range(num_runs):
        detector = TriePrefixDetector(min_prefix_len=min_prefix, min_group_size=config.min_group_size)
        planner = PrefixSharingPlanner(config, detector=detector)

        # ---- Step 1: detector.detect() ----
        t0 = time.perf_counter()
        detection = detector.detect(sequences)
        detector_times.append((time.perf_counter() - t0) * 1000)

        # ---- Step 2: plan_from_detection() ----
        tracemalloc.start()
        t0 = time.perf_counter()
        plan = planner.plan_from_detection(sequences, detection)
        t1 = time.perf_counter()
        _, pfd_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pfd_ms = (t1 - t0) * 1000
        pfd_times.append(pfd_ms)
        pfd_peak_mbs.append(pfd_peak / 1024 / 1024)

        # ---- Step 3: field counts ----
        fields = [
            "original_lengths", "reuse_specs", "group_ids", "is_provider",
            "provider_index", "prefix_lens", "suffix_lens", "kept_lengths_q",
            "expanded_lengths_kv", "cu_seqlens_q", "cu_seqlens_kv",
            "q_position_offsets", "kv_position_offsets",
            "input_keep_ranges", "label_keep_ranges", "loss_mask_keep_ranges",
            "prefix_last_restore",
        ]
        total_list_els = 0
        for f_name in fields:
            val = getattr(plan, f_name, None)
            if isinstance(val, (list, tuple)):
                total_list_els += len(val)
                if val and isinstance(val[0], (list, tuple)):
                    total_list_els += sum(len(v) if isinstance(v, (list, tuple)) else 0 for v in val)
        list_fields_total.append(len(fields))
        list_elements_total.append(total_list_els)

        # ---- planner.plan() total (for comparison) ----
        detector2 = TriePrefixDetector(min_prefix_len=min_prefix, min_group_size=config.min_group_size)
        planner2 = PrefixSharingPlanner(config, detector=detector2)
        t0 = time.perf_counter()
        plan2 = planner2.plan(sequences)
        planner_times.append((time.perf_counter() - t0) * 1000)

        # ---- py_objects estimate ----
        py_obj = (
            plan.batch_size * 14
            + len(plan.reuse_specs) * 3
            + len(plan.prefix_last_restore) * 5
            + 2 * (plan.batch_size + 1)
        )
        py_objects_est.append(py_obj)

    def pctl(lst, p):
        s = sorted(lst)
        return s[int(len(s) * p / 100)]

    detector_ms_p50 = pctl(detector_times, 50)
    detector_ms_p90 = pctl(detector_times, 90)
    pfd_ms_p50 = pctl(pfd_times, 50)
    pfd_ms_p90 = pctl(pfd_times, 90)
    planner_ms_p50 = pctl(planner_times, 50)
    pfd_ratio = pfd_ms_p50 / planner_ms_p50 if planner_ms_p50 > 0 else 0

    # Decision logic from impr-perf.md
    if pfd_ratio < 0.2:
        conclusion = f"plan_from_detection仅占planner{pfd_ratio:.0%}，非主要瓶颈，plan representation降级"
    elif pfd_ratio < 0.5:
        conclusion = f"plan_from_detection占{pfd_ratio:.0%}，可适度优化但不紧迫"
    elif pfd_ms_p50 > 50:
        conclusion = f"plan_from_detection占{pfd_ratio:.0%}，绝对耗时{pfd_ms_p50:.1f}ms，紧凑化进入P0/P1"
    else:
        conclusion = f"plan_from_detection占{pfd_ratio:.0%}，绝对耗时较低，暂不优化"

    return {
        "device": "gpu_4090",
        "case": sharing,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": seq_len,
        "min_prefix_len": min_prefix,
        "reused_tokens": sum(plan.original_lengths) - sum(plan.kept_lengths_q),
        "total_valid_tokens": sum(plan.original_lengths),
        "provider_count": sum(plan.is_provider),
        "reuser_count": sum(1 for i in range(plan.batch_size) if plan.is_reuser(i)),
        # detector timing
        "detector_ms_p50": detector_ms_p50,
        "detector_ms_p90": detector_ms_p90,
        "detector_ms_p99": pctl(detector_times, 99),
        # plan_from_detection timing
        "plan_from_detection_ms_p50": pfd_ms_p50,
        "plan_from_detection_ms_p90": pfd_ms_p90,
        "plan_from_detection_ms_p99": pctl(pfd_times, 99),
        # planner.plan() total (reference)
        "planner_plan_ms_p50": planner_ms_p50,
        # ratio
        "pfd_ratio_of_planner": round(pfd_ratio, 3),
        # plan object stats
        "plan_list_fields": pctl(list_fields_total, 50),
        "plan_list_elements": pctl(list_elements_total, 50),
        "plan_py_objects_estimate": pctl(py_objects_est, 50),
        "plan_peak_python_mb": pctl(pfd_peak_mbs, 50),
        # conclusion
        "conclusion": conclusion,
    }


def main():
    parser = argparse.ArgumentParser(description="Test E: Plan Representation Verification")
    parser.add_argument("--output", default="test_e_results.jsonl")
    parser.add_argument("--num-runs", default=50, type=int)
    args = parser.parse_args()

    # Experiment grid
    experiments = []
    batch_sizes = [32, 64, 128]
    prompt_lens = [256, 1024, 2048]
    sharings = ["one_provider", "no_sharing", "chain"]

    for bs in batch_sizes:
        for pl in prompt_lens:
            for sh in sharings:
                experiments.append({"sharing": sh, "batch_size": bs,
                                    "prompt_len": pl, "response_len": 256})

    print(f"Test E: {len(experiments)} experiments, {args.num_runs} runs each")
    print()

    results = []
    for i, exp in enumerate(experiments):
        sh = exp["sharing"]
        bs = exp["batch_size"]
        pl = exp["prompt_len"]
        rl = exp["response_len"]

        print(f"[{i+1}/{len(experiments)}] sharing={sh} bs={bs} prompt={pl} response={rl} seq={pl+rl}")

        result = run_test_e(sh, bs, pl, rl, num_runs=args.num_runs)

        print(f"  detector p50={result['detector_ms_p50']:.2f}ms "
              f"plan_from_detection p50={result['plan_from_detection_ms_p50']:.2f}ms "
              f"planner.plan p50={result['planner_plan_ms_p50']:.2f}ms")
        print(f"  pfd/planner={result['pfd_ratio_of_planner']:.1%} "
              f"list_fields={result['plan_list_fields']} "
              f"list_elements={result['plan_list_elements']} "
              f"py_objects={result['plan_py_objects_estimate']} "
              f"py_peak={result['plan_peak_python_mb']:.2f}MB")
        print(f"  → {result['conclusion']}")

        results.append(result)
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"\n=== Test E complete. {len(results)} results in {args.output} ===")

    # Print summary table
    print()
    print("=" * 120)
    print(f"{'Case':<18} {'Bs':>4} {'P/R':>10} {'detect_p50':>10} {'pfd_p50':>9} {'pln_p50':>9} {'ratio':>6} {'list_el':>7} {'py_obj':>7} {'py_MB':>7}  {'Conclusion'}")
    print("-" * 120)
    for r in results:
        print(f"{r['case']:<18} {r['batch_size']:>4} {r['prompt_len']}/{r['response_len']:<5} "
              f"{r['detector_ms_p50']:>8.1f}ms {r['plan_from_detection_ms_p50']:>7.1f}ms "
              f"{r['planner_plan_ms_p50']:>7.1f}ms "
              f"{r['pfd_ratio_of_planner']:>5.1%} "
              f"{r['plan_list_elements']:>7} {r['plan_py_objects_estimate']:>7} "
              f"{r['plan_peak_python_mb']:>6.2f}  {r['conclusion'][:40]}")
    print("=" * 120)


if __name__ == "__main__":
    main()
