"""GEMM Precision Baseline — Within-Batch Pairwise Comparison.

Compares N identical copies WITHIN a single forward pass to verify
same-GEMM-kernel bit-identical reproduction.  Reuses ``cmp_diag_verl080``
printing for consistent output.

Usage::

    export PREFIX_SHARING_DIAG_DUMP=/dump_multi
    # forward with batch=[A x N]

    python cmp_baseline_within_batch.py --dir-multi /dump_multi --num-seq 4
"""

from __future__ import annotations

import argparse
import os

import torch

from prefix_sharing.tools.cmp_diag_verl080 import (
    CheckResult,
    _SEP_DOUBLE,
    _SEP_SINGLE,
    _CHECK,
    _CROSS,
    _cosine_sim,
    _pearson_r,
    _dump_json,
    _load_tensor,
    _print_logits_packed,
    _print_2d_result,
    _print_topk_vec,
    _print_topk_2d,
    _print_summary,
    _print_shapes,
)


# ══════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════

def _load_per_layer_dict(directory: str, filename: str) -> dict | None:
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    data = torch.load(filepath, weights_only=True)
    return data if isinstance(data, dict) else None


def _load_cu_seqlens(directory: str) -> torch.Tensor | None:
    filepath = os.path.join(directory, "cu_seqlens_q.pt")
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, weights_only=True)


def _slice_sequence(packed_tensor: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    sequence_index: int) -> torch.Tensor:
    return packed_tensor[int(cu_seqlens[sequence_index]):
                         int(cu_seqlens[sequence_index + 1])]


def _sorted_layer_keys(data: dict) -> list[int]:
    return sorted(int(k) for k in data.keys())


def _group_by_token_count(tensors: list[torch.Tensor]
                          ) -> dict[int, list[torch.Tensor]]:
    """Group tensors by their leading dimension (token count)."""
    groups: dict[int, list[torch.Tensor]] = {}
    for tensor in tensors:
        groups.setdefault(tensor.shape[0], []).append(tensor)
    return groups


# ══════════════════════════════════════════════════════════════════
#  Pairwise comparison within a group of same-length tensors
# ══════════════════════════════════════════════════════════════════

def _pairwise_metrics(copies: list[torch.Tensor]) -> dict:
    """Compare all pairs within a group; return worst max_diff, cos_min,
    and average cos_avg."""
    worst_max_diff = 0.0
    worst_cos_min = 1.0
    all_cos_values: list[float] = []

    for i in range(len(copies)):
        for j in range(i + 1, len(copies)):
            flat_i = copies[i].reshape(copies[i].shape[0], -1).float()
            flat_j = copies[j].reshape(copies[j].shape[0], -1).float()

            per_token_cos = _cosine_sim(flat_i, flat_j, dim=-1)
            all_cos_values.extend(per_token_cos.tolist())
            worst_max_diff = max(worst_max_diff, float((flat_i - flat_j).abs().max()))
            worst_cos_min = min(worst_cos_min, float(per_token_cos.min()))

    return {
        "max_diff": worst_max_diff,
        "cos_avg": sum(all_cos_values) / len(all_cos_values) if all_cos_values else 0.0,
        "cos_min": worst_cos_min,
    }


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — plain per-layer dicts
# ══════════════════════════════════════════════════════════════════

def _compare_plain_within(
    directory: str, filename: str,
    cu_seqlens: torch.Tensor, total_sequences: int,
    filter_layer: int | None, label: str,
) -> CheckResult | None:
    """Pairwise-compare copies within a single dump for ``filename``."""
    data = _load_per_layer_dict(directory, filename)
    if data is None:
        return None

    layers = _sorted_layer_keys(data)
    if filter_layer is not None:
        layers = [l for l in layers if l == filter_layer]
    if not layers:
        return None

    per_layer: dict = {}
    worst_max_diff = 0.0
    worst_cos_min = 1.0

    for layer_index in layers:
        multi_tensor = data[layer_index].float()
        copies = [_slice_sequence(multi_tensor, cu_seqlens, seq_index)
                  for seq_index in range(total_sequences)]
        groups = _group_by_token_count(copies)

        layer_max_diff = 0.0
        layer_cos_min = 1.0
        group_cos_avgs: list[float] = []

        for group_copies in groups.values():
            if len(group_copies) < 2:
                continue
            metrics = _pairwise_metrics(group_copies)
            layer_max_diff = max(layer_max_diff, metrics["max_diff"])
            layer_cos_min = min(layer_cos_min, metrics["cos_min"])
            group_cos_avgs.append(metrics["cos_avg"])

        per_layer[layer_index] = {
            "max_diff": layer_max_diff,
            "cos_avg": (sum(group_cos_avgs) / len(group_cos_avgs)
                        if group_cos_avgs else 0.0),
            "cos_min": layer_cos_min,
            "n_tokens": multi_tensor.shape[0],
            "on_T": multi_tensor.shape[0],
            "off_T": multi_tensor.shape[0],
        }
        worst_max_diff = max(worst_max_diff, layer_max_diff)
        worst_cos_min = min(worst_cos_min, layer_cos_min)

    passed = worst_max_diff == 0.0
    result_name = f"{label}_L{filter_layer}" if filter_layer is not None else label
    return CheckResult(
        name=result_name, passed=passed,
        metrics={"layers": per_layer, "max_diff": worst_max_diff,
                 "cos_min": worst_cos_min, "num_layers": len(layers)},
    )


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — per-layer KV dicts
# ══════════════════════════════════════════════════════════════════

def _compare_kv_within(
    directory: str, filename: str,
    cu_seqlens: torch.Tensor, total_sequences: int,
    filter_layer: int | None, label: str,
    field_first: str, field_second: str,
) -> CheckResult | None:
    """Pairwise-compare ``{layer: {field_first, field_second}}`` within a dump."""
    data = _load_per_layer_dict(directory, filename)
    if data is None:
        return None

    layers = _sorted_layer_keys(data)
    if filter_layer is not None:
        layers = [l for l in layers if l == filter_layer]
    if not layers:
        return None

    per_layer: dict = {}

    for layer_index in layers:
        multi_first = data[layer_index][field_first].float()
        multi_second = data[layer_index][field_second].float()
        first_copies = [_slice_sequence(multi_first, cu_seqlens, seq_index)
                        for seq_index in range(total_sequences)]
        second_copies = [_slice_sequence(multi_second, cu_seqlens, seq_index)
                         for seq_index in range(total_sequences)]

        first_groups = _group_by_token_count(first_copies)
        second_groups = _group_by_token_count(second_copies)

        first_worst = {"max_diff": 0.0, "cos_min": 1.0, "cos_avg": 0.0}
        second_worst = {"max_diff": 0.0, "cos_min": 1.0, "cos_avg": 0.0}
        first_avgs, second_avgs = [], []

        for group in first_groups.values():
            if len(group) >= 2:
                metrics = _pairwise_metrics(group)
                first_worst["max_diff"] = max(first_worst["max_diff"], metrics["max_diff"])
                first_worst["cos_min"] = min(first_worst["cos_min"], metrics["cos_min"])
                first_avgs.append(metrics["cos_avg"])

        for group in second_groups.values():
            if len(group) >= 2:
                metrics = _pairwise_metrics(group)
                second_worst["max_diff"] = max(second_worst["max_diff"], metrics["max_diff"])
                second_worst["cos_min"] = min(second_worst["cos_min"], metrics["cos_min"])
                second_avgs.append(metrics["cos_avg"])

        per_layer[layer_index] = {
            "Q_max_diff": first_worst["max_diff"],
            "K_max_diff": second_worst["max_diff"],
            "Q_cos_avg": sum(first_avgs) / len(first_avgs) if first_avgs else 0.0,
            "Q_cos_min": first_worst["cos_min"],
            "K_cos_avg": sum(second_avgs) / len(second_avgs) if second_avgs else 0.0,
            "K_cos_min": second_worst["cos_min"],
            "n_tokens": sum(g[0].shape[0] * len(g) for g in first_groups.values()),
        }

    result_name = f"{label}_L{filter_layer}" if filter_layer is not None else label
    return CheckResult(name=result_name, passed=True, metrics={"layers": per_layer})


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — logits
# ══════════════════════════════════════════════════════════════════

def _compare_logits_within(
    directory: str,
    cu_seqlens: torch.Tensor,
    total_sequences: int,
) -> CheckResult | None:
    """Pairwise-compare packed logits within a dump."""
    filepath = os.path.join(directory, "logits.pt")
    if not os.path.exists(filepath):
        return None

    multi_logits = torch.load(filepath, weights_only=True).float()
    multi_logits = multi_logits.reshape(-1, multi_logits.size(-1))
    copies = [_slice_sequence(multi_logits, cu_seqlens, seq_index)
              for seq_index in range(total_sequences)]
    groups = _group_by_token_count(copies)

    worst_max_diff = 0.0
    worst_cos_min = 1.0
    all_cos_avgs: list[float] = []

    for group in groups.values():
        if len(group) >= 2:
            metrics = _pairwise_metrics(group)
            worst_max_diff = max(worst_max_diff, metrics["max_diff"])
            worst_cos_min = min(worst_cos_min, metrics["cos_min"])
            all_cos_avgs.append(metrics["cos_avg"])

    return CheckResult(
        name="logits", passed=worst_max_diff == 0.0,
        metrics={
            "n_tokens": copies[0].shape[0] if copies else 0,
            "cos_avg": (sum(all_cos_avgs) / len(all_cos_avgs)
                        if all_cos_avgs else 0.0),
            "cos_min": worst_cos_min,
        },
    )


# ══════════════════════════════════════════════════════════════════
#  Print wrappers (within-batch labels)
# ══════════════════════════════════════════════════════════════════

def _print_table_baseline(result: CheckResult):
    """Print a per-layer comparison table for within-batch."""
    print(_SEP_SINGLE + f"\n  [{result.name}]  within-batch pairwise")
    print(_SEP_SINGLE)
    metrics = result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n")
        return

    layers = metrics.get("layers", {})
    header = (f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS_AVG':>10s} {'COS_MIN':>10s}  "
              f"{'TOKENS':>8s}  {'STATUS':>8s}")
    print(header)
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10} {'─' * 10}  {'─' * 8}  {'─' * 8}")

    for layer_index in sorted(layers):
        entry = layers[layer_index]
        if "max_diff" not in entry:
            print(f"  {layer_index:>6d}  {entry.get('error', '')}")
            continue
        max_diff = entry["max_diff"]
        cos_avg = entry.get("cos_avg", 0.0)
        cos_min = entry.get("cos_min", 1.0)
        tokens = entry.get("n_tokens", "—")
        ok = max_diff == 0.0
        print(f"  {layer_index:>6d}  {max_diff:>12.3e} {cos_avg:>10.6f} {cos_min:>10.6f}  "
              f"{tokens:>8}  {'PASS' if ok else 'DIFF':>8s}")

    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if result.passed else _CROSS}")
    print()


def _print_kv_table_baseline(result: CheckResult, label: str):
    """Print a per-layer Q/K or K/V comparison table for within-batch."""
    print(_SEP_SINGLE + f"\n  [{result.name}]  {label}  within-batch pairwise")
    print(_SEP_SINGLE)
    layers = result.metrics.get("layers")
    if not isinstance(layers, dict):
        return

    header = (f"  {'LAYER':>6s}  {'Q_MAXDIFF':>12s} {'Q_COS_AVG':>12s} {'Q_COS_MIN':>12s}  "
              f"{'K_MAXDIFF':>12s} {'K_COS_AVG':>12s} {'K_COS_MIN':>12s}  "
              f"{'TOKENS':>8s}")
    print(header)
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 12} {'─' * 12}  "
          f"{'─' * 12} {'─' * 12} {'─' * 12}  {'─' * 8}")

    for layer_index in sorted(layers.keys()):
        entry = layers[layer_index]
        if "error" in entry:
            print(f"  {layer_index:>6d}  {entry['error']}")
            continue
        print(f"  {layer_index:>6d}  "
              f"{entry.get('Q_max_diff', 0.0):>12.3e} {entry.get('Q_cos_avg', 0.0):>12.6f} "
              f"{entry.get('Q_cos_min', 0.0):>12.6f}  "
              f"{entry.get('K_max_diff', 0.0):>12.3e} {entry.get('K_cos_avg', 0.0):>12.6f} "
              f"{entry.get('K_cos_min', 0.0):>12.6f}  {entry.get('n_tokens', '—'):>8}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GEMM precision baseline — within-batch pairwise comparison",
        epilog=__doc__,
    )
    parser.add_argument("--dir-multi", required=True,
                        help="Multi-copy dump directory")
    parser.add_argument("--num-seq", type=int, default=None,
                        help="Number of distinct sequences (required for 2D within-batch)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Compare specific layer (1-indexed, default: all)")
    parser.add_argument("--tag", default="old",
                        help="2D file tag for logprobs/entropy (default: old)")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for 2D (default: 1e-5)")
    parser.add_argument("--topk", type=int, default=0,
                        help="top-K worst dims (0=disabled)")
    parser.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                        help="top-K sort order: abs / rel / val")
    parser.add_argument("--output", "-o", default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    cu_seqlens = _load_cu_seqlens(args.dir_multi)
    if cu_seqlens is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing")
        return 1

    total_sequences = cu_seqlens.numel() - 1
    lengths = [int(cu_seqlens[i + 1]) - int(cu_seqlens[i])
               for i in range(total_sequences)]

    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Within-Batch Pairwise Comparison")
    print(f"  Directory: {args.dir_multi}  ({total_sequences} sequences)")
    print(f"  Sequence lengths: {lengths}")
    print(_SEP_DOUBLE)

    _print_shapes(args.dir_multi, args.dir_multi, args.tag)

    all_results: list[CheckResult] = []

    # ── Per-layer plain dicts ──
    for filename, label in [
        ("hidden_states.pt", "hidden_states"),
        ("build_kv_input_v.pt", "build_kv_input_v"),
        ("rope_freqs.pt", "rope_freqs"),
        ("attn_outputs.pt", "attn_outputs"),
    ]:
        result = _compare_plain_within(
            args.dir_multi, filename, cu_seqlens, total_sequences,
            args.layer, label,
        )
        if result:
            all_results.append(result)
            _print_table_baseline(result)

    # ── Per-layer KV dicts ──
    for filename, label, field_a, field_b in [
        ("rope_preqk.pt", "rope_preqk", "query", "key"),
        ("rope_postqk.pt", "rope_postqk", "query", "key"),
        ("full_kv.pt", "full_kv", "key", "value"),
    ]:
        result = _compare_kv_within(
            args.dir_multi, filename, cu_seqlens, total_sequences,
            args.layer, label, field_a, field_b,
        )
        if result:
            all_results.append(result)
            _print_kv_table_baseline(result, label)

    # ── Logits ──
    result = _compare_logits_within(args.dir_multi, cu_seqlens, total_sequences)
    if result:
        all_results.append(result)
        _print_logits_packed(result)

    # ── 2D ──
    _2d_tensors: list[tuple[str, torch.Tensor]] = []
    num_sequences = args.num_seq or total_sequences

    for file_tag, compare_name in [("logprobs", "logp"), ("entropy", "entropy")]:
        filename = f"{file_tag}_{args.tag}.pt"
        tensor_2d = _load_tensor(args.dir_multi, filename)
        if tensor_2d is None or tensor_2d.dim() < 2:
            continue
        tensor_2d = tensor_2d.float()
        batch_size = tensor_2d.shape[0]
        stack = (batch_size // num_sequences
                 if num_sequences > 0 and batch_size % num_sequences == 0 else 1)

        all_abs_diffs: list[float] = []
        all_rel_diffs: list[float] = []
        worst_max_diff = 0.0
        worst_cos_min = 1.0
        worst_rel_max = 0.0

        # Only compare stack copies of the SAME logical sequence
        for seq_index in range(num_sequences):
            if seq_index >= batch_size:
                break
            row_a = tensor_2d[seq_index].reshape(-1)
            for copy_index in range(1, stack):
                row_b_offset = seq_index + copy_index * num_sequences
                if row_b_offset >= batch_size:
                    continue
                row_b = tensor_2d[row_b_offset].reshape(-1)

                abs_diff = (row_a - row_b).abs()
                rel_diff = abs_diff / row_a.abs().clamp(min=1e-8)
                all_abs_diffs.extend(abs_diff.tolist())
                all_rel_diffs.extend(rel_diff.tolist())
                worst_max_diff = max(worst_max_diff, float(abs_diff.max()))
                worst_rel_max = max(worst_rel_max, float(rel_diff.max()))
                worst_cos_min = min(worst_cos_min,
                                    float(_cosine_sim(row_a, row_b, dim=-1)))

        # Pearson on first two stack copies
        pearson_val = 1.0
        if batch_size >= 2 * num_sequences:
            pearson_val = _pearson_r(
                tensor_2d[:num_sequences].float().reshape(-1),
                tensor_2d[num_sequences:2 * num_sequences].float().reshape(-1),
            )

        result = CheckResult(
            name=f"{compare_name}_{args.tag}",
            passed=worst_max_diff == 0.0,
            metrics={
                "shape": tuple(tensor_2d.shape),
                "active": num_sequences * tensor_2d.shape[1],
                "abs_max": worst_max_diff,
                "abs_mean": (sum(all_abs_diffs) / len(all_abs_diffs)
                             if all_abs_diffs else 0.0),
                "rel_max": worst_rel_max,
                "rel_mean": (sum(all_rel_diffs) / len(all_rel_diffs)
                             if all_rel_diffs else 0.0),
                "pearson_r": pearson_val,
                "atol": args.atol,
            },
        )
        all_results.append(result)
        _print_2d_result(result)
        _2d_tensors.append((f"{compare_name}_{args.tag}", tensor_2d))

    # ── Top-K ──
    if args.topk > 0:
        # 2D top-K: compare row 0 vs row num_seq (stack copies of same seq)
        for label, tensor_2d in _2d_tensors:
            ns = (num_sequences if num_sequences > 0
                  and tensor_2d.shape[0] % num_sequences == 0
                  else tensor_2d.shape[0])
            if tensor_2d.shape[0] >= ns + 1:
                _print_topk_2d(tensor_2d[:1].cpu(), tensor_2d[ns:ns + 1].cpu(),
                               None, args.topk, args.sort_err, label)

        # build_kv_input_v top-K
        data = _load_per_layer_dict(args.dir_multi, "build_kv_input_v.pt")
        if data:
            last_layer = max(int(k) for k in data.keys())
            multi_tensor = data[last_layer].float()
            copies = [_slice_sequence(multi_tensor, cu_seqlens, seq_index)
                      for seq_index in range(total_sequences)]
            groups = _group_by_token_count(copies)
            worst_max_diff = 0.0
            worst_a = worst_b = None
            for group in groups.values():
                if len(group) < 2:
                    continue
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        max_diff = float((group[i].reshape(-1) -
                                          group[j].reshape(-1)).abs().max())
                        if max_diff > worst_max_diff:
                            worst_max_diff = max_diff
                            worst_a = group[i].reshape(-1)
                            worst_b = group[j].reshape(-1)
            if worst_a is not None:
                _print_topk_vec(worst_a.cpu(), worst_b.cpu(),
                                args.topk, args.sort_err,
                                f"build_kv_input_v_L{last_layer}_token0")

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, "—", args.dir_multi,
                   tag=f"within_batch_N{total_sequences}", dir_off2=None)


if __name__ == "__main__":
    main()
