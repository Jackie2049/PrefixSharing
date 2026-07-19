"""GEMM Precision Baseline — Cross-Batch-Size Comparison (FSDP).

Compares the SAME data processed at DIFFERENT batch sizes to quantify
GEMM floating-point noise.  Reuses ``cmp_diag`` printing for consistent output.

Usage::

    # Run 1: single copy
    export PREFIX_SHARING_DIAG_DUMP=/dump_single
    PREFIX_SHARING_BASELINE_STACK=1  python ...

    # Run 2: stacked copies
    export PREFIX_SHARING_DIAG_DUMP=/dump_stacked
    PREFIX_SHARING_BASELINE_STACK=4  python ...

    python cmp_baseline_cross_batch.py \\
        --dir-single /dump_single --dir-stacked /dump_stacked --num-copies 4
"""

from __future__ import annotations

import argparse
import os

import torch

from prefix_sharing.tools.cmp_diag import (  # noqa: E402
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


# ============================================================
#  I/O helpers
# ============================================================

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
    start = int(cu_seqlens[sequence_index])
    end = int(cu_seqlens[sequence_index + 1])
    return packed_tensor[start:end]


def _sorted_layer_keys(data: dict) -> list[int]:
    return sorted(int(k) for k in data.keys())


# ============================================================
#  Comparison — plain per-layer dicts  {layer: [T, hidden]}
# ============================================================

def _compare_plain_per_layer(
    dir_single: str, dir_stacked: str,
    filename: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filter_layer: int | None,
    label: str,
) -> CheckResult | None:
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return None

    layers = _sorted_layer_keys(single_data)
    if filter_layer is not None:
        layers = [l for l in layers if l == filter_layer]
    if not layers:
        return None

    num_sequences = cu_seqlens_single.numel() - 1
    per_layer: dict = {}
    worst_max_diff = 0.0
    worst_cos_min = 1.0

    for layer_index in layers:
        single_tensor = single_data[layer_index].float()
        multi_tensor = multi_data[layer_index].float()
        layer_max_diff = 0.0
        layer_cos_min = 1.0
        all_cos_values: list[float] = []

        for seq_index in range(num_sequences):
            single_seq = _slice_sequence(single_tensor, cu_seqlens_single, seq_index)
            tokens_in_seq = single_seq.shape[0]
            if tokens_in_seq == 0:
                continue
            single_flat = single_seq.reshape(tokens_in_seq, -1)

            for copy_index in range(stack_count):
                multi_offset = seq_index + copy_index * num_sequences
                multi_seq = _slice_sequence(multi_tensor, cu_seqlens_multi, multi_offset)
                if multi_seq.shape[0] != tokens_in_seq:
                    continue
                multi_flat = multi_seq.reshape(tokens_in_seq, -1)

                per_token_cos = _cosine_sim(single_flat, multi_flat, dim=-1)
                all_cos_values.extend(per_token_cos.tolist())
                layer_max_diff = max(layer_max_diff, float((single_flat - multi_flat).abs().max()))
                layer_cos_min = min(layer_cos_min, float(per_token_cos.min()))

        per_layer[layer_index] = {
            "max_diff": layer_max_diff,
            "cos_avg": sum(all_cos_values) / len(all_cos_values) if all_cos_values else 0.0,
            "cos_min": layer_cos_min,
            "n_tokens": single_tensor.shape[0],
            "on_T": single_tensor.shape[0],
            "off_T": multi_tensor.shape[0],
        }
        worst_max_diff = max(worst_max_diff, layer_max_diff)
        worst_cos_min = min(worst_cos_min, layer_cos_min)

    passed = worst_max_diff < 1e-5
    result_name = f"{label}_L{filter_layer}" if filter_layer is not None else label
    return CheckResult(
        name=result_name, passed=passed,
        metrics={"layers": per_layer, "max_diff": worst_max_diff,
                 "cos_min": worst_cos_min, "num_layers": len(layers)},
    )


# ============================================================
#  Comparison — per-layer KV dicts  {layer: {query, key}}
# ============================================================

def _compare_kv_per_layer(
    dir_single: str, dir_stacked: str,
    filename: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filter_layer: int | None,
    label: str,
    field_first: str,
    field_second: str,
) -> CheckResult | None:
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return None

    layers = _sorted_layer_keys(single_data)
    if filter_layer is not None:
        layers = [l for l in layers if l == filter_layer]
    if not layers:
        return None

    num_sequences = cu_seqlens_single.numel() - 1
    per_layer: dict = {}

    for layer_index in layers:
        single_first = single_data[layer_index][field_first].float()
        single_second = single_data[layer_index][field_second].float()
        multi_first = multi_data[layer_index][field_first].float()
        multi_second = multi_data[layer_index][field_second].float()

        first_max_diff, second_max_diff = 0.0, 0.0
        first_cos_min, second_cos_min = 1.0, 1.0
        first_cos_list, second_cos_list = [], []

        for seq_index in range(num_sequences):
            single_seq_f = _slice_sequence(single_first, cu_seqlens_single, seq_index)
            single_seq_s = _slice_sequence(single_second, cu_seqlens_single, seq_index)
            tokens_in_seq = single_seq_f.shape[0]
            if tokens_in_seq == 0:
                continue
            single_flat_f = single_seq_f.reshape(tokens_in_seq, -1)
            single_flat_s = single_seq_s.reshape(tokens_in_seq, -1)

            for copy_index in range(stack_count):
                multi_offset = seq_index + copy_index * num_sequences
                multi_seq_f = _slice_sequence(multi_first, cu_seqlens_multi, multi_offset)
                multi_seq_s = _slice_sequence(multi_second, cu_seqlens_multi, multi_offset)
                if multi_seq_f.shape[0] != tokens_in_seq:
                    continue
                multi_flat_f = multi_seq_f.reshape(tokens_in_seq, -1)
                multi_flat_s = multi_seq_s.reshape(tokens_in_seq, -1)

                cos_f = _cosine_sim(single_flat_f, multi_flat_f, dim=-1)
                cos_s = _cosine_sim(single_flat_s, multi_flat_s, dim=-1)
                first_cos_list.extend(cos_f.tolist())
                second_cos_list.extend(cos_s.tolist())
                first_max_diff = max(first_max_diff, float((single_flat_f - multi_flat_f).abs().max()))
                second_max_diff = max(second_max_diff, float((single_flat_s - multi_flat_s).abs().max()))
                first_cos_min = min(first_cos_min, float(cos_f.min()))
                second_cos_min = min(second_cos_min, float(cos_s.min()))

        per_layer[layer_index] = {
            "Q_max_diff": first_max_diff, "K_max_diff": second_max_diff,
            "Q_cos_avg": sum(first_cos_list) / len(first_cos_list) if first_cos_list else 0.0,
            "Q_cos_min": first_cos_min,
            "K_cos_avg": sum(second_cos_list) / len(second_cos_list) if second_cos_list else 0.0,
            "K_cos_min": second_cos_min,
            "n_tokens": len(first_cos_list),
        }

    result_name = f"{label}_L{filter_layer}" if filter_layer is not None else label
    return CheckResult(name=result_name, passed=True, metrics={"layers": per_layer})


# ============================================================
#  Comparison — logits
# ============================================================

def _compare_logits_cross_batch(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
) -> CheckResult | None:
    single_path = os.path.join(dir_single, "logits.pt")
    multi_path = os.path.join(dir_stacked, "logits.pt")
    if not os.path.exists(single_path) or not os.path.exists(multi_path):
        return None

    single_logits = torch.load(single_path, weights_only=True).float()
    multi_logits = torch.load(multi_path, weights_only=True).float()
    single_logits = single_logits.reshape(-1, single_logits.size(-1))
    multi_logits = multi_logits.reshape(-1, multi_logits.size(-1))

    num_sequences = cu_seqlens_single.numel() - 1
    worst_max_diff = 0.0
    worst_cos_min = 1.0
    all_cos_values: list[float] = []
    total_tokens = 0

    for seq_index in range(num_sequences):
        single_seq = _slice_sequence(single_logits, cu_seqlens_single, seq_index)
        if single_seq.shape[0] == 0:
            continue
        total_tokens += single_seq.shape[0]

        for copy_index in range(stack_count):
            multi_offset = seq_index + copy_index * num_sequences
            multi_seq = _slice_sequence(multi_logits, cu_seqlens_multi, multi_offset)
            if multi_seq.shape[0] != single_seq.shape[0]:
                continue
            per_token_cos = _cosine_sim(single_seq, multi_seq, dim=-1)
            all_cos_values.extend(per_token_cos.tolist())
            worst_max_diff = max(worst_max_diff, float((single_seq - multi_seq).abs().max()))
            worst_cos_min = min(worst_cos_min, float(per_token_cos.min()))

    return CheckResult(
        name="logits", passed=worst_max_diff < 1e-5,
        metrics={
            "n_tokens": total_tokens,
            "cos_avg": sum(all_cos_values) / len(all_cos_values) if all_cos_values else 0.0,
            "cos_min": worst_cos_min,
        },
    )


# ============================================================
#  Comparison — 2D (logprobs / entropy)
# ============================================================

def _compare_2d_cross_batch(
    dir_single: str, dir_stacked: str,
    filename: str, label: str,
    stack_count: int, num_sequences: int,
    atol: float = 1e-5,
) -> tuple[CheckResult | None, torch.Tensor | None, torch.Tensor | None]:
    single_2d = _load_tensor(dir_single, filename)
    multi_2d = _load_tensor(dir_stacked, filename)
    if single_2d is None or multi_2d is None:
        return None, single_2d, multi_2d

    single_2d = single_2d.float()
    multi_2d = multi_2d.float()
    if single_2d.dim() < 2 or multi_2d.dim() < 2:
        return None, single_2d, multi_2d

    worst_max_diff = 0.0
    worst_cos_min = 1.0
    worst_rel_max = 0.0
    all_abs_diffs: list[float] = []
    all_rel_diffs: list[float] = []
    pearson_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for seq_index in range(num_sequences):
        if seq_index >= single_2d.shape[0]:
            break
        single_row = single_2d[seq_index].reshape(-1)

        for copy_index in range(stack_count):
            multi_offset = seq_index + copy_index * num_sequences
            if multi_offset >= multi_2d.shape[0]:
                continue
            multi_row = multi_2d[multi_offset].reshape(-1)

            abs_diff = (single_row - multi_row).abs()
            rel_diff = abs_diff / single_row.abs().clamp(min=1e-8)
            all_abs_diffs.extend(abs_diff.tolist())
            all_rel_diffs.extend(rel_diff.tolist())

            worst_max_diff = max(worst_max_diff, float(abs_diff.max()))
            worst_rel_max = max(worst_rel_max, float(rel_diff.max()))
            worst_cos_min = min(worst_cos_min, float(_cosine_sim(single_row, multi_row, dim=-1)))
            pearson_pairs.append((single_row, multi_row))

    pearson_values = [_pearson_r(a, b) for a, b in pearson_pairs[:10]]
    pearson_avg = (sum(pearson_values) / len(pearson_values)
                   if pearson_values else 0.0)

    result = CheckResult(
        name=label, passed=worst_max_diff <= atol,
        metrics={
            "shape": tuple(single_2d.shape),
            "active": single_2d[:num_sequences].numel(),
            "abs_max": worst_max_diff,
            "abs_mean": sum(all_abs_diffs) / len(all_abs_diffs) if all_abs_diffs else 0.0,
            "rel_max": worst_rel_max,
            "rel_mean": sum(all_rel_diffs) / len(all_rel_diffs) if all_rel_diffs else 0.0,
            "pearson_r": pearson_avg,
            "atol": atol,
        },
    )
    return result, single_2d, multi_2d


# ============================================================
#  Top-K helper — find worst pair (cos_min or rel_max)
# ============================================================

def _worst_pair_by_cos(
    single_data: dict, multi_data: dict,
    cu_seqlens_single: torch.Tensor, cu_seqlens_multi: torch.Tensor,
    num_sequences: int, stack_count: int,
    layer_idx: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, int, int]:
    """Return (single_flat, multi_flat, cos_min, seq, copy) of the pair
    with the smallest per-token cosine across all seq×copy comparisons."""
    single_field = single_data[layer_idx].float()
    multi_field = multi_data[layer_idx].float()
    worst_cos_min = 1.0
    worst_single = worst_multi = None
    w_seq = w_copy = -1

    for seq_index in range(num_sequences):
        single_seq = _slice_sequence(single_field, cu_seqlens_single, seq_index)
        if single_seq.shape[0] == 0:
            continue
        single_flat = single_seq.reshape(single_seq.shape[0], -1)

        for copy_index in range(stack_count):
            multi_seq = _slice_sequence(multi_field, cu_seqlens_multi,
                                        seq_index + copy_index * num_sequences)
            if multi_seq.shape[0] != single_seq.shape[0]:
                continue
            multi_flat = multi_seq.reshape(multi_seq.shape[0], -1)

            cos_vec = _cosine_sim(single_flat, multi_flat, dim=-1)
            cmin = float(cos_vec.min())
            if cmin < worst_cos_min:
                worst_cos_min = cmin
                worst_single = single_flat[cos_vec.argmin()].cpu()
                worst_multi = multi_flat[cos_vec.argmin()].cpu()
                w_seq, w_copy = seq_index, copy_index

    return worst_single, worst_multi, worst_cos_min, w_seq, w_copy


def _worst_pair_by_rel(
    single_2d: torch.Tensor, multi_2d: torch.Tensor,
    num_sequences: int, stack_count: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float, int, int]:
    """Return (single_row, multi_row, rel_max, seq, copy) of the pair
    with the largest relative difference across all seq×copy comparisons."""
    worst_rel_max = 0.0
    worst_single = worst_multi = None
    w_seq = w_copy = -1

    for seq_index in range(num_sequences):
        if seq_index >= single_2d.shape[0]:
            break
        single_row = single_2d[seq_index]

        for copy_index in range(stack_count):
            multi_off = seq_index + copy_index * num_sequences
            if multi_off >= multi_2d.shape[0]:
                continue
            multi_row = multi_2d[multi_off]

            rel_diff = (single_row - multi_row).abs() / torch.maximum(single_row.abs(), multi_row.abs()).clamp(min=1e-8)
            rmax = float(rel_diff.max())
            if rmax > worst_rel_max:
                worst_rel_max = rmax
                worst_single = single_row.cpu()
                worst_multi = multi_row.cpu()
                w_seq, w_copy = seq_index, copy_index

    return worst_single, worst_multi, worst_rel_max, w_seq, w_copy


def _print_topk_plain(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor, cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filename: str, label: str,
    topk: int, sort_err: str,
):
    """Per-layer plain dict: pick the cos_min-worst pair and print top-K dims."""
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return

    last_layer = max(_sorted_layer_keys(single_data))
    num_sequences = cu_seqlens_single.numel() - 1
    a, b, cmin, wseq, wcopy = _worst_pair_by_cos(
        single_data, multi_data, cu_seqlens_single, cu_seqlens_multi,
        num_sequences, stack_count, last_layer,
    )
    if a is not None:
        _print_topk_vec(a, b, topk, sort_err,
                        f"{label}_L{last_layer}_single_seq{wseq}_vs_stacked_copy{wcopy}_cosmin_{cmin:.4f}")


def _print_topk_kv(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor, cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filename: str, field_a: str, field_b: str,
    label: str, topk: int, sort_err: str,
):
    """Per-layer KV dict: pick cos_min-worst pair for each of Q and K."""
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return

    last_layer = max(_sorted_layer_keys(single_data))
    num_sequences = cu_seqlens_single.numel() - 1

    for field, tag in [(field_a, f"{label}_{field_a}"),
                       (field_b, f"{label}_{field_b}")]:
        # Extract just that field into a flat {layer: tensor}
        single_f = {k: v[field] for k, v in single_data.items() if field in v}
        multi_f = {k: v[field] for k, v in multi_data.items() if field in v}
        a, b, cmin, wseq, wcopy = _worst_pair_by_cos(
            single_f, multi_f, cu_seqlens_single, cu_seqlens_multi,
            num_sequences, stack_count, last_layer,
        )
        if a is not None:
            _print_topk_vec(a, b, topk, sort_err,
                            f"{tag}_L{last_layer}_single_seq{wseq}_vs_stacked_copy{wcopy}_cosmin_{cmin:.4f}")


# ============================================================
#  Print wrappers
# ============================================================

def _print_table_baseline(result: CheckResult):
    print(_SEP_SINGLE + f"\n  [{result.name}]  Single vs Stacked (per-seq aligned)")
    print(_SEP_SINGLE)
    metrics = result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n")
        return

    layers = metrics.get("layers", {})
    header = (f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS_AVG':>10s} {'COS_MIN':>10s}  "
              f"{'SNG_T':>8s} {'STK_T':>8s}  {'STATUS':>8s}")
    print(header)
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10} {'─' * 10}  {'─' * 8} {'─' * 8}  {'─' * 8}")

    for layer_index in sorted(layers):
        entry = layers[layer_index]
        if "max_diff" not in entry:
            print(f"  {layer_index:>6d}  {entry.get('error', '')}")
            continue
        ok = entry["max_diff"] < 1e-5
        print(f"  {layer_index:>6d}  {entry['max_diff']:>12.3e} {entry['cos_avg']:>10.6f} "
              f"{entry['cos_min']:>10.6f}  "
              f"{entry['on_T']:>8} {entry['off_T']:>8}  "
              f"{'OK' if ok else 'DIFF':>8s}")

    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if result.passed else _CROSS}")
    print()


def _print_kv_table_baseline(result: CheckResult, label: str):
    print(_SEP_SINGLE + f"\n  [{result.name}]  {label}  Single vs Stacked")
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


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GEMM precision baseline — cross-batch-size (FSDP)",
        epilog=__doc__,
    )
    parser.add_argument("--dir-single", required=True,
                        help="Single-copy dump directory")
    parser.add_argument("--dir-stacked", required=True,
                        help="Stacked-copies dump directory")
    parser.add_argument("--num-copies", type=int, required=True,
                        help="Number of stacked copies (stack count)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Compare specific layer (1-indexed, default: all)")
    parser.add_argument("--tag", default="old",
                        help="2D file tag for logprobs/entropy (default: old)")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for 2D (default: 1e-5)")
    parser.add_argument("--topk", type=int, default=0,
                        help="top-K worst dims for packed token (0=disabled)")
    parser.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                        help="top-K sort order: abs / rel / val")
    parser.add_argument("--output", "-o", default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    cu_seqlens_single = _load_cu_seqlens(args.dir_single)
    cu_seqlens_multi = _load_cu_seqlens(args.dir_stacked)
    if cu_seqlens_single is None or cu_seqlens_multi is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing")
        return 1

    stack_count = args.num_copies
    num_sequences = cu_seqlens_single.numel() - 1
    total_tokens_single = int(cu_seqlens_single[-1])

    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Cross-Batch-Size Comparison (FSDP)")
    print(f"  Single :  {args.dir_single}  ({num_sequences} seqs, {total_tokens_single} tokens)")
    print(f"  Stacked:  {args.dir_stacked}  ({num_sequences * stack_count} seqs, "
          f"{total_tokens_single * stack_count} tokens, {stack_count}x stack)")
    print(_SEP_DOUBLE)

    _print_shapes(args.dir_single, args.dir_stacked, args.tag)

    all_results: list[CheckResult] = []

    # ── Per-layer plain dicts ──
    for filename, label in [
        ("attn_outputs.pt", "attn_outputs"),
        ("build_kv_input_v.pt", "build_kv_input_v"),
        ("expanded_kv.pt", "expanded_kv"),
        ("attn_grads.pt", "attn_grads"),
    ]:
        result = _compare_plain_per_layer(
            args.dir_single, args.dir_stacked, filename,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            args.layer, label,
        )
        if result:
            all_results.append(result)
            _print_table_baseline(result)

    # ── Per-layer KV dicts (Q/K) ──
    for filename, label, field_a, field_b in [
        ("rope_postqk.pt", "rope_postqk", "query", "key"),
    ]:
        result = _compare_kv_per_layer(
            args.dir_single, args.dir_stacked, filename,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            args.layer, label, field_a, field_b,
        )
        if result:
            all_results.append(result)
            _print_kv_table_baseline(result, label)

    # ── Logits ──
    result = _compare_logits_cross_batch(
        args.dir_single, args.dir_stacked,
        cu_seqlens_single, cu_seqlens_multi, stack_count,
    )
    if result:
        all_results.append(result)
        _print_logits_packed(result)

    # ── 2D ──
    _2d_tensors: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for file_tag, compare_name in [("logprobs", "logp"), ("entropy", "entropy")]:
        filename = f"{file_tag}_{args.tag}.pt"
        result, single_2d, multi_2d = _compare_2d_cross_batch(
            args.dir_single, args.dir_stacked, filename,
            f"{compare_name}_{args.tag}", stack_count, num_sequences, args.atol,
        )
        if result:
            all_results.append(result)
            _print_2d_result(result)
        if single_2d is not None and multi_2d is not None:
            _2d_tensors.append((f"{compare_name}_{args.tag}", single_2d, multi_2d))

    # ── Top-K ──
    if args.topk > 0:
        # Per-layer plain dict: cos_min-worst pair → top-K dims
        for _fn, _lb in [("attn_outputs.pt", "attn_outputs"),
                         ("build_kv_input_v.pt", "build_kv_input_v"),
                         ("attn_grads.pt", "attn_grads")]:
            _print_topk_plain(
                args.dir_single, args.dir_stacked,
                cu_seqlens_single, cu_seqlens_multi, stack_count,
                _fn, _lb, args.topk, args.sort_err,
            )
        # Per-layer KV dict (Q/K): cos_min-worst pair each → top-K dims
        _print_topk_kv(
            args.dir_single, args.dir_stacked,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            "rope_postqk.pt", "query", "key", "rope_postqk",
            args.topk, args.sort_err,
        )
        # 2D scalar data: rel_max-worst pair → top-K dims
        for label, single_2d, multi_2d in _2d_tensors:
            if (single_2d.dim() >= 2 and multi_2d.dim() >= 2
                    and single_2d.shape[1] == multi_2d.shape[1]):
                t1, t2, rel, wseq, wcopy = _worst_pair_by_rel(
                    single_2d, multi_2d, num_sequences, stack_count)
                if t1 is not None:
                    _print_topk_2d(t1.unsqueeze(0), t2.unsqueeze(0), None,
                                   args.topk, args.sort_err,
                                   f"{label}_single_seq{wseq}_vs_stacked_copy{wcopy}_relmax_{rel:.4f}")

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_single, args.dir_stacked,
                   tag=f"cross_batch_N{stack_count}", dir_off2=None)


if __name__ == "__main__":
    main()
