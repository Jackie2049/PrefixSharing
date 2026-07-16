"""GEMM Precision Baseline — Cross-Batch-Size Comparison.

Compares the SAME data processed at DIFFERENT batch sizes to quantify
GEMM floating-point noise.  Reuses comparison metrics and printing
functions from ``cmp_diag_verl080`` for consistent output.

Usage::

    # Run 1: single copy
    export PREFIX_SHARING_DIAG_DUMP=/dump_single
    # forward with batch=[A]

    # Run 2: stacked copies
    export PREFIX_SHARING_DIAG_DUMP=/dump_stacked
    # forward with batch=[A x N]

    python cmp_baseline_cross_batch.py \
        --dir-single /dump_single --dir-stacked /dump_stacked --num-copies 4
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
    """Load a ``{layer_index: tensor_or_dict}`` file."""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    data = torch.load(filepath, weights_only=True)
    return data if isinstance(data, dict) else None


def _load_cu_seqlens(directory: str) -> torch.Tensor | None:
    """Load ``cu_seqlens_q.pt`` (cumulative token boundaries)."""
    filepath = os.path.join(directory, "cu_seqlens_q.pt")
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, weights_only=True)


def _slice_sequence(packed_tensor: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    sequence_index: int) -> torch.Tensor:
    """Slice one sequence from a packed tensor using cu_seqlens."""
    start = int(cu_seqlens[sequence_index])
    end = int(cu_seqlens[sequence_index + 1])
    return packed_tensor[start:end]


def _sorted_layer_keys(data: dict) -> list[int]:
    return sorted(int(k) for k in data.keys())


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — plain per-layer dicts
# ══════════════════════════════════════════════════════════════════

def _compare_plain_per_layer(
    dir_single: str, dir_stacked: str,
    filename: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filter_layer: int | None,
    label: str,
) -> CheckResult | None:
    """Compare ``{layer: [total_tokens, ...]}`` across batch sizes.

    Matches each sequence in the single dump against its *stack_count*
    copies in the stacked dump (located at index
    ``seq_index + copy_index * num_sequences``).
    """
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
                diff_i = float((single_flat - multi_flat).abs().max())
                cos_min_i = float(per_token_cos.min())
                layer_max_diff = max(layer_max_diff, diff_i)
                layer_cos_min = min(layer_cos_min, cos_min_i)

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


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — per-layer KV dicts (rope_preqk / rope_postqk /
#  full_kv)
# ══════════════════════════════════════════════════════════════════

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
    """Compare ``{layer: {field_first, field_second}}`` across batch sizes."""
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


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — logits (single packed tensor)
# ══════════════════════════════════════════════════════════════════

def _compare_logits_cross_batch(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
) -> CheckResult | None:
    """Compare packed logits across batch sizes."""
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


# ══════════════════════════════════════════════════════════════════
#  Comparison logic — 2D (logprobs / entropy)
# ══════════════════════════════════════════════════════════════════

def _compare_2d_cross_batch(
    dir_single: str, dir_stacked: str,
    filename: str, label: str,
    stack_count: int, num_sequences: int,
    atol: float = 1e-5,
) -> tuple[CheckResult | None, torch.Tensor | None, torch.Tensor | None]:
    """Compare 2D [batch, L_max] — row ``i`` from single vs rows
    ``i + k * num_sequences`` from stacked."""
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

    # pearson: average over up to 10 pairs
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


# ══════════════════════════════════════════════════════════════════
#  Top-K helpers
# ══════════════════════════════════════════════════════════════════

def _print_topk_plain(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filename: str, label: str,
    topk: int, sort_err: str,
):
    """Print top-K worst dimensions for a plain per-layer file."""
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return

    last_layer = max(int(k) for k in single_data.keys())
    single_tensor = single_data[last_layer].float()
    multi_tensor = multi_data[last_layer].float()
    num_sequences = cu_seqlens_single.numel() - 1

    worst_max_diff = 0.0
    worst_single_flat = worst_multi_flat = None

    for seq_index in range(num_sequences):
        single_seq = _slice_sequence(single_tensor, cu_seqlens_single, seq_index)
        if single_seq.shape[0] == 0:
            continue
        single_flat = single_seq.reshape(single_seq.shape[0], -1)

        for copy_index in range(stack_count):
            multi_offset = seq_index + copy_index * num_sequences
            multi_seq = _slice_sequence(multi_tensor, cu_seqlens_multi, multi_offset)
            if multi_seq.shape[0] != single_seq.shape[0]:
                continue
            multi_flat = multi_seq.reshape(multi_seq.shape[0], -1)
            max_diff = float((single_flat - multi_flat).abs().max())
            if max_diff > worst_max_diff:
                worst_max_diff = max_diff
                worst_single_flat = single_flat
                worst_multi_flat = multi_flat

    if worst_single_flat is not None:
        _print_topk_vec(worst_single_flat[0].cpu(), worst_multi_flat[0].cpu(),
                        topk, sort_err, f"{label}_L{last_layer}_token0")


def _print_topk_kv(
    dir_single: str, dir_stacked: str,
    cu_seqlens_single: torch.Tensor,
    cu_seqlens_multi: torch.Tensor,
    stack_count: int,
    filename: str, field_first: str, field_second: str,
    label: str, topk: int, sort_err: str,
):
    """Print top-K worst dimensions for a KV-style per-layer file."""
    single_data = _load_per_layer_dict(dir_single, filename)
    multi_data = _load_per_layer_dict(dir_stacked, filename)
    if single_data is None or multi_data is None:
        return

    last_layer = max(int(k) for k in single_data.keys())
    num_sequences = cu_seqlens_single.numel() - 1

    for field, tag in [(field_first, f"{label}_{field_first}"),
                       (field_second, f"{label}_{field_second}")]:
        try:
            single_field = single_data[last_layer][field].float()
            multi_field = multi_data[last_layer][field].float()
        except (KeyError, TypeError, AttributeError) as exc:
            print(f"  [top-K skip] {label} {field}: {exc}")
            continue

        worst_max_diff = 0.0
        worst_single_flat = worst_multi_flat = None

        for seq_index in range(num_sequences):
            single_seq = _slice_sequence(single_field, cu_seqlens_single, seq_index)
            if single_seq.shape[0] == 0:
                continue
            single_flat = single_seq.reshape(single_seq.shape[0], -1)

            for copy_index in range(stack_count):
                multi_offset = seq_index + copy_index * num_sequences
                multi_seq = _slice_sequence(multi_field, cu_seqlens_multi, multi_offset)
                if multi_seq.shape[0] != single_seq.shape[0]:
                    continue
                multi_flat = multi_seq.reshape(multi_seq.shape[0], -1)
                max_diff = float((single_flat - multi_flat).abs().max())
                if max_diff > worst_max_diff:
                    worst_max_diff = max_diff
                    worst_single_flat = single_flat
                    worst_multi_flat = multi_flat

        if worst_single_flat is not None:
            _print_topk_vec(worst_single_flat[0].cpu(), worst_multi_flat[0].cpu(),
                            topk, sort_err, f"{tag}_L{last_layer}_token0")


# ══════════════════════════════════════════════════════════════════
#  Print wrappers (Single/Stacked labels instead of ON/OFF)
# ══════════════════════════════════════════════════════════════════

def _print_table_baseline(result: CheckResult):
    """Print a per-layer comparison table with Single/Stacked labels."""
    print(_SEP_SINGLE + f"\n  [{result.name}]  Single vs Stacked（per-seq aligned）")
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
        max_diff = entry["max_diff"]
        cos_avg = entry.get("cos_avg", 0.0)
        cos_min = entry.get("cos_min", 1.0)
        single_tokens = entry.get("on_T", "—")
        stacked_tokens = entry.get("off_T", "—")
        ok = max_diff < 1e-5
        print(f"  {layer_index:>6d}  {max_diff:>12.3e} {cos_avg:>10.6f} {cos_min:>10.6f}  "
              f"{single_tokens:>8} {stacked_tokens:>8}  "
              f"{'OK' if ok else 'DIFF':>8s}")

    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if result.passed else _CROSS}")
    print()


def _print_kv_table_baseline(result: CheckResult, label: str):
    """Print a per-layer Q/K or K/V comparison table."""
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


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GEMM precision baseline — cross-batch-size (single vs N copies)",
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
    print("  GEMM Precision Baseline — Cross-Batch-Size Comparison")
    print(f"  Single :  {args.dir_single}  ({num_sequences} seqs, {total_tokens_single} tokens)")
    print(f"  Stacked:  {args.dir_stacked}  ({num_sequences * stack_count} seqs, "
          f"{total_tokens_single * stack_count} tokens, {stack_count}x stack)")
    print(_SEP_DOUBLE)

    # Shape diagnostics
    _print_shapes(args.dir_single, args.dir_stacked, args.tag)

    all_results: list[CheckResult] = []

    # ── Per-layer plain dicts ──
    for filename, label in [
        ("hidden_states.pt", "hidden_states"),
        ("build_kv_input_v.pt", "build_kv_input_v"),
        ("rope_freqs.pt", "rope_freqs"),
        ("attn_outputs.pt", "attn_outputs"),
    ]:
        result = _compare_plain_per_layer(
            args.dir_single, args.dir_stacked, filename,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
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
        _print_topk_plain(
            args.dir_single, args.dir_stacked,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            "build_kv_input_v.pt", "build_kv_input_v",
            args.topk, args.sort_err,
        )
        _print_topk_kv(
            args.dir_single, args.dir_stacked,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            "rope_preqk.pt", "query", "key", "rope_preqk",
            args.topk, args.sort_err,
        )
        _print_topk_kv(
            args.dir_single, args.dir_stacked,
            cu_seqlens_single, cu_seqlens_multi, stack_count,
            "rope_postqk.pt", "query", "key", "rope_postqk",
            args.topk, args.sort_err,
        )
        for label, single_2d, multi_2d in _2d_tensors:
            if (single_2d.dim() >= 2 and multi_2d.dim() >= 2
                    and single_2d.shape[1] == multi_2d.shape[1]):
                t1 = single_2d[:num_sequences]
                t2 = multi_2d[:num_sequences]
                if t1.shape == t2.shape:
                    _print_topk_2d(t1.cpu(), t2.cpu(), None,
                                   args.topk, args.sort_err, label)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_single, args.dir_stacked,
                   tag=f"cross_batch_N{stack_count}", dir_off2=None)


if __name__ == "__main__":
    main()
