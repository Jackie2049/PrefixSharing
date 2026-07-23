"""Multi-rank diagnostic dump assembler — merges PP/TP/DP-sharded files into a
single-card-compatible flat directory.

Reads a raw dump directory produced by the diagnostic dump infrastructure under
TP/PP/DP parallelism (with ``_pp{p}`` / ``_tp{r}`` / ``_dp{r}`` file suffixes)
and assembles a clean flat directory that looks exactly like a single-card dump.
The output can then be fed directly to the **unmodified** ``cmp_diag.py``.

Usage::

    python assemble_dump.py --input-dir /path/to/raw_dump --output-dir /path/to/assembled

Single-card dumps (tp==1, pp==1, dp==1) are a fast path: all files are copied verbatim.

Assembly rules follow the semantic shard axis rather than the tensor rank:

* DP metadata ``prefix_lens [B]`` is concatenated on the batch axis.
* DP cumulative offsets ``cu_seqlens [B+1]`` are rebased before concatenation.
* DP packed per-layer tensors ``[N_rank, ...]`` are concatenated on ``dim=0``.
* DP FSDP logits ``[1, N_rank, V]`` are concatenated on ``dim=1`` so the
  assembled result preserves the single-card ``[1, N_total, V]`` contract.
* DP tagged tensors such as masks and log-probabilities ``[B_rank, L]`` are
  concatenated on the batch axis.
* TP logits are vocabulary shards and are concatenated on the last dimension.
* PP files contain disjoint layer dictionaries, so their layer mappings are
  combined rather than tensor-concatenated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch

# These files are dictionaries keyed by layer number.  Under PP, each stage
# contributes a disjoint subset of layers, so assembly combines dictionary
# entries without concatenating their tensor values.
_PP_STAGE_STEMS: list[str] = [
    "attn_outputs",
    "rope_preqk",
    "rope_postqk",
    "rope_freqs",
    "expanded_kv",
    "full_kv",
    "build_kv_input_v",
    "hidden_states",
]

# Each DP rank processes different samples/tokens and writes one file with a
# ``_dp{rank}`` suffix.  The exact merge rule is selected below by stem:
# cumulative metadata needs offset rebasing, while packed layer tensors use
# their leading token dimension.
_DP_SHARDABLE_STEMS: list[str] = [
    "prefix_lens",
    "cu_seqlens_q",
    "cu_seqlens_q_logits",
    "attn_outputs",
    "rope_postqk",
    "build_kv_input_v",
    "expanded_kv",
    "attn_grads",
]

# These DP artifacts have the outer structure ``{layer_number: value}``.
# A value may be a tensor (attention output/gradient/input V) or another dict
# containing tensor leaves (RoPE Q/K and expanded K/V).
_DP_PER_LAYER_STEMS: set[str] = {
    "attn_outputs",
    "rope_postqk",
    "build_kv_input_v",
    "expanded_kv",
    "attn_grads",
}

# These names contain a runtime tag (for example ``old`` or ``train``), so
# they are discovered by prefix instead of by one fixed stem.  Their tensors
# use the batch-major ``[B_rank, L]`` representation.
_DP_TAG_PREFIXES: list[str] = [
    "attention_mask_",
    "label_mask_",
    "logprobs_",
    "entropy_",
    "input_ids_",
]


def _collect_dp_shards(input_dir: str, base_stem: str) -> list[tuple[int, str]]:
    """Find all DP shards for a given file stem.

    Returns list of (dp_rank, filepath) sorted by dp_rank.
    Matches both ``{stem}.pt`` (no suffix) and ``{stem}_dp{r}.pt``.
    """
    shards: list[tuple[int, str]] = []
    # Single file (dp==1 or rank 0 only)
    plain = os.path.join(input_dir, f"{base_stem}.pt")
    if os.path.exists(plain):
        shards.append((0, plain))

    # DP-sharded files: {stem}_dp0.pt, {stem}_dp1.pt, ...
    for filename in os.listdir(input_dir):
        if not filename.startswith(base_stem) or not filename.endswith(".pt"):
            continue
        # Match {stem}_dp{digit}.pt
        remainder = filename[len(base_stem):]
        if remainder.startswith("_dp") and remainder.endswith(".pt"):
            try:
                rank_str = remainder[3:-3]  # strip "_dp" prefix and ".pt" suffix
                rank = int(rank_str)
                shards.append((rank, os.path.join(input_dir, filename)))
            except ValueError:
                pass
    return sorted(shards)


def _merge_dp_2d(shards: list[tuple[int, str]]) -> torch.Tensor | None:
    """Merge batch-major DP tensors ``[B_rank, L]`` on ``dim=0``.

    The non-batch dimensions must already match across ranks.  The diagnostic
    dump currently uses one shared sequence width; a shape mismatch is left as
    an explicit ``torch.cat`` error instead of silently padding incompatible
    coordinate systems.
    """
    tensors = []
    for _, filepath in shards:
        tensor = torch.load(filepath, weights_only=True)
        if tensor is not None:
            tensors.append(tensor)
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _merge_dp_1d(shards: list[tuple[int, str]], adjust_offsets: bool = False
                 ) -> torch.Tensor | None:
    """Merge one-dimensional DP metadata in rank order.

    ``prefix_lens`` uses ordinary concatenation because every element belongs
    to one sample.  A cumulative sequence array is different: every rank starts
    at zero, so ``adjust_offsets=True`` removes each later rank's duplicate zero
    and rebases its remaining boundaries by the previous rank's final offset.

    Example: ``[0, 3, 8] + [0, 2, 6]`` becomes
    ``[0, 3, 8, 10, 14]``, not ``[0, 3, 8, 0, 2, 6]``.
    """
    tensors = []
    cumulative_base = 0
    for _, filepath in shards:
        tensor = torch.load(filepath, weights_only=True)
        if tensor is None:
            continue
        if adjust_offsets and cumulative_base > 0 and tensors:
            # First element of this shard's cu_seqlens is 0 (start of shard).
            # Add cumulative_base to all entries except the first, then skip
            # the first when concatenating (previous shard's tail = this shard's head).
            tensor = tensor.clone()
            tensor[1:] = tensor[1:] + cumulative_base
            cumulative_base = int(tensor[-1])
            tensors.append(tensor[1:])  # skip duplicate head
        else:
            tensors.append(tensor)
            if adjust_offsets:
                cumulative_base = int(tensor[-1])
    if not tensors:
        return None
    # For non-offset-adjusted: simple concat
    if not adjust_offsets:
        return torch.cat(tensors, dim=0)
    # For offset-adjusted: first shard includes head, rest had their head
    # stripped at append time (tensor[1:]), so a plain concat reconstructs
    # the full cumulative sequence.
    return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]


def _merge_dp_values(values: list[Any]) -> Any:
    """Recursively merge one logical value from multiple DP ranks.

    Per-layer diagnostic files are not uniform: ``attn_outputs`` and
    ``attn_grads`` map layer -> Tensor, while ``rope_postqk`` and
    ``expanded_kv`` map layer -> {name: Tensor | None}.  Tensor leaves are
    concatenated on the token axis (dim 0); nested dictionaries are merged
    recursively.
    """
    non_none = [value for value in values if value is not None]
    # Optional fields such as RoPE positions may be absent on every rank.
    if not non_none:
        return None
    first = non_none[0]
    if isinstance(first, torch.Tensor):
        # A tensor field must be present on every rank.  Mixing Tensor and None
        # would lose the correspondence between the assembled token stream and
        # this field, so fail instead of silently dropping one rank.
        if len(non_none) != len(values):
            raise ValueError("DP shards disagree on optional tensor presence")
        # Per-layer tensors use token-major shapes [N_rank, ...].  Feature
        # dimensions (hidden size, heads, head dim) remain unchanged under DP.
        return torch.cat(non_none, dim=0)
    if isinstance(first, dict):
        # Nested records (for example {query, key, positions}) are structural;
        # recurse until tensor leaves are reached.
        keys = set().union(*(value.keys() for value in non_none))
        return {
            key: _merge_dp_values([value.get(key) for value in values])
            for key in keys
        }
    if all(value == first for value in non_none):
        return first
    raise TypeError(f"Unsupported or inconsistent DP-sharded value type: {type(first).__name__}")


def _merge_dp_logits(shards: list[tuple[int, str]]) -> torch.Tensor:
    """Merge FSDP packed logits while preserving the single-card format.

    FSDP emits logits as ``[1, N_rank, V]``: the leading dimension is a
    singleton packed-sequence container, ``N_rank`` is the number of packed
    tokens handled by this DP rank, and ``V`` is the full vocabulary.  DP
    therefore concatenates on ``dim=1`` and produces ``[1, N_total, V]``.

    This is intentionally strict.  Accepting ``[N, V]`` or concatenating on
    ``dim=0`` would create a format that no longer matches a single-card dump.
    """
    tensors = [torch.load(filepath, weights_only=True) for _, filepath in shards]
    for dp_rank, tensor in zip((rank for rank, _ in shards), tensors):
        if tensor.dim() != 3 or tensor.shape[0] != 1:
            raise ValueError(
                f"logits_dp{dp_rank}.pt must have shape [1, N, V], "
                f"got {tuple(tensor.shape)}"
            )
    vocab_size = tensors[0].shape[2]
    for (dp_rank, _), tensor in zip(shards[1:], tensors[1:]):
        if tensor.shape[2] != vocab_size:
            raise ValueError(
                f"logits_dp{dp_rank}.pt vocab size {tensor.shape[2]} "
                f"does not match {vocab_size}"
            )
    return torch.cat(tensors, dim=1)


def _merge_dp_per_layer_dict(shards: list[tuple[int, str]]) -> dict | None:
    """Merge ``{layer: value}`` DP shards in DP-rank order.

    Every DP rank runs all model layers, so values for the same layer represent
    consecutive segments of the global packed-token stream.  Group those values
    by layer first, then recursively merge tensor leaves on their leading token
    dimension via :func:`_merge_dp_values`.
    """
    by_layer: dict[int, list[Any]] = {}
    for _, filepath in shards:
        layer_dict = torch.load(filepath, weights_only=True)
        if not isinstance(layer_dict, dict):
            continue
        for layer_idx, value in layer_dict.items():
            by_layer.setdefault(layer_idx, []).append(value)
    if not by_layer:
        return None
    return {
        layer_idx: _merge_dp_values(values)
        for layer_idx, values in by_layer.items()
    }


def _collect_dp_tag_files(input_dir: str, prefix: str
                          ) -> dict[str, list[tuple[int, str]]]:
    """Collect DP-sharded tag-suffixed files grouped by full stem.

    Returns {full_stem: [(dp_rank, filepath), ...]}.
    E.g., "logprobs_train" → [(0, "logprobs_train_dp0.pt"), (1, ...)].
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for filename in os.listdir(input_dir):
        if not filename.startswith(prefix) or not filename.endswith(".pt"):
            continue
        # Strip .pt, then check for _dp{digit} suffix
        stem = filename[:-3]
        if "_dp" in stem:
            parts = stem.rsplit("_dp", 1)
            base = parts[0]
            try:
                rank = int(parts[1])
            except ValueError:
                continue
        else:
            base = stem
            rank = 0
        groups.setdefault(base, []).append((rank, os.path.join(input_dir, filename)))
    return groups


def assemble(input_dir: str, output_dir: str) -> None:
    """Assemble rank-local diagnostics into the format emitted by one card.

    Assembly order matters: DP shards are reduced first because every DP rank
    owns different samples/tokens; PP dictionaries are then combined by layer;
    logits are handled last because DP shards use the packed-token axis while TP
    shards use the vocabulary axis.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(input_dir, "parallel_info.json")
    if not os.path.exists(manifest_path):
        print(f"[assemble] ERROR: {manifest_path} not found — is this a diagnostic dump?")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    tp_size: int = manifest.get("tp_size", 1)
    pp_size: int = manifest.get("pp_size", 1)
    dp_size: int = manifest.get("dp_size", 1)
    scopes: dict[str, str] = manifest.get("scopes", {})

    is_multi_rank = tp_size > 1 or pp_size > 1 or dp_size > 1
    if not is_multi_rank:
        print("[assemble] single-card — fast path: copying all files")
        _copy_tree(input_dir, output_dir)
        return

    print(f"[assemble] tp_size={tp_size} pp_size={pp_size} dp_size={dp_size}")

    # ── Copy parallel_info.json ────────────────────────────────────
    shutil.copy2(manifest_path, os.path.join(output_dir, "parallel_info.json"))
    print("  [copy] parallel_info.json")

    # ── DP shard assembly for stem-based files ─────────────────────
    for stem in _DP_SHARDABLE_STEMS:
        shards = _collect_dp_shards(input_dir, stem)
        if len(shards) <= 1:
            # No DP sharding for this file — copy plain version
            src = os.path.join(input_dir, f"{stem}.pt")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, f"{stem}.pt"))
                print(f"  [copy] {stem}.pt")
            continue

        # Dispatch by data semantics, not merely by tensor rank:
        # layer dictionaries contain packed tensors, cu_seqlens needs rebasing,
        # and the only remaining fixed stem (prefix_lens) is ordinary 1D data.
        print(f"  [merge] {stem}.pt ← {len(shards)} dp shards")
        if stem in _DP_PER_LAYER_STEMS:
            merged = _merge_dp_per_layer_dict(shards)
            if merged is not None:
                torch.save(merged, os.path.join(output_dir, f"{stem}.pt"))
        elif stem.startswith("cu_seqlens"):
            result = _merge_dp_1d(shards, adjust_offsets=True)
            if result is not None:
                torch.save(result, os.path.join(output_dir, f"{stem}.pt"))
        else:
            # 1D tensors: prefix_lens
            result = _merge_dp_1d(shards)
            if result is not None:
                torch.save(result, os.path.join(output_dir, f"{stem}.pt"))

    # ── DP shard assembly for tag-suffixed 2D files ─────────────────
    for prefix in _DP_TAG_PREFIXES:
        groups = _collect_dp_tag_files(input_dir, prefix)
        for base_stem, shards in groups.items():
            if len(shards) <= 1:
                # Single file — copy
                for _, filepath in shards:
                    dst = os.path.join(output_dir, os.path.basename(filepath))
                    # Use plain name without _dp{r} suffix
                    plain_name = f"{base_stem}.pt"
                    shutil.copy2(filepath, os.path.join(output_dir, plain_name))
                    print(f"  [copy] {plain_name}")
                continue

            # Multiple DP shards — merge
            result = _merge_dp_2d(sorted(shards))
            if result is not None:
                plain_name = f"{base_stem}.pt"
                torch.save(result, os.path.join(output_dir, plain_name))
                print(f"  [merge] {plain_name} ← {len(shards)} dp shards, shape {list(result.shape)}")

    # ── PP stage dict merging ──────────────────────────────────────
    for stem in _PP_STAGE_STEMS:
        filename = f"{stem}.pt"
        scope = scopes.get(stem, "")
        if scope != "pp_stage" or pp_size <= 1:
            src = os.path.join(output_dir, filename)
            if not os.path.exists(src):
                src = os.path.join(input_dir, filename)
            if os.path.exists(src) and not os.path.exists(os.path.join(output_dir, filename)):
                shutil.copy2(src, os.path.join(output_dir, filename))
            continue

        merged: dict[int, Any] = {}
        found_any = False
        # Check output_dir first (may have been DP-merged already)
        for search_dir in (output_dir, input_dir):
            for p in range(pp_size):
                src = os.path.join(input_dir, f"{stem}_pp{p}.pt")
                if os.path.exists(src):
                    layer_dict = torch.load(src, weights_only=True)
                    if isinstance(layer_dict, dict):
                        merged.update(layer_dict)
                        found_any = True
        if found_any:
            torch.save(merged, os.path.join(output_dir, filename))
            print(f"  [merge] {filename} ← {pp_size} stage(s), {len(merged)} layers")

    # DP and TP split different logical axes of logits.  Pure FSDP produces
    # one full-vocabulary [1, N_rank, V] tensor per DP rank, whereas TP produces
    # vocabulary slices.  Prefer the DP path whenever explicit _dp files exist.
    dp_shards = _collect_dp_shards(input_dir, "logits")
    if len(dp_shards) > 1:
        full = _merge_dp_logits(dp_shards)
        torch.save(full, os.path.join(output_dir, "logits.pt"))
        print(f"  [merge] logits.pt ← {len(dp_shards)} dp shards, shape {list(full.shape)}")
    elif scopes.get("logits", "") == "tp_vocab" and tp_size > 1:
        shards = []
        for t in range(tp_size):
            src = os.path.join(input_dir, f"logits_tp{t}.pt")
            if os.path.exists(src):
                shards.append(torch.load(src, weights_only=True))
        if shards:
            full = torch.cat(shards, dim=-1)
            torch.save(full, os.path.join(output_dir, "logits.pt"))
            print(f"  [concat] logits.pt ← {len(shards)} tp shards, shape {list(full.shape)}")
    else:
        src = os.path.join(output_dir, "logits.pt")
        if not os.path.exists(src):
            src = os.path.join(input_dir, "logits.pt")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, "logits.pt"))
            print("  [copy] logits.pt")

    n_files = len(os.listdir(output_dir))
    print(f"\n[assemble] done — {n_files} files written to {output_dir}")


def _copy_tree(src_dir: str, dst_dir: str) -> None:
    """Copy all .pt and .json files (fast path for single-card)."""
    if not os.path.isdir(src_dir):
        return
    for filename in os.listdir(src_dir):
        if filename.endswith(".pt") or filename.endswith(".json"):
            src = os.path.join(src_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, filename))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble multi-rank (TP/PP/DP) diagnostic dump into single-card format",
    )
    ap.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Raw multi-rank dump directory (with _pp{p}/_tp{r}/_dp{r} suffixes)",
    )
    ap.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Assembled single-card-compatible directory",
    )
    args = ap.parse_args()
    assemble(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
