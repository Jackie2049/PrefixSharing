"""Multi-rank diagnostic dump assembler — merges PP/TP/DP-sharded files into a
single-card-compatible flat directory.

Reads a raw dump directory produced by the diagnostic dump infrastructure under
TP/PP/DP parallelism (with ``_pp{p}`` / ``_tp{r}`` / ``_dp{r}`` file suffixes)
and assembles a clean flat directory that looks exactly like a single-card dump.
The output can then be fed directly to the **unmodified** ``cmp_diag.py``.

Usage::

    python assemble_dump.py --input-dir /path/to/raw_dump --output-dir /path/to/assembled

Single-card dumps (tp==1, pp==1, dp==1) are a fast path: all files are copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch

# ── Per-layer dict files that may be PP-sharded ──────────────────
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

# ── Files that may be DP-sharded (each DP rank writes its own) ───
# Format: (stem, file_pattern) — pattern may have tag suffix
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

# ── DP-shardable files that are per-layer dicts (not 1D/2D tensors) ──
_DP_PER_LAYER_STEMS: set[str] = {
    "attn_outputs",
    "rope_postqk",
    "build_kv_input_v",
    "expanded_kv",
    "attn_grads",
}

# ── Tag-suffixed files that may be DP-sharded ────────────────────
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
    """Merge DP-sharded 2D tensors [B_rank, L] → concat on dim 0 (batch)."""
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
    """Merge DP-sharded 1D tensors → concat on dim 0.

    When ``adjust_offsets=True``, each shard's tail is added as a cumulative
    base to the next shard (for cu_seqlens merging).
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


def _merge_dp_per_layer_dict(shards: list[tuple[int, str]]) -> dict | None:
    """Merge DP-sharded per-layer dicts {layer: [T_rank, ...]} → concat each layer."""
    merged: dict[int, Any] = {}
    for _, filepath in shards:
        layer_dict = torch.load(filepath, weights_only=True)
        if not isinstance(layer_dict, dict):
            continue
        for layer_idx, tensor in layer_dict.items():
            if layer_idx not in merged:
                merged[layer_idx] = []
            merged[layer_idx].append(tensor)
    if not merged:
        return None
    return {layer_idx: torch.cat(tensors, dim=0) for layer_idx, tensors in merged.items()}


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
    """Assemble a multi-rank dump into a single-card-compatible directory."""
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

    # ── logits: DP shard merge (token axis) takes precedence over TP concat ──
    dp_shards = _collect_dp_shards(input_dir, "logits")
    if len(dp_shards) > 1:
        # DP-sharded logits (logits_dp{r}.pt, pure FSDP): concat on dim 0
        # (token axis) to reconstruct the full packed logits.
        tensors = [torch.load(fp, weights_only=True) for _, fp in dp_shards]
        full = torch.cat(tensors, dim=0)
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
