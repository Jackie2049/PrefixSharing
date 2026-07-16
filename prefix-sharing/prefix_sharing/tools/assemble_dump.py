"""Multi-rank diagnostic dump assembler — merges PP/TP-sharded files into a
single-card-compatible flat directory.

Reads a raw dump directory produced by the diagnostic dump infrastructure under
TP/PP parallelism (with ``_pp{p}`` / ``_tp{r}`` file suffixes) and assembles a
clean flat directory that looks exactly like a single-card dump.  The output can
then be fed directly to the **unmodified** ``cmp_diag_verl080.py``.

Usage::

    python assemble_dump.py --input-dir /path/to/raw_dump --output-dir /path/to/assembled

Single-card dumps (tp==1, pp==1) are a fast path: all files are copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import torch

# ── Per-layer dict files that may be PP-sharded ──────────────────
# These glob patterns match the stem (without _pp suffix or .pt extension).
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

# ── Global (non-sharded) files — copied verbatim ─────────────────
_GLOBAL_FILES: list[str] = [
    "parallel_info.json",
    "cu_seqlens_q.pt",
    "cu_seqlens_q_logits.pt",
    "prefix_lens.pt",
]

# ── Optional tag-suffixed files (glob-matched) ───────────────────
_GLOB_TAG_FILES: list[str] = [
    "attention_mask_",
    "label_mask_",
    "logprobs_",
    "entropy_",
]


def _strip_ext(fname: str) -> tuple[str, str]:
    """Split ``stem.ext`` → ``(stem, ext)``.  ``ext`` includes the dot."""
    idx = fname.rfind(".")
    if idx == -1:
        return fname, ""
    return fname[:idx], fname[idx:]


def assemble(input_dir: str, output_dir: str) -> None:
    """Assemble a multi-rank dump into a single-card-compatible directory.

    Reads ``parallel_info.json`` to determine topology, then:
      - copies global metadata files verbatim
      - merges per-PP-stage layer dicts into single dicts
      - concatenates TP-sharded logits along the vocab dimension
      - copies 2D files verbatim
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load topology ─────────────────────────────────────────────
    manifest_path = os.path.join(input_dir, "parallel_info.json")
    if not os.path.exists(manifest_path):
        print(f"[assemble] ERROR: {manifest_path} not found — is this a diagnostic dump?")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    tp_size: int = manifest.get("tp_size", 1)
    pp_size: int = manifest.get("pp_size", 1)
    scopes: dict[str, str] = manifest.get("scopes", {})

    is_multi_rank = tp_size > 1 or pp_size > 1
    if not is_multi_rank:
        print("[assemble] tp=1 pp=1 — fast path: copying all files")
        _copy_tree(input_dir, output_dir)
        return

    print(f"[assemble] tp_size={tp_size} pp_size={pp_size}")

    # ── Copy global metadata files ─────────────────────────────────
    for fname in _GLOBAL_FILES:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  [copy] {fname}")

    # ── Copy glob-tagged files (attention_mask_old.pt, etc.) ────────
    for prefix in _GLOB_TAG_FILES:
        if not os.path.isdir(input_dir):
            continue
        for fname in os.listdir(input_dir):
            if fname.startswith(prefix) and fname.endswith(".pt"):
                src = os.path.join(input_dir, fname)
                shutil.copy2(src, os.path.join(output_dir, fname))
                print(f"  [copy] {fname}")

    # ── Merge per-PP-stage layer dicts ─────────────────────────────
    for stem in _PP_STAGE_STEMS:
        fname = f"{stem}.pt"
        # Determine if this file is PP-sharded from manifest scopes
        scope = scopes.get(stem, "")
        if scope != "pp_stage" or pp_size <= 1:
            # Single file — copy verbatim
            src = os.path.join(input_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
                print(f"  [copy] {fname}")
            continue

        # PP-sharded: load and merge
        merged: dict[int, Any] = {}
        found_any = False
        for p in range(pp_size):
            stem_ext = f"{stem}_pp{p}.pt"
            src = os.path.join(input_dir, stem_ext)
            if os.path.exists(src):
                d = torch.load(src, weights_only=True)
                if isinstance(d, dict):
                    merged.update(d)
                    found_any = True
        if found_any:
            torch.save(merged, os.path.join(output_dir, fname))
            print(f"  [merge] {fname} ← {pp_size} stage(s), {len(merged)} layers")

    # ── Concat TP-sharded logits ──────────────────────────────────
    if scopes.get("logits", "") == "tp_vocab" and tp_size > 1:
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
        # tp_size==1: copy logits.pt verbatim
        src = os.path.join(input_dir, "logits.pt")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, "logits.pt"))
            print("  [copy] logits.pt")

    # ── Summary ───────────────────────────────────────────────────
    n_files = len(os.listdir(output_dir))
    print(f"\n[assemble] done — {n_files} files written to {output_dir}")


def _copy_tree(src_dir: str, dst_dir: str) -> None:
    """Copy all .pt and .json files from src_dir to dst_dir (fast path for single-card)."""
    if not os.path.isdir(src_dir):
        return
    for fname in os.listdir(src_dir):
        if fname.endswith(".pt") or fname.endswith(".json"):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst_dir, fname))


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble multi-rank (TP/PP) diagnostic dump into single-card format",
    )
    ap.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Raw multi-rank dump directory (with _pp{p}/_tp{r} suffixes)",
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
