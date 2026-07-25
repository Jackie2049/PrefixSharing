"""Compare ON vs OFF model weight gradients dumped by ``dump_weight_grads_verl080``.

Usage::

    python cmp_weight_grads.py \
        --dir-on  /path/to/dump_on \
        --dir-off /path/to/dump_off \
        --tag train \
        --dp-rank 0

Outputs per-parameter cosine similarity and norm ratio.  In FSDP DP mode each
rank writes its own shard; compare rank-local shards by setting ``--dp-rank``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any


def _load_grad_dict(dir_path: str, tag: str, dp_rank: int | None) -> dict[str, Any]:
    """Load ``weight_grads_{tag}.pt`` or ``weight_grads_{tag}_dp{rank}.pt``."""
    import torch

    candidates = []
    if dp_rank is not None:
        candidates.append(os.path.join(dir_path, f"weight_grads_{tag}_dp{dp_rank}.pt"))
    candidates.append(os.path.join(dir_path, f"weight_grads_{tag}.pt"))

    for path in candidates:
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)

    searched = ", ".join(candidates)
    raise FileNotFoundError(f"No weight gradient dump found in {dir_path} (searched: {searched})")


def _cosine_similarity(a: Any, b: Any) -> float:
    """Flattened cosine similarity between two tensors."""
    import torch
    import torch.nn.functional as F

    a_flat = a.detach().flatten().float()
    b_flat = b.detach().flatten().float()
    if a_flat.numel() == 0 or b_flat.numel() == 0:
        return float("nan")
    # cosine_similarity expects shape (N, D); use (1, D) for single vectors.
    return float(
        F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=-1).item()
    )


def compare(
    dir_on: str,
    dir_off: str,
    *,
    tag: str = "train",
    dp_rank: int | None = None,
    output_csv: str | None = None,
    cos_threshold: float = 0.999,
) -> int:
    """Compare weight gradients and print a report."""
    import torch

    on_grads = _load_grad_dict(dir_on, tag, dp_rank)
    off_grads = _load_grad_dict(dir_off, tag, dp_rank)

    common_keys = sorted(set(on_grads) & set(off_grads))
    on_only = sorted(set(on_grads) - set(off_grads))
    off_only = sorted(set(off_grads) - set(on_grads))

    rows: list[dict[str, Any]] = []
    bad_params: list[tuple[str, float]] = []
    cos_values: list[float] = []

    for name in common_keys:
        g_on = on_grads[name]
        g_off = off_grads[name]
        cos = _cosine_similarity(g_on, g_off)
        norm_on = float(torch.norm(g_on.detach().float()))
        norm_off = float(torch.norm(g_off.detach().float()))
        norm_ratio = norm_on / norm_off if norm_off > 0 else float("inf")

        cos_values.append(cos)
        rows.append({
            "param": name,
            "shape": "x".join(str(d) for d in g_on.shape),
            "cos": round(cos, 6),
            "norm_on": round(norm_on, 6),
            "norm_off": round(norm_off, 6),
            "norm_ratio": round(norm_ratio, 6),
        })
        if cos < cos_threshold:
            bad_params.append((name, cos))

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["param", "shape", "cos", "norm_on", "norm_off", "norm_ratio"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[cmp_weight_grads] wrote {len(rows)} rows to {output_csv}")

    print(f"[cmp_weight_grads] common params: {len(common_keys)}")
    if on_only:
        print(f"[cmp_weight_grads] warning: {len(on_only)} params only in ON dump")
    if off_only:
        print(f"[cmp_weight_grads] warning: {len(off_only)} params only in OFF dump")

    if cos_values:
        import math

        avg_cos = sum(cos_values) / len(cos_values)
        min_cos = min(cos_values)
        max_cos = max(cos_values)
        print(f"[cmp_weight_grads] cosine  avg={avg_cos:.6f}  min={min_cos:.6f}  max={max_cos:.6f}")
        nan_count = sum(1 for v in cos_values if math.isnan(v))
        if nan_count:
            print(f"[cmp_weight_grads] warning: {nan_count} params have NaN cosine (empty grad)")

    if bad_params:
        print(f"[cmp_weight_grads] {len(bad_params)} params below cos threshold {cos_threshold}:")
        for name, cos in bad_params[:20]:
            print(f"  {name}: cos={cos:.6f}")
        if len(bad_params) > 20:
            print(f"  ... and {len(bad_params) - 20} more")
        return 1

    print("[cmp_weight_grads] all params passed cosine threshold")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ON vs OFF weight gradients")
    parser.add_argument("--dir-on", required=True, help="Dump directory for PrefixSharing ON")
    parser.add_argument("--dir-off", required=True, help="Dump directory for PrefixSharing OFF")
    parser.add_argument("--tag", default="train", help="Tag used in dump filenames")
    parser.add_argument("--dp-rank", type=int, default=None, help="DP rank suffix to compare")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--cos-threshold", type=float, default=0.999, help="Cosine similarity threshold")
    args = parser.parse_args()
    return compare(
        args.dir_on,
        args.dir_off,
        tag=args.tag,
        dp_rank=args.dp_rank,
        output_csv=args.output_csv,
        cos_threshold=args.cos_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
