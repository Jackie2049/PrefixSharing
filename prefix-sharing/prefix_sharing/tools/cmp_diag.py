"""Precision comparison — ON vs OFF (and OFF vs OFF) diagnostic dumps.

Covers the precision verification spec (see plan.md):

  1. position_ids      — absolute equality (STOP if fail)
  2. RoPE encoding     — absolute equality (STOP if fail)
  3. attn_output per-layer cos  — avg / min per layer
  4. First-token suite — attn[0] + logits[0] (mean_abs/max_abs/rel/cos/pearson)
  4b.Logits packed align — full packed logits via attention_mask + dual pointer
  5. entropy / logp    — error_abs(max,mean) / error_rel(max,mean) / pearson
  6. OFF vs OFF baseline — noise floor for all metrics

  Top-K 可选: --topk N --sort-err abs|rel 控制 2D 标量排序
               first_token 固定按 val (max magnitude) 排，不受 --sort-err 影响

Usage:
    # 最小对比 (attention_output + logits + logprobs + entropy)
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag old

    # 只看某一层 attention_output
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag old --layer 12

    # OFF vs OFF baseline (测自然噪声底)
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \\\\
        --dir-off2 ./dump_off2 --tag old

    # 导出 JSON 报告
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \\\\
        --tag old -o report.json

    # 打印每种比较最差的 10 个元素 (按绝对误差排)
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \\\\
        --tag old --topk 10

    # 打印每种比较最差的 20 个元素 (按相对误差排)
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \\\\
        --tag old --topk 20 --sort-err rel

    # 打印 first_token/logp/entropy 值最大的 10 个维度 (不受 0 附近噪声干扰)
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \\\\
        --tag old --topk 10 --sort-err val

Parameters:
    --dir-on      (必需) ON模式 dump 目录，含 attn_outputs.pt / logits.pt /
                          logprobs_{tag}.pt / entropy_{tag}.pt / attention_mask.pt
    --dir-off     (必需) OFF模式 dump 目录，文件同 --dir-on
    --dir-off2    (可选) 第二个 OFF 目录，用于 OFF-vs-OFF baseline
                          两个独立 OFF run 之间的自然误差可作噪声参考基准
    --tag         (可选) 2D文件标签，如 old / train / before_restore
                          对应文件: logprobs_{tag}.pt / entropy_{tag}.pt
    --tag-on      (可选) 覆盖 ON 侧的 tag (不传则用 --tag)
    --tag-off     (可选) 覆盖 OFF 侧的 tag (不传则用 --tag)
    --layer       (可选) 只对比指定层的 attention output (1-indexed)
                          不传则所有层遍历对比
    --atol        (可选) 2D 对比绝对容差，默认 1e-5
    --mask-file   (可选) 外部 label_mask.pt，会裁剪每行最后一个 True
    --no-trim     (flag)  不裁剪 label_mask 的最后一个 True
    -o, --output  (可选) 将对比报告保存为 JSON 文件
    --topk        (可选) 打印前 N 个误差最大的元素 (0=关闭, 默认 0)
                          对 first_token_attn/logits 按维度排名
                          对 logp/entropy 按 (batch, pos) 位置排名
    --sort-err    (可选) 2D 标量 top-K 排序依据: abs=绝对误差, rel=相对误差 (默认 abs)
                          first_token 固定按 val (max magnitude) 排，不受此参数影响
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field

import torch

_log = logging.getLogger(__name__)

# ── Constants ──
_SEP_DOUBLE = "=" * 70
_SEP_SINGLE = "-" * 70
_SEP_THIN   = "─" * 70
_CHECK      = "\u2713"
_CROSS      = "\u2717"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _fmt_shape(s: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in s) + ")"


# ══════════════════════════════════════════════════════════════════
#  Dataclasses
# ══════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool = True
    metrics: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
#  Mask helpers
# ══════════════════════════════════════════════════════════════════

def _trim_last_true_per_row(mask: torch.Tensor) -> torch.Tensor:
    """In each row of a 2D bool mask, flip the last True to False."""
    if mask.dim() != 2:
        return mask
    result = mask.clone()
    for row in range(mask.shape[0]):
        pos = mask[row].nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            result[row, pos[-1]] = False
    return result


def _load_label_mask(dir_path: str) -> torch.Tensor | None:
    path = os.path.join(dir_path, "label_mask.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).to(torch.bool)


def _resolve_label_mask(mask_file: str | None, dir_off: str,
                        trim_last: bool = True) -> torch.Tensor | None:
    """Resolve effective label_mask: external file > auto-load from OFF dump.

    Always applies last-True→False trim when ``trim_last=True``.
    """
    if mask_file:
        mask = torch.load(mask_file, weights_only=True).to(torch.bool)
        return _trim_last_true_per_row(mask) if trim_last else mask
    mask = _load_label_mask(dir_off)
    if mask is not None and trim_last:
        mask = _trim_last_true_per_row(mask)
    return mask


# ══════════════════════════════════════════════════════════════════
#  Metric computation helpers
# ══════════════════════════════════════════════════════════════════

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine similarity along ``dim``.

    Inputs are promoted to float32 before computing to avoid low-precision
    (bf16) rounding artifacts: even when a==b, the numerator
    ``(a*b).sum()`` and the denominator ``|a|*|b| = sqrt(sum(a^2)) *
    sqrt(sum(b^2))`` take different reduction paths and round to
    different bf16 values, yielding cos < 1.0 (e.g. 0.988) for identical
    tensors. In float32 the two paths agree closely enough that a==b
    gives cos ≈ 1.0.

    Near-zero vectors (both norms < ``sqrt(eps)``) are treated as
    identical and return 1.0 to avoid the clamp artifact on tiny norms.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    eps = 1e-8
    na = a.norm(dim=dim)
    nb = b.norm(dim=dim)
    denom = (na * nb).clamp(min=eps)
    cos = (a * b).sum(dim=dim) / denom
    near_zero = (na < eps ** 0.5) & (nb < eps ** 0.5)
    return torch.where(near_zero, torch.ones_like(cos), cos)


def _pearson_r(t1: torch.Tensor, t2: torch.Tensor,
               mask: torch.Tensor | None = None) -> float:
    x = (t1[mask] if mask is not None else t1.flatten()).to(torch.float64)
    y = (t2[mask] if mask is not None else t2.flatten()).to(torch.float64)
    if x.numel() < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx, sy = ((x - mx) ** 2).sum().sqrt(), ((y - my) ** 2).sum().sqrt()
    return float("nan") if sx == 0 or sy == 0 else float(cov / (sx * sy))


def _error_abs_rel(a: torch.Tensor, b: torch.Tensor,
                   mask: torch.Tensor | None = None) -> dict:
    diff = (a - b).abs()
    rel = diff / torch.maximum(a.abs(), b.abs()).clamp(min=1e-8)
    if mask is not None:
        diff, rel = diff[mask], rel[mask]
    if diff.numel() == 0:
        return {"abs_max": 0.0, "abs_mean": 0.0, "rel_max": 0.0, "rel_mean": 0.0}
    return {"abs_max": float(diff.max()), "abs_mean": float(diff.mean()),
            "rel_max": float(rel.max()), "rel_mean": float(rel.mean())}


def _first_token_metrics(a_vec: torch.Tensor, b_vec: torch.Tensor) -> dict:
    err = _error_abs_rel(a_vec, b_vec)
    cos = float(_cosine_sim(a_vec, b_vec, dim=-1))
    pr  = _pearson_r(a_vec, b_vec)
    return {"mean_abs": err["abs_mean"], "max_abs": err["abs_max"],
            "rel_max": err["rel_max"], "rel_mean": err["rel_mean"],
            "cos": cos, "pearson": pr}


def _logits_first_token(logits_on: torch.Tensor, logits_off: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extract first token's full-vocab vector from ON/OFF logits.

    Logits are either [N, V] (token-major) or [B, N, V] (batched token-major).
    Vocab is always the last dim (verified from Megatron model forward:
    output_layer → [S, B, V//tp] → model returns [B, S, V//tp]).
    """
    # Flatten all leading batch/token dims into N, keep V as last dim
    on_flat = logits_on.reshape(-1, logits_on.size(-1))
    off_flat = logits_off.reshape(-1, logits_off.size(-1))
    # First token = first row → full-vocab vector [V]
    return on_flat[0, :].contiguous(), off_flat[0, :].contiguous()


def _logits_ensure_token_major(logits_on: torch.Tensor, logits_off: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure logits are 2D [N, V] for packed alignment.

    Vocab is always the last dim (verified from Megatron model forward).
    Batch dim on leading axes is flattened into N.
    """
    return (logits_on.reshape(-1, logits_on.size(-1)).contiguous(),
            logits_off.reshape(-1, logits_off.size(-1)).contiguous())


# ══════════════════════════════════════════════════════════════════
#  Loading helpers
# ══════════════════════════════════════════════════════════════════

def _load_tensor(dir_path: str, filename: str) -> torch.Tensor | None:
    filepath = os.path.join(dir_path, filename)
    return torch.load(filepath, weights_only=True).float() if os.path.exists(filepath) else None


def _load_packed_meta(dir_path: str,
                      cu_fname: str = "cu_seqlens_q.pt") -> dict | None:
    filepath = os.path.join(dir_path, cu_fname)
    if not os.path.exists(filepath):
        filepath = os.path.join(dir_path, "cu_seqlens_q.pt")
        if not os.path.exists(filepath):
            return None
    prefix_lens_filepath = os.path.join(dir_path, "prefix_lens.pt")
    if not os.path.exists(prefix_lens_filepath):
        return None
    return {"cu_seqlens": torch.load(filepath, weights_only=True),
            "prefix_lens": torch.load(prefix_lens_filepath, weights_only=True)}


def _load_attn_output(dir_path: str, layer: int) -> torch.Tensor | None:
    """Load a single layer's attn_output from attn_outputs.pt dict."""
    filepath = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(filepath):
        return None
    d = torch.load(filepath, weights_only=True)
    return d.get(layer) if isinstance(d, dict) else None


def _load_attention_mask_2d(dir_path: str) -> torch.Tensor | None:
    """Load 2D [B, L_max] attention_mask.pt dumped by ``dump_attention_mask_2d``.

    Semantics (see diagnostic_dump.dump_attention_mask_2d):
      - ON: True only at suffix columns per row (rewritten by verl_mcore)
      - OFF: True at all valid token columns per row

    Both share the same [B, L_max] shape and per-row column semantics
    (column ``c`` corresponds to the same original token position in both
    runs).  ON mask is a strict subset of OFF mask per row, which
    ``_build_alignment_mask_from_2d`` relies on.
    """
    filepath = os.path.join(dir_path, "attention_mask.pt")
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, weights_only=True).to(torch.bool)


def _get_num_layers(dir_path: str) -> int:
    filepath = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(filepath):
        return 0
    d = torch.load(filepath, weights_only=True)
    return max(d.keys()) if isinstance(d, dict) and d else 0


# ══════════════════════════════════════════════════════════════════
#  Packed alignment — build 1D suffix mask, align ON/OFF tensors
# ══════════════════════════════════════════════════════════════════

def _build_alignment_mask(cu_seqlens: torch.Tensor,
                          prefix_lens: torch.Tensor,
                          total_tokens: int) -> torch.Tensor:
    """Build 1D bool mask [total_tokens]: True iff position is a suffix token.

    Built from ON's metadata and applied to OFF's full packed data to extract
    the same suffix region, yielding equal-shape tensors for comparison.

    Example:
        cu_seqlens=[0,7,13], prefix_lens=[3,4], total_tokens=13
        → [0,0,0, 1,1,1,1,  0,0,0,0, 1,1]
    """
    mask = torch.zeros(total_tokens, dtype=torch.bool)
    for i in range(cu_seqlens.shape[0] - 1):
        pf = int(prefix_lens[i])
        start = int(cu_seqlens[i]) + pf
        end = int(cu_seqlens[i + 1])
        mask[start:end] = True
    return mask


def _align_packed(on_tensor: torch.Tensor,
                  off_tensor: torch.Tensor,
                  alignment_mask: torch.Tensor,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Align ON-only-suffix with OFF-full-packed via 1D alignment_mask.

    Returns (on, off_suffix) where off_suffix has same shape[0] as on_tensor.

    Double-pointer semantics:
      for each i in [0, OFF_total_tokens):
        if alignment_mask[i]: compare ON[on_ptr++] vs OFF[i]
        else:                skip (prefix position in OFF)
    """
    T = off_tensor.shape[0]
    n_suffix = int(alignment_mask.sum())
    if alignment_mask.shape[0] != T:
        raise ValueError(
            f"alignment_mask len {alignment_mask.shape[0]} != OFF tokens {T}")
    if on_tensor.shape[0] != n_suffix:
        raise ValueError(
            f"ON tokens {on_tensor.shape[0]} != alignment True count {n_suffix}")
    return on_tensor, off_tensor[alignment_mask]


def _build_alignment_mask_from_2d(
    mask_on_2d: torch.Tensor,
    mask_off_2d: torch.Tensor,
    cu_seqlens_off: torch.Tensor,
    total_off: int,
) -> torch.Tensor:
    """Build 1D alignment mask [total_off] from ON/OFF 2D [B, L_max] masks.

    ON mask is a strict subset of OFF mask per row (ON ⊂ OFF).
    For each sequence row r:
      - OFF packed row r = cu_seqlens_off[r] : cu_seqlens_off[r+1],
        whose tokens correspond to mask_off_2d[r] True columns in order.
      - ON wants the suffix tokens → columns where mask_on_2d[r] is True.
      - Since mask_on[r] ⊂ mask_off[r], we intersect to find which OFF
        packed positions correspond to suffix tokens in ON.
    """
    align = torch.zeros(total_off, dtype=torch.bool)
    for r in range(mask_on_2d.shape[0]):
        row_start = int(cu_seqlens_off[r])
        row_end = int(cu_seqlens_off[r + 1])
        if row_start >= row_end:
            continue
        off_len = row_end - row_start
        # Columns in original L_max that correspond to OFF packed row tokens
        off_cols = mask_off_2d[r].nonzero(as_tuple=True)[0]
        if off_cols.numel() < off_len:
            continue
        off_cols = off_cols[:off_len]
        # Mark positions where ON mask also has True (these are suffix token columns)
        is_on = mask_on_2d[r, off_cols].to(torch.bool)
        align[row_start:row_end] = is_on
    return align



# ══════════════════════════════════════════════════════════════════
#  1. position_ids — absolute equality
# ══════════════════════════════════════════════════════════════════

def cmp_position_ids(dir_a: str, dir_b: str) -> CheckResult | None:
    a = _load_tensor(dir_a, "position_ids.pt")
    b = _load_tensor(dir_b, "position_ids.pt")
    if a is None or b is None:
        return None
    ok = bool((a.to(torch.long) == b.to(torch.long)).all())
    return CheckResult(name="position_ids", passed=ok,
                       metrics={"shape": _fmt_shape(tuple(a.shape))})


# ══════════════════════════════════════════════════════════════════
#  2. RoPE encoding — absolute equality
# ══════════════════════════════════════════════════════════════════

def cmp_rope_postqk(dir_on: str, dir_off: str) -> CheckResult | None:
    """Compare post-RoPE Q/K between ON and OFF, aligned by position IDs.

    ON  positions are absolute (preserved from original input).
    OFF positions are per-segment relative (0..L-1 for each row).

    Comparison strategy:
    1. First packed token (row 0, pos 0): always a provider, must match exactly.
    2. Within the first row: match tokens by position ID (0..L-1 for both paths).
    3. For reuser rows: ON uses absolute positions (prefix_len..), OFF uses
       relative (0..).  Positions differ by design — skip direct comparison.
    """
    filepath_on = os.path.join(dir_on, "rope_postqk.pt")
    filepath_off = os.path.join(dir_off, "rope_postqk.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    a = torch.load(filepath_on, weights_only=True)
    b = torch.load(filepath_off, weights_only=True)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    layers_on, layers_off =set(a.keys()), set(b.keys())
    if layers_on != layers_off:
        return CheckResult(name="rope_postqk", passed=False,
                           metrics={"error": "layer set mismatch"})

    max_diff_q = 0.0
    max_diff_k = 0.0
    first_token_q_diff = 0.0
    first_token_k_diff = 0.0
    first_row_q_diff = 0.0
    first_row_k_diff = 0.0
    first_row_len = 0
    num_layers = len(layers_on)

    for layer_idx in sorted(layers_on):
        on_entry = a[layer_idx]
        off_entry = b[layer_idx]
        on_q, on_k = on_entry["query"], on_entry["key"]
        off_q, off_k = off_entry["query"], off_entry["key"]
        on_pos = on_entry.get("positions")
        off_pos = off_entry.get("positions")

        # ── Check 1: first packed token (row 0, position 0) ──
        first_token_q_diff = max(first_token_q_diff,
                                 float((on_q[0] - off_q[0]).abs().max()))
        first_token_k_diff = max(first_token_k_diff,
                                 float((on_k[0] - off_k[0]).abs().max()))

        # ── Check 2: first row — match by ON position IDs ──
        if on_pos is not None and off_pos is not None:
            on_pos_t = on_pos.long()
            off_pos_t = off_pos.long()
            # Find first-row extent in ON: tokens before the first position reset
            # (position decreases or jumps to prefix_start)
            on_row1_end = 1
            for j in range(1, len(on_pos_t)):
                if on_pos_t[j] <= on_pos_t[j - 1]:
                    break
                on_row1_end = j + 1
            on_row1_len = on_row1_end  # positions 0..L-1

            # First row in OFF: positions go 0..L-1 (find matching extent)
            off_row1_end = 1
            for j in range(1, len(off_pos_t)):
                if off_pos_t[j] <= off_pos_t[j - 1]:
                    break
                off_row1_end = j + 1
            off_row1_len = off_row1_end

            compare_length = min(on_row1_len, off_row1_len)
            if compare_length > 0:
                first_row_len = max(first_row_len, compare_length)
                # Match by position ID within the first row
                for position_id in range(compare_length):
                    on_index = position_id  # ON row 1 starts at packed index 0
                    off_index = position_id  # OFF row 1 starts at packed index 0
                    first_row_q_diff = max(first_row_q_diff,
                        float((on_q[on_index] - off_q[off_index]).abs().max()))
                    first_row_k_diff = max(first_row_k_diff,
                        float((on_k[on_index] - off_k[off_index]).abs().max()))
                max_diff_q = max(max_diff_q, first_row_q_diff)
                max_diff_k = max(max_diff_k, first_row_k_diff)
        else:
            # Fallback: no positions available, compare first row by direct offset
            compare_length = min(on_q.shape[0], off_q.shape[0], 128)
            first_row_len = compare_length
            first_row_q_diff = float((on_q[:compare_length] - off_q[:compare_length]).abs().max())
            first_row_k_diff = float((on_k[:compare_length] - off_k[:compare_length]).abs().max())
            max_diff_q = first_row_q_diff
            max_diff_k = first_row_k_diff

    # Pass if all diffs are near zero
    threshold = 1e-7
    passed = (first_token_q_diff < threshold and first_token_k_diff < threshold
              and first_row_q_diff < threshold and first_row_k_diff < threshold)
    return CheckResult(name="rope_postqk", passed=passed, metrics={
        "num_layers": num_layers,
        "first_row_len": first_row_len,
        "first_token_q_maxdiff": first_token_q_diff,
        "first_token_k_maxdiff": first_token_k_diff,
        "first_row_q_maxdiff": first_row_q_diff,
        "first_row_k_maxdiff": first_row_k_diff,
        "max_diff_q": max_diff_q,
        "max_diff_k": max_diff_k,
    })


def cmp_rope_freqs(dir_on: str, dir_off: str) -> CheckResult | None:
    """Compare pre-RoPE angles (angle table, not cos/sin) — suffix-aligned.

    ON  ``rope_freqs_on.pt``:  per-token angles [T_on, 1, 1, D]
    OFF ``rope_freqs_off.pt``: raw angle table [L0, 1, 1, D]

    OFF per-token angles are reconstructed from the raw table via
    ``cu_seqlens_off`` (each segment takes ``[:seg_len]`` from position 0).

    The two are aligned to suffix-only via the same attention_mask dual-pointer
    logic used for attn_outputs / logits.
    """
    filepath_on = os.path.join(dir_on, "rope_freqs_on.pt")
    filepath_off = os.path.join(dir_off, "rope_freqs_off.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None

    layers_on, layers_off =set(on_dict.keys()), set(off_dict.keys())
    if layers_on != layers_off:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "layer set mismatch",
                                    "on_layers": sorted(layers_on),
                                    "off_layers": sorted(layers_off)})

    # Load OFF metadata for per-token reconstruction + alignment
    meta_off = _load_packed_meta(dir_off)
    if meta_off is None:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "OFF cu_seqlens missing"})
    cu_off = meta_off["cu_seqlens"]
    T_off = int(cu_off[-1]) if cu_off.numel() > 0 else 0

    # Build alignment mask (prefer 2D attention_mask)
    mask_on_2d = _load_attention_mask_2d(dir_on)
    mask_off_2d = _load_attention_mask_2d(dir_off)
    meta_on = _load_packed_meta(dir_on)
    if mask_on_2d is not None and mask_off_2d is not None:
        align_mask = _build_alignment_mask_from_2d(
            mask_on_2d, mask_off_2d, cu_off, T_off)
    elif meta_on is not None:
        align_mask = _build_alignment_mask(
            cu_off, meta_on["prefix_lens"], T_off)
    else:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "cannot build alignment mask"})

    # Reconstruct OFF per-token angles from raw table
    seqlens = (cu_off[1:] - cu_off[:-1]).tolist()

    max_diff = 0.0
    mismatches: list[dict] = []  # [{layer, token_idx, dim, on_val, off_val, diff}]
    for layer_idx in sorted(layers_on):
        on_freqs = on_dict[layer_idx]                          # [T_on, 1, 1, D]
        # Reconstruct OFF per-token for this layer
        off_freqs = torch.cat(
            [off_dict[layer_idx][:s, :, :, :] for s in seqlens], dim=0)  # [T_off, 1, 1, D]

        try:
            on_aligned, off_aligned = _align_packed(
                on_freqs, off_freqs, align_mask)
        except ValueError as e:
            return CheckResult(name="rope_freqs", passed=False,
                               metrics={"error": f"align failed L{layer_idx}: {e}"})

        diff = (on_aligned - off_aligned).abs()                # [N, 1, 1, D]
        md = float(diff.max())
        max_diff = max(max_diff, md)

        if md > 0:
            # per-token max diff across all dims
            token_diff = diff.squeeze(1).squeeze(1).max(dim=-1)  # values [N], indices [N]
            bad_mask = token_diff.values > 0
            for t in bad_mask.nonzero(as_tuple=True)[0].tolist():
                t = int(t)
                d = int(token_diff.indices[t])
                mismatches.append({
                    "layer": layer_idx,
                    "token_idx": t,
                    "dim": d,
                    "on_val": float(on_aligned[t, 0, 0, d]),
                    "off_val": float(off_aligned[t, 0, 0, d]),
                    "diff": float(token_diff.values[t]),
                })

    metrics: dict = {"max_diff": max_diff, "num_layers": len(layers_on)}
    if mismatches:
        metrics["mismatches"] = mismatches[:20]  # cap to top 20
        metrics["total_mismatches"] = len(mismatches)
    return CheckResult(name="rope_freqs", passed=max_diff == 0.0,
                       metrics=metrics)


# ══════════════════════════════════════════════════════════════════
#  3. attention_output per-layer cos
# ══════════════════════════════════════════════════════════════════

def _per_layer_cos(dir_on: str, dir_off: str, layer: int | None) -> dict | None:
    """Per-layer cosine similarity with packed alignment.

    ON attn_output may have fewer tokens (suffix only) than OFF (all tokens).
    Builds alignment_mask from ON's cu_seqlens/prefix_lens and uses it to
    extract the matching suffix region from OFF before comparison.
    """

    def _cos_for_layer(a, b, layer_idx, need_align, align_mask):
        if a.dim() == 3:
            a, b = a.squeeze(1), b.squeeze(1)
        if need_align and a.shape[0] != b.shape[0]:
            try:
                a, b = _align_packed(a, b, align_mask)
            except ValueError as e:
                return {"cos_avg": 0.0, "cos_min": 0.0,
                        "n_tokens": a.shape[0], "error": str(e)}
        cos = _cosine_sim(a, b, dim=-1)
        return {"cos_avg": float(cos.mean()), "cos_min": float(cos.min()),
                "n_tokens": a.shape[0]}

    # Single-layer mode
    if layer is not None:
        a = _load_attn_output(dir_on, layer)
        b = _load_attn_output(dir_off, layer)
        if a is None or b is None:
            return None
        meta_on = _load_packed_meta(dir_on)
        align_mask = None
        need_align = (meta_on is not None and a.shape[0] != b.shape[0])
        if need_align:
            meta_off = _load_packed_meta(dir_off)
            T = int(meta_off["cu_seqlens"][-1]) if meta_off and meta_off["cu_seqlens"].numel() > 0 else b.shape[0]
            # Prefer 2D attention_mask.pt for exact suffix alignment (dp aligned)
            mask_on_2d = _load_attention_mask_2d(dir_on)
            mask_off_2d = _load_attention_mask_2d(dir_off)
            if mask_on_2d is not None and mask_off_2d is not None and meta_off is not None:
                align_mask = _build_alignment_mask_from_2d(
                    mask_on_2d, mask_off_2d, meta_off["cu_seqlens"], T)
            else:
                align_mask = _build_alignment_mask(
                    meta_off["cu_seqlens"], meta_on["prefix_lens"], T)
        d = _cos_for_layer(a, b, layer, need_align, align_mask)
        d["layer"] = layer
        return d

    # All-layers mode
    filepath_on = os.path.join(dir_on, "attn_outputs.pt")
    filepath_off = os.path.join(dir_off, "attn_outputs.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    attn_dict_on = torch.load(filepath_on, weights_only=True)
    attn_dict_off = torch.load(filepath_off, weights_only=True)
    if not isinstance(attn_dict_on, dict) or not isinstance(attn_dict_off, dict):
        return None

    # Build alignment mask once from ON metadata
    meta_on = _load_packed_meta(dir_on)
    align_mask = None
    need_align = False
    if meta_on is not None:
        meta_off = _load_packed_meta(dir_off)
        T = int(meta_off["cu_seqlens"][-1]) if meta_off and meta_off["cu_seqlens"].numel() > 0 else 0
        if T > 0:
            # Check if any layer has shape mismatch
            for layer_idx in attn_dict_on:
                if layer_idx in attn_dict_off and attn_dict_on[layer_idx].shape != attn_dict_off[layer_idx].shape:
                    need_align = True
                    break
            if need_align:
                # Prefer 2D attention_mask.pt for exact suffix alignment
                mask_on_2d = _load_attention_mask_2d(dir_on)
                mask_off_2d = _load_attention_mask_2d(dir_off)
                if mask_on_2d is not None and mask_off_2d is not None:
                    align_mask = _build_alignment_mask_from_2d(
                        mask_on_2d, mask_off_2d, meta_off["cu_seqlens"], T)
                else:
                    align_mask = _build_alignment_mask(
                        meta_off["cu_seqlens"], meta_on["prefix_lens"], T)

    results = {}
    for layer_idx in sorted(set(attn_dict_on.keys()) & set(attn_dict_off.keys())):
        results[layer_idx] = _cos_for_layer(attn_dict_on[layer_idx], attn_dict_off[layer_idx], layer_idx, need_align, align_mask)
    return results


def cmp_attn_layer(dir_on: str, dir_off: str,
                   layer: int | None) -> CheckResult | None:
    r = _per_layer_cos(dir_on, dir_off, layer)
    if r is None:
        return None
    if "cos_avg" in r:
        layer_idx = r["layer"]
        return CheckResult(name=f"attn_L{layer_idx}",
                           passed=r["cos_avg"] > 0.9999 and r["cos_min"] > 0.999,
                           metrics=r)
    return CheckResult(name="attn_per_layer", passed=True,
                       metrics={"layers": r})


# ══════════════════════════════════════════════════════════════════
#  4. First-token metrics (last-layer attn[0] + logits[0])
# ══════════════════════════════════════════════════════════════════

def cmp_first_token(dir_on: str, dir_off: str) -> list[CheckResult]:
    """Compare packed[0] of last-layer attn_output + logits.

    The first sentence never becomes a resuer so its packed[0] is always a
    complete suffix token — safe to compare directly without alignment.
    """
    results: list[CheckResult] = []
    last = _get_num_layers(dir_on) or _get_num_layers(dir_off)

    # attn_output (last layer) — packed[0]
    if last:
        a = _load_attn_output(dir_on, last)
        b = _load_attn_output(dir_off, last)
        if a is not None and b is not None:
            on_token0 = a.squeeze(1) if a.dim() == 3 else a
            off_token0 = b.squeeze(1) if b.dim() == 3 else b
            metrics = _first_token_metrics(on_token0[0], off_token0[0])
            results.append(CheckResult(name="first_token_attn", metrics=metrics))

    # logits — packed[0], auto-detect [N,V] vs [V,N] format
    logits_on = _load_tensor(dir_on,  "logits.pt")
    logits_off = _load_tensor(dir_off, "logits.pt")
    if logits_on is not None and logits_off is not None:
        logits_on_first, logits_off_first = _logits_first_token(logits_on, logits_off)
        if logits_on_first is not None:
            metrics = _first_token_metrics(logits_on_first, logits_off_first)
            results.append(CheckResult(name="first_token_logits", metrics=metrics))
        else:
            _log.warning("first_token_logits skipped: cannot determine token dim "
                         "(ON %s, OFF %s)", _fmt_shape(logits_on.shape), _fmt_shape(logits_off.shape))

    return results


# ══════════════════════════════════════════════════════════════════
#  4b. Logits — full packed alignment (attention_mask + dual pointer)
# ══════════════════════════════════════════════════════════════════

def cmp_logits_packed(dir_on: str, dir_off: str) -> CheckResult | None:
    """Compare ON vs OFF logits with full packed alignment.

    Uses attention_mask_2d + dual-pointer to align ON (suffix-only)
    with OFF (full-sequence) logits, same alignment logic as attn_output
    per-layer comparison.
    """
    logits_on = _load_tensor(dir_on,  "logits.pt")
    logits_off = _load_tensor(dir_off, "logits.pt")
    if logits_on is None or logits_off is None:
        return None

    # Logits may be [N,V] or [V,N] — ensure token-major [N,V] for alignment
    logits_on, logits_off = _logits_ensure_token_major(logits_on, logits_off)

    # Metadata for logits uses cu_seqlens_q_logits.pt
    meta_on = _load_packed_meta(dir_on,  "cu_seqlens_q_logits.pt")
    meta_off = _load_packed_meta(dir_off, "cu_seqlens_q_logits.pt")
    if meta_on is None or meta_off is None:
        return None

    total_off_tokens = int(meta_off["cu_seqlens"][-1]) if meta_off["cu_seqlens"].numel() > 0 else 0
    if total_off_tokens == 0 or logits_on.shape[0] == 0 or logits_off.shape[0] == 0:
        return None

    # Build alignment mask (same logic as attn_output all-layers)
    mask_on_2d = _load_attention_mask_2d(dir_on)
    mask_off_2d = _load_attention_mask_2d(dir_off)
    if mask_on_2d is not None and mask_off_2d is not None:
        align_mask = _build_alignment_mask_from_2d(
            mask_on_2d, mask_off_2d, meta_off["cu_seqlens"], total_off_tokens)
    else:
        align_mask = _build_alignment_mask(
            meta_off["cu_seqlens"], meta_on["prefix_lens"], total_off_tokens)

    # Align: extract suffix-only region from OFF
    try:
        on_aligned, off_aligned = _align_packed(logits_on, logits_off, align_mask)
    except ValueError as e:
        return CheckResult(name="logits", passed=False,
                           metrics={"error": str(e),
                                    "n_on": logits_on.shape[0],
                                    "n_off": logits_off.shape[0]})

    n_tokens = on_aligned.shape[0]

    # Per-token cosine similarity (dim=-1 across vocab)
    cos = _cosine_sim(on_aligned, off_aligned, dim=-1)
    cos_avg = float(cos.mean())
    cos_min = float(cos.min())

    # Logits live in vocab space (V ~ 1e5); abs/rel error and pearson are
    # dominated by a few extreme dims and carry little signal. Per-token
    # cosine similarity is the meaningful metric here.
    return CheckResult(
        name="logits",
        passed=cos_avg > 0.9999 and cos_min > 0.999,
        metrics={"n_tokens": n_tokens,
                 "cos_avg": cos_avg,
                 "cos_min": cos_min})


# ══════════════════════════════════════════════════════════════════
#  5. 2D comparison (entropy / logp) — error_abs + error_rel + pearson
# ══════════════════════════════════════════════════════════════════

def cmp_2d(dir_on: str, dir_off: str, filename: str, name: str,
           mask: torch.Tensor | None, atol: float,
           label: torch.Tensor | None = None) -> CheckResult:
    t1 = _load_tensor(dir_on,  filename)
    t2 = _load_tensor(dir_off, filename)
    if t1 is None or t2 is None:
        return CheckResult(name=name, passed=False,
                           metrics={"error": "file missing"})
    if t1.shape != t2.shape:
        return CheckResult(name=name, passed=False,
                           metrics={"error": "shape mismatch",
                                    "s_on": _fmt_shape(tuple(t1.shape)),
                                    "s_off": _fmt_shape(tuple(t2.shape))})
    if mask is None:
        mask = torch.ones(t1.shape, dtype=torch.bool, device=t1.device)
    else:
        mask = mask.to(t1.device)
    mask = mask & ~torch.isnan(t1) & ~torch.isnan(t2)
    err = _error_abs_rel(t1, t2, mask)
    n_act = int(mask.sum())
    return CheckResult(name=name,
                       passed=n_act == 0 or err["abs_max"] <= atol,
                       metrics={"shape": _fmt_shape(tuple(t1.shape)),
                                "active": n_act,
                                "abs_max": err["abs_max"],
                                "abs_mean": err["abs_mean"],
                                "rel_max": err["rel_max"],
                                "rel_mean": err["rel_mean"],
                                "pearson_r": _pearson_r(t1, t2, mask),
                                "atol": atol})


# ══════════════════════════════════════════════════════════════════
#  Output helpers
# ══════════════════════════════════════════════════════════════════

def _print_header(dir_on, dir_off, dir_off2, tag, mask_file, layer):
    print(_SEP_DOUBLE)
    print("  Prefix-Sharing Diag Report")
    print(f"  ON :  {dir_on}\n  OFF:  {dir_off}")
    if dir_off2:
        print(f"  OFF2: {dir_off2}")
    if tag:
        print(f"  TAG: {tag}")
    if mask_file:
        print(f"  MASK: {mask_file} (external, last-token trimmed)")
    if layer is not None:
        print(f"  LAYER: {layer}")
    print(_SEP_DOUBLE + "\n")


def _print_pos_ids(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [position_ids]  {_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE + f"\n  shape: {r.metrics.get('shape','—')}")
    if not r.passed:
        print(f"  {_CROSS} CRITICAL — STOP.\n")
    print()


def _print_rope(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [rope_postqk]  {_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {m['error']}")
    else:
        print(f"  layers: {m.get('num_layers','—')}  max_diff: {m.get('max_diff','—')}")
    if not r.passed:
        print(f"  {_CROSS} CRITICAL — STOP.\n")
    print()


def _print_rope_freqs(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [rope_freqs]  {_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {m['error']}")
    else:
        print(f"  layers: {m.get('num_layers','—')}  max_diff: {m.get('max_diff','—')}")
        total = m.get("total_mismatches", 0)
        if total > 0:
            print(f"  mismatched tokens: {total}"
                  f"{' (showing first 20)' if total > 20 else ''}")
            print(f"  {'Layer':>6s} {'Token':>6s} {'Dim':>6s}  "
                  f"{'ON_val':>14s}  {'OFF_val':>14s}  {'Abs_err':>12s}")
            for mm in m.get("mismatches", []):
                print(f"  {mm['layer']:>6d} {mm['token_idx']:>6d} {mm['dim']:>6d}  "
                      f"{mm['on_val']:>14.6e}  {mm['off_val']:>14.6e}  "
                      f"{mm['diff']:>12.6e}")
    if not r.passed:
        print(f"  {_CROSS} CRITICAL — STOP.\n")
    print()


def _print_per_layer(r: CheckResult):
    print(_SEP_SINGLE + "\n  [attn_output]  Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = r.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'COS_AVG':>14s}  {'COS_MIN':>14s}  {'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*8}")
        bad = []
        for layer_idx in sorted(layers.keys()):
            d = layers[layer_idx]
            ok = d["cos_avg"] > 0.9999 and d["cos_min"] > 0.999
            print(f"  {layer_idx:>6d}  {d['cos_avg']:>14.6e}  {d['cos_min']:>14.6e}  "
                  f"{d['n_tokens']:>8d}  {'PASS' if ok else 'WARN':>8s}")
            if not ok:
                bad.append(layer_idx)
        if bad:
            print(f"\n  ⚠ First deviating layer: {bad[0]}")
    elif "cos_avg" in r.metrics:
        d = r.metrics
        print(f"  L{d['layer']}  cos_avg={d['cos_avg']:.6e}  cos_min={d['cos_min']:.6e}")
    print()


def _print_first_token(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [first_token]  {r.name}  (packed position [0])")
    print(_SEP_SINGLE)
    m = r.metrics
    for k in ["mean_abs", "max_abs", "rel_max", "rel_mean", "cos", "pearson"]:
        v = m.get(k)
        if v is not None:
            print(f"  {k:>12s}  {v:>14.6e}")
    print()


def _print_logits_packed(r: CheckResult):
    """Print full packed logits comparison (attention_mask + dual pointer aligned)."""
    print(_SEP_SINGLE + "\n  [logits]  packed alignment (attention_mask + dual pointer)")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}")
    else:
        print(f"  n_tokens={m.get('n_tokens','—')}  "
              f"cos_avg={m.get('cos_avg',0):.6e}  cos_min={m.get('cos_min',0):.6e}  "
              f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print()


def _print_topk_vec(on_vec: torch.Tensor, off_vec: torch.Tensor,
                    topk: int, sort_by: str, label: str):
    """Print top-K dimensions from two 1D vectors.

    sort_by = "abs"  → sort by absolute error  |on - off|
    sort_by = "rel"  → sort by relative error  |on - off| / max(|on|,|off|)
    sort_by = "val"  → sort by max magnitude    max(|on|,|off|)  (largest dims)
    sort_by = "real" → sort by ON's actual value (signed), largest first.
                       Useful for logits where positive values dominate
                       sampling probability; ranking by magnitude surfaces
                       "never selected" negative tokens instead.
    """
    abs_err = (on_vec - off_vec).abs()
    rel_err = abs_err / torch.maximum(on_vec.abs(), off_vec.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    elif sort_by == "real":
        sort_key = on_vec
    else:  # "val"
        sort_key = torch.maximum(on_vec.abs(), off_vec.abs())
    _, idx = sort_key.topk(min(topk, sort_key.numel()))
    idx = idx.to(torch.long)
    print(f"\n  [{label}]  top-{topk} dims (sort by {sort_by})")
    print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}  {'REL_ERR':>12s}")
    for k, i in enumerate(idx.tolist()):
        print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
              f"  {float(abs_err[i]):>12.6e}  {float(rel_err[i]):>12.6e}")


def _print_topk_2d(on_t: torch.Tensor, off_t: torch.Tensor,
                   mask: torch.Tensor, topk: int, sort_by: str, label: str):
    """Print top-K (batch, pos) elements from two 2D tensors.

    sort_by = "abs" → sort by absolute error  |on - off|
    sort_by = "rel" → sort by relative error    |on - off| / max(|on|,|off|)
    sort_by = "val" → sort by max magnitude      max(|on|,|off|)
    """
    abs_err = (on_t - off_t).abs()
    rel_err = abs_err / torch.maximum(on_t.abs(), off_t.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    else:  # "val"
        sort_key = torch.maximum(on_t.abs(), off_t.abs())
    # Mask invalid positions to -inf so they never sort into top-K
    if mask is not None:
        sort_key = sort_key.clone()
        sort_key[~mask.to(sort_key.device)] = float("-inf")
    flat, idx = sort_key.flatten().topk(min(topk, sort_key.numel()))
    # Convert flat indices → (row, col)
    rows = idx // sort_key.shape[1]
    cols = idx % sort_key.shape[1]
    print(f"\n  [{label}]  top-{topk} positions (sort by {sort_by})")
    print(f"  {'Seq_idx':>7s} {'POS':>6s}  {'ON':>14s}  {'OFF':>14s}  "
          f"{'ABS_ERR':>12s}  {'REL_ERR':>12s}")
    for k in range(len(flat)):
        r, c = int(rows[k]), int(cols[k])
        print(f"  {r:>7d} {c:>6d}  {float(on_t[r,c]):>14.6e}  "
              f"{float(off_t[r,c]):>14.6e}"
              f"  {float(abs_err[r,c]):>12.6e}  {float(rel_err[r,c]):>12.6e}")


def _print_summary(all_results: list[CheckResult]):
    print(_SEP_DOUBLE + "\n  SUMMARY")
    print(_SEP_DOUBLE)
    hdr = (f"  {'NAME':<28s} {'SHAPE':<18s} {'ABS_MAX':>12s}  {'REL_MAX':>10s}  "
           f"{'PEARSON_R':>10s}  {'STATUS':>8s}")
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for r in all_results:
        m = r.metrics
        shape = m.get("shape", m.get("error", "—"))
        am = f"{m.get('abs_max', float('nan')):.6e}" if "abs_max" in m else "—"
        rm = f"{m.get('rel_max', float('nan')):.6e}" if "rel_max" in m else "—"
        pr = f"{m.get('pearson_r', float('nan')):.6f}" if m.get('pearson_r') is not None else "—"
        s  = f"  {_CHECK} PASS" if r.passed else f"  {_CROSS} FAIL"
        print(f"  {r.name:<28s} {shape:<18s} {am:>12s}  {rm:>10s}  {pr:>10s}  {s}")
    print(_SEP_DOUBLE + "\n")


def _dump_json(all_results: list[CheckResult], path: str,
               dir_on: str, dir_off: str, tag: str, dir_off2: str | None):
    record = {"dir_on": dir_on, "dir_off": dir_off}
    if dir_off2:
        record["dir_off2"] = dir_off2
    if tag:
        record["tag"] = tag
    record["results"] = [{k: v for k, v in r.metrics.items()} | {"name": r.name, "passed": r.passed}
                         for r in all_results]
    record["all_passed"] = all(r.passed for r in all_results)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {path}\n")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Precision comparison — ON vs OFF diagnostic dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--dir-off2", default=None,
                    help="Second OFF directory for OFF-vs-OFF baseline")
    ap.add_argument("--tag", default=None,
                    help="Dump tag for 2D files (e.g. 'old', 'train')")
    ap.add_argument("--tag-on", default=None, help="Override tag for ON")
    ap.add_argument("--tag-off", default=None, help="Override tag for OFF")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D comparison (default: 1e-5)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific attn layer (default: all layers)")
    ap.add_argument("--mask-file", default=None,
                    help="External label_mask.pt file (auto-trims last True per row)")
    ap.add_argument("--no-trim", action="store_true",
                    help="Skip last-True trimming of label_mask")
    ap.add_argument("--output", "-o", default=None, help="Save report as JSON")
    ap.add_argument("--topk", type=int, default=0,
                    help="Print top-K worst elements (0 = disabled)")
    ap.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="Sort top-K by abs error / rel error / max magnitude (val)")
    args = ap.parse_args()

    tag_on  = args.tag_on  or args.tag
    tag_off = args.tag_off or args.tag

    _print_header(args.dir_on, args.dir_off, args.dir_off2,
                  tag_on, args.mask_file, args.layer)

    # Resolve label mask (with trim)
    label_mask = _resolve_label_mask(args.mask_file, args.dir_off,
                                     trim_last=not args.no_trim)
    if label_mask is not None:
        print(f"  label_mask: {_fmt_shape(tuple(label_mask.shape))}  "
              f"active={int(label_mask.sum())}\n")

    all_results: list[CheckResult] = []
    stop = False

    # ── ① position_ids ──
    r = cmp_position_ids(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_pos_ids(r)
        if not r.passed:
            stop = True

    # ── ② RoPE freqs (pre-apply angle table, suffix-aligned) ──
    if not stop:
        r = cmp_rope_freqs(args.dir_on, args.dir_off)
        if r:
            all_results.append(r)
            _print_rope_freqs(r)
            if not r.passed:
                stop = True

    # ── ②b RoPE encoding (post-apply rotated Q/K) ──
    if not stop:
        r = cmp_rope_postqk(args.dir_on, args.dir_off)
        if r:
            all_results.append(r)
            _print_rope(r)
            if not r.passed:
                stop = True

    # ── ③ attn_output per-layer cos ──
    if not stop:
        r = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
        if r:
            all_results.append(r)
            _print_per_layer(r)

    # ── ④ First-token (attn[0] + logits[0]) ──
    if not stop:
        for r in cmp_first_token(args.dir_on, args.dir_off):
            all_results.append(r)
            _print_first_token(r)

        # Top-K per-dimension errors
        if args.topk > 0:
            last = _get_num_layers(args.dir_on) or _get_num_layers(args.dir_off)
            # first_token_attn — per-dim top-K
            if last:
                a = _load_attn_output(args.dir_on, last)
                b = _load_attn_output(args.dir_off, last)
                if a is not None and b is not None:
                    on_token0 = (a.squeeze(1) if a.dim() == 3 else a)[0].cpu()
                    off_token0 = (b.squeeze(1) if b.dim() == 3 else b)[0].cpu()
                    _print_topk_vec(on_token0, off_token0, args.topk, "val",
                                    "first_token_attn")
            # first_token_logits — per-dim top-K (real = ON signed value,
            # so positive logits — the tokens actually selectable by
            # sampling — surface first, not the large-magnitude negatives)
            logits_on = _load_tensor(args.dir_on,  "logits.pt")
            logits_off = _load_tensor(args.dir_off, "logits.pt")
            if logits_on is not None and logits_off is not None:
                first_token_pair = _logits_first_token(logits_on, logits_off)
                if first_token_pair is not None:
                    _print_topk_vec(first_token_pair[0].cpu(), first_token_pair[1].cpu(),
                                    args.topk, "real", "first_token_logits")

    # ── ④b Logits (full packed alignment via attention_mask + dual pointer) ──
    if not stop:
        r = cmp_logits_packed(args.dir_on, args.dir_off)
        if r:
            all_results.append(r)
            _print_logits_packed(r)

    # ── ⑤ 2D comparison (entropy / logp) ──
    if tag_on:
        for fname, cname in [("entropy", "entropy"), ("logprobs", "logp")]:
            fn = f"{fname}_{tag_on}.pt"
            t1 = _load_tensor(args.dir_on, fn)
            t2 = _load_tensor(args.dir_off, f"{fname}_{tag_off or tag_on}.pt")
            if t1 is None or t2 is None:
                continue
            r = cmp_2d(args.dir_on, args.dir_off, fn,
                       f"{cname}_{tag_on}", label_mask, args.atol)
            all_results.append(r)
            print(_SEP_THIN)
            m = r.metrics
            print(f"  [{r.name}]  shape={m.get('shape','—')}  active={m.get('active','—')}  "
                  f"abs_max={m.get('abs_max',0):.6e}  rel_max={m.get('rel_max',0):.6e}  "
                  f"pearson={m.get('pearson_r',float('nan')):.8f}")
            print(f"  abs_mean={m.get('abs_mean',0):.6e}  rel_mean={m.get('rel_mean',0):.6e}  "
                  f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
            # Top-K worst positions
            if args.topk > 0:
                _print_topk_2d(t1.cpu(), t2.cpu(), label_mask, args.topk,
                               args.sort_err, r.name)
            print()

    # ── ⑥ OFF vs OFF baseline ──
    if args.dir_off2:
        print(_SEP_DOUBLE + "\n  [BASELINE]  OFF vs OFF2 Noise Floor\n" + _SEP_DOUBLE)
        # position_ids baseline
        rb = cmp_position_ids(args.dir_off, args.dir_off2)
        if rb:
            rb.name = "bl_pos_ids"
            print(f"  bl_pos_ids: {_CHECK if rb.passed else _CROSS} {'PASS' if rb.passed else 'FAIL'}\n")
            all_results.append(rb)
        # 2D baseline
        if tag_on:
            for fname, cname in [("entropy", "entropy"), ("logprobs", "logp")]:
                fn = f"{fname}_{tag_on}.pt"
                r = cmp_2d(args.dir_off, args.dir_off2, fn,
                           f"bl_{cname}_{tag_on}", label_mask, args.atol)
                all_results.append(r)
                m = r.metrics
                print(_SEP_THIN)
                print(f"  [{r.name}]  active={m.get('active','—')}  "
                      f"abs_max={m.get('abs_max',0):.6e}  rel_max={m.get('rel_max',0):.6e}  "
                      f"pearson={m.get('pearson_r',float('nan')):.8f}")
                print()

    # ── Summary ──
    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_on, args.dir_off,
                   tag_on, args.dir_off2)


if __name__ == "__main__":
    main()
