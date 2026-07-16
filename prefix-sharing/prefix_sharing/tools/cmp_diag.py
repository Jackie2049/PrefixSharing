"""verl080 精度比较 — ON vs OFF（自包含，不依赖 v070 cmp_diag）。

本文件**完全自包含**，不从 ``cmp_diag`` / ``diagnostic_dump`` import 任何东西，
便于将来独立维护（verl070 对应文件可能被废弃）。

对比项分两类：

  **packed（suffix 对齐）**
    - attention_output per-layer cos   每层 attention 输出余弦相似度
    - packed_token                      packed[pos]（attn[pos] + logits[pos]，--token 指定 pos，默认 0）
    - logits packed                     全 packed logits suffix 对齐对比

  **2D（v080 特有，restore 后 ``[B, L_max]``）**
    - logprobs / entropy                直接逐元素对比（ON/OFF 同坐标系）

suffix 对齐逻辑（v080）：ON 物理裁剪后 packed 只含 suffix 区段，OFF 含完整序列。
用 OFF 的 ``cu_seqlens_q.pt`` + ON 的 ``prefix_lens.pt`` 构建 1D suffix mask，
从 OFF 完整 packed 提取与 ON 对应的 suffix 段，再逐 token 比对。

dump 文件约定（``diagnostic_dump_verl080``）::

    logprobs_{tag}.pt       [B, L_max]       restore 后 2D log_probs
    entropy_{tag}.pt        [B, L_max]       2D entropy
    attention_mask_{tag}.pt [B, L_max] bool  [0,L_i-1) 所有 predict 有效位
    label_mask_{tag}.pt     [B, L_max] bool  [prompt-last,L_i-1) PPO loss 范围
    logits.pt               [N, V//tp]       packed logits（ON 裁剪后 / OFF 完整）
    attn_outputs.pt         dict {layer: [N, hidden]}  per-layer packed attn output
    rope_freqs.pt           dict {layer: [T,1,1,D]}    per-token RoPE 角度（ON/OFF 同款）
    rope_preqk.pt           dict {layer: [T,H,D]}      旋转前 Q/K（pre-RoPE）
    rope_postqk.pt          dict {layer: [T,H,D]}      旋转后 Q/K（post-RoPE）
    prefix_lens.pt          [B]              ON=plan.prefix_lens / OFF=全0
    cu_seqlens_q.pt         [B+1]            NestedTensor offsets（ON 裁剪后 / OFF 完整）
    cu_seqlens_q_logits.pt  [B+1]            logits packed 边界（同上）

Usage:
    # 完整对比（attn per-layer + packed_token + logits + logprobs + entropy）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off --tag old

    # 只看某一层 attention（1-indexed）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off \\
        --tag old --layer 12

    # top-K 误差最大位置（2D + packed_token）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off \\
        --tag old --topk 20

    # OFF vs OFF baseline（噪声底）
    python cmp_diag_verl080.py --dir-on ./dump_off --dir-off ./dump_off2 \\
        --tag old

Parameters:
    --dir-on     (必需) ON dump 目录
    --dir-off    (必需) OFF dump 目录
    --dir-off2   (可选) 第二个 OFF 目录，OFF-vs-OFF baseline
    --tag        (必需) 2D 文件标签 old / train
    --mask       (可选) 2D 对比 mask: label(默认) / attention / none
    --layer      (可选) 只对比指定层 attention (1-indexed)，不传则所有层
    --atol       (可选) 2D 对比绝对容差，默认 1e-5
    --topk       (可选) top-K 误差位置（0=关闭）
    --sort-err   (可选) top-K 排序: abs(默认) / rel / val
    -o, --output (可选) JSON 报告
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import torch

_SEP_DOUBLE = "=" * 70
_SEP_SINGLE = "-" * 70
_SEP_THIN = "─" * 70
_CHECK = "✓"
_CROSS = "✗"

# attn per-layer cos 通过阈值（logits/attn 向量级）
_COS_AVG_PASS = 0.9999
_COS_MIN_PASS = 0.999


@dataclass
class CheckResult:
    name: str
    passed: bool = True
    metrics: dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
#  Metric helpers
# ════════════════════════════════════════════════════════════════

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cosine similarity along ``dim``, promoted to float32 for bf16 noise immunity.

    Near-zero vectors (both norms < sqrt(epsilon)) are treated as identical, returning 1.0.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    epsilon = 1e-8
    norm_a = a.norm(dim=dim)
    norm_b = b.norm(dim=dim)
    denominator = (norm_a * norm_b).clamp(min=epsilon)
    cosine_values = (a * b).sum(dim=dim) / denominator
    near_zero_mask = (norm_a < epsilon ** 0.5) & (norm_b < epsilon ** 0.5)
    return torch.where(near_zero_mask, torch.ones_like(cosine_values), cosine_values)


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


def _pearson_r(on_tensor: torch.Tensor, off_tensor: torch.Tensor,
               mask: torch.Tensor | None = None) -> float:
    values_on = (on_tensor[mask] if mask is not None else on_tensor.flatten()).to(torch.float64)
    values_off = (off_tensor[mask] if mask is not None else off_tensor.flatten()).to(torch.float64)
    if values_on.numel() < 2:
        return float("nan")
    mean_on, mean_off = values_on.mean(), values_off.mean()
    covariance = ((values_on - mean_on) * (values_off - mean_off)).sum()
    std_on = ((values_on - mean_on) ** 2).sum().sqrt()
    std_off = ((values_off - mean_off) ** 2).sum().sqrt()
    return float("nan") if std_on == 0 or std_off == 0 else float(covariance / (std_on * std_off))


def _vec_metrics(on_vec: torch.Tensor, off_vec: torch.Tensor) -> dict:
    error_metrics = _error_abs_rel(on_vec, off_vec)
    cosine_val = float(_cosine_sim(on_vec, off_vec, dim=-1))
    pearson_correlation = _pearson_r(on_vec, off_vec)
    return {"mean_abs": error_metrics["abs_mean"], "max_abs": error_metrics["abs_max"],
            "rel_max": error_metrics["rel_max"], "rel_mean": error_metrics["rel_mean"],
            "cos": cosine_val, "pearson": pearson_correlation}


# ════════════════════════════════════════════════════════════════
#  Loading helpers
# ════════════════════════════════════════════════════════════════

def _load_tensor(dir_path: str, filename: str) -> torch.Tensor | None:
    filepath = os.path.join(dir_path, filename)
    return torch.load(filepath, weights_only=True).float() if os.path.exists(filepath) else None


def _load_logits(dir_path: str) -> torch.Tensor | None:
    """Load packed logits from ``logits.pt``.

    Multi-rank assembly (tp vocab concat) is done by ``assemble_dump.py``
    before cmp is called; cmp works on flat single-card data only.
    """
    return _load_tensor(dir_path, "logits.pt")


def _load_packed_meta(dir_path: str,
                      cumulative_filename: str = "cu_seqlens_q.pt") -> dict | None:
    """Load cu_seqlens + prefix_lens (needed for suffix alignment)."""
    filepath = os.path.join(dir_path, cumulative_filename)
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
    """加载单层 attn_output（attn_outputs.pt = dict {layer: tensor}）。"""
    filepath = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(filepath):
        return None
    attn_dict = torch.load(filepath, weights_only=True)
    return attn_dict.get(layer) if isinstance(attn_dict, dict) else None


def _load_attn_grad(dir_path: str, layer: int) -> torch.Tensor | None:
    """加载单层 attn_grad（attn_grads.pt = dict {layer: tensor}）。"""
    filepath = os.path.join(dir_path, "attn_grads.pt")
    if not os.path.exists(filepath):
        return None
    grad_dict = torch.load(filepath, weights_only=True)
    return grad_dict.get(layer) if isinstance(grad_dict, dict) else None


def _get_num_layers(dir_path: str) -> int:
    filepath = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(filepath):
        return 0
    attn_dict = torch.load(filepath, weights_only=True)
    return max(attn_dict.keys()) if isinstance(attn_dict, dict) and attn_dict else 0


# ════════════════════════════════════════════════════════════════
#  Packed suffix alignment
# ════════════════════════════════════════════════════════════════

def _build_alignment_mask(cu_seqlens: torch.Tensor,
                          prefix_lens: torch.Tensor,
                          total_tokens: int) -> torch.Tensor:
    """构建 1D suffix mask ``[total_tokens]``：True = suffix token。

    用 OFF 的 cu_seqlens + ON 的 prefix_lens：每行 ``[cu[i]+prefix_len[i] : cu[i+1]]``
    为 suffix 区段。应用于 OFF 完整 packed 提取与 ON（裁剪后只含 suffix）对应的段。

    例：cu=[0,7,13], prefix_lens=[3,4], total=13 → [0,0,0,1,1,1,1, 0,0,0,0,1,1]
    """
    mask = torch.zeros(total_tokens, dtype=torch.bool)
    for seq_idx in range(cu_seqlens.shape[0] - 1):
        prefix_len = int(prefix_lens[seq_idx])
        start = int(cu_seqlens[seq_idx]) + prefix_len
        end = int(cu_seqlens[seq_idx + 1])
        mask[start:end] = True
    return mask


def _align_packed(on_tensor: torch.Tensor, off_tensor: torch.Tensor,
                  alignment_mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """ON（suffix-only）与 OFF（full-packed）按 alignment_mask 对齐。

    返回 ``(on, off_suffix)``，其中 ``off_suffix = off[alignment_mask]``，
    shape[0] == on_tensor.shape[0]。
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


# ════════════════════════════════════════════════════════════════
#  Logits helpers
# ════════════════════════════════════════════════════════════════

def _aligned_vec_at_pos(
    on_tensor: torch.Tensor | None,
    off_tensor: torch.Tensor | None,
    is_attn: bool,
    pos: int,
    align_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """ON(suffix-only)/OFF(full-packed) 的 packed 张量 **suffix 对齐后** 取 [pos]。

    ON 物理裁剪后只含 suffix，OFF 含完整序列，两者 token 不直接对应——必须先用
    align_mask 把 OFF 的 suffix 段抽出来与 ON 对齐，再取 [pos]。pos 索引的是
    对齐后的 suffix-packed 空间（ON/OFF 一致，指向同一个 token）。

    - is_attn=True：attn_output ``[T,1,hidden]`` → ``[T,hidden]``。
    - is_attn=False：logits → ``[N, V]``（vocab 恒在最后一维）。
    返回 (on_vec, off_vec)（同 token、同向量长度），或 None（数据缺失 / pos 越界 /
    对齐失败）。
    """
    if on_tensor is None or off_tensor is None:
        return None
    if is_attn:
        on = on_tensor.squeeze(1) if on_tensor.dim() == 3 else on_tensor
        off = off_tensor.squeeze(1) if off_tensor.dim() == 3 else off_tensor
    else:
        on = on_tensor.reshape(-1, on_tensor.size(-1))
        off = off_tensor.reshape(-1, off_tensor.size(-1))
    if align_mask is not None and on.shape[0] != off.shape[0]:
        try:
            on, off = _align_packed(on, off, align_mask)
        except ValueError:
            return None
    n = min(on.shape[0], off.shape[0])
    if pos < 0 or pos >= n:
        return None
    return on[pos].contiguous(), off[pos].contiguous()


def _logits_ensure_token_major(logits_on: torch.Tensor, logits_off: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """确保 logits 为 2D [N, V]（token-major），vocab 在最后一维。"""
    return (logits_on.reshape(-1, logits_on.size(-1)).contiguous(),
            logits_off.reshape(-1, logits_off.size(-1)).contiguous())


# ════════════════════════════════════════════════════════════════
#  Packed compare: attention_output / packed_token / logits
# ════════════════════════════════════════════════════════════════

def _cos_for_layer(on_tensor: torch.Tensor, off_tensor: torch.Tensor,
                   align_mask: torch.Tensor | None = None) -> dict:
    """Single layer attn_output alignment + per-token cosine."""
    if on_tensor.dim() == 3:
        on_tensor, off_tensor = on_tensor.squeeze(1), off_tensor.squeeze(1)
    if align_mask is not None and on_tensor.shape[0] != off_tensor.shape[0]:
        on_tensor, off_tensor = _align_packed(on_tensor, off_tensor, align_mask)
    cosine_values = _cosine_sim(on_tensor, off_tensor, dim=-1)
    return {"cos_avg": float(cosine_values.mean()), "cos_min": float(cosine_values.min()),
            "n_tokens": on_tensor.shape[0]}


def _build_attn_align_mask(dir_on: str, dir_off: str) -> torch.Tensor | None:
    """Build suffix alignment mask from OFF cu_seqlens + ON prefix_lens."""
    on_meta = _load_packed_meta(dir_on)
    off_meta = _load_packed_meta(dir_off)
    if on_meta is None or off_meta is None:
        return None
    off_cu_seqlens = off_meta["cu_seqlens"]
    token_count = int(off_cu_seqlens[-1]) if off_cu_seqlens.numel() > 0 else 0
    if token_count == 0:
        return None
    return _build_alignment_mask(off_cu_seqlens, on_meta["prefix_lens"], token_count)


def cmp_attn_layer(dir_on: str, dir_off: str,
                   layer: int | None) -> CheckResult | None:
    """attention_output per-layer cos（suffix 对齐）。

    单层模式（layer 给定）：返回该层 cos。全层模式：返回所有层 cos 汇总。
    """
    align_mask = _build_attn_align_mask(dir_on, dir_off)

    if layer is not None:
        on_tensor = _load_attn_output(dir_on, layer)
        off_tensor = _load_attn_output(dir_off, layer)
        if on_tensor is None or off_tensor is None:
            return None
        needs_alignment = align_mask is not None and on_tensor.shape[0] != off_tensor.shape[0]
        try:
            layer_metrics = _cos_for_layer(on_tensor, off_tensor, align_mask if needs_alignment else None)
        except ValueError as exc:
            return CheckResult(name=f"attn_L{layer}", passed=False,
                               metrics={"error": str(exc)})
        layer_metrics["layer"] = layer
        return CheckResult(name=f"attn_L{layer}",
                           passed=layer_metrics["cos_avg"] > _COS_AVG_PASS
                           and layer_metrics["cos_min"] > _COS_MIN_PASS, metrics=layer_metrics)

    filepath_on = os.path.join(dir_on, "attn_outputs.pt")
    filepath_off = os.path.join(dir_off, "attn_outputs.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    attn_dict_on = torch.load(filepath_on, weights_only=True)
    attn_dict_off = torch.load(filepath_off, weights_only=True)
    if not isinstance(attn_dict_on, dict) or not isinstance(attn_dict_off, dict):
        return None

    results = {}
    for layer_idx in sorted(set(attn_dict_on.keys()) & set(attn_dict_off.keys())):
        on_tensor, off_tensor = attn_dict_on[layer_idx], attn_dict_off[layer_idx]
        needs_alignment = align_mask is not None and on_tensor.shape[0] != off_tensor.shape[0]
        try:
            results[layer_idx] = _cos_for_layer(on_tensor, off_tensor, align_mask if needs_alignment else None)
        except ValueError as exc:
            results[layer_idx] = {"error": str(exc)}
    return CheckResult(name="attn_per_layer", passed=True,
                       metrics={"layers": results})


def cmp_attn_grads(dir_on: str, dir_off: str,
                     layer: int | None) -> CheckResult | None:
    """attn_grads per-layer cosine (suffix alignment, same as attn_output).

    Single-layer mode (layer given): returns cos for that layer.
    Full-layer mode: aggregates all layers.
    """
    align_mask = _build_attn_align_mask(dir_on, dir_off)

    if layer is not None:
        on_tensor = _load_attn_grad(dir_on, layer)
        off_tensor = _load_attn_grad(dir_off, layer)
        if on_tensor is None or off_tensor is None:
            return None
        needs_alignment = align_mask is not None and on_tensor.shape[0] != off_tensor.shape[0]
        try:
            layer_metrics = _cos_for_layer(on_tensor, off_tensor, align_mask if needs_alignment else None)
        except ValueError as exc:
            return CheckResult(name=f"attn_grad_L{layer}", passed=False,
                               metrics={"error": str(exc)})
        layer_metrics["layer"] = layer
        return CheckResult(name=f"attn_grad_L{layer}",
                           passed=layer_metrics["cos_avg"] > _COS_AVG_PASS
                           and layer_metrics["cos_min"] > _COS_MIN_PASS, metrics=layer_metrics)

    filepath_on = os.path.join(dir_on, "attn_grads.pt")
    filepath_off = os.path.join(dir_off, "attn_grads.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    grad_dict_on = torch.load(filepath_on, weights_only=True)
    grad_dict_off = torch.load(filepath_off, weights_only=True)
    if not isinstance(grad_dict_on, dict) or not isinstance(grad_dict_off, dict):
        return None

    results = {}
    for layer_idx in sorted(set(grad_dict_on.keys()) & set(grad_dict_off.keys())):
        on_tensor, off_tensor = grad_dict_on[layer_idx], grad_dict_off[layer_idx]
        needs_alignment = align_mask is not None and on_tensor.shape[0] != off_tensor.shape[0]
        try:
            results[layer_idx] = _cos_for_layer(on_tensor, off_tensor, align_mask if needs_alignment else None)
        except ValueError as exc:
            results[layer_idx] = {"error": str(exc)}
    return CheckResult(name="attn_grad_per_layer", passed=True,
                       metrics={"layers": results})


def cmp_packed_token(dir_on: str, dir_off: str,
                     pos: int = 0, layer: int | None = None,
                     align_mask: torch.Tensor | None = None) -> list[CheckResult]:
    """packed[pos] 对比（**suffix 对齐后**）：attn[pos]（可指定层）+ logits[pos]（仅最后一层）。

    ON 是裁剪后的 suffix-only packed，OFF 是完整 packed，两者 token **不直接对应**——
    必须先用 align_mask（OFF cu_seqlens + ON prefix_lens）把 OFF 的 suffix 段抽出来
    与 ON 对齐，再取 [pos]。pos 索引的是对齐后的 suffix-packed 空间（ON/OFF 一致）。

    - pos：对齐后 suffix-packed 里的位置（单个 int，默认 0）。
    - attn：用 *layer*（默认最后一层）。对比第 1 层可区分
      "结构错（第 1 层就偏）" vs "数值累积（第 1 层完美、深层才偏）"。
    - logits：永远最后一层。
    - align_mask：可选，复用调用方已构建的；None 则内部构建。
    """
    if align_mask is None:
        align_mask = _build_attn_align_mask(dir_on, dir_off)
    results: list[CheckResult] = []
    attn_layer = layer if layer is not None else (
        _get_num_layers(dir_on) or _get_num_layers(dir_off))

    if attn_layer:
        on_tensor = _load_attn_output(dir_on, attn_layer)
        off_tensor = _load_attn_output(dir_off, attn_layer)
        vecs = _aligned_vec_at_pos(on_tensor, off_tensor, True, pos, align_mask)
        if vecs is None:
            results.append(CheckResult(
                name=f"attn_L{attn_layer}_pos{pos}",
                metrics={"error": f"无法对齐或 pos {pos} 越界"}))
        else:
            results.append(CheckResult(
                name=f"attn_L{attn_layer}_pos{pos}",
                metrics=_vec_metrics(vecs[0], vecs[1])))

    logits_on = _load_logits(dir_on)
    logits_off = _load_logits(dir_off)
    vecs = _aligned_vec_at_pos(logits_on, logits_off, False, pos, align_mask)
    if vecs is None:
        results.append(CheckResult(
            name=f"logits_pos{pos}",
            metrics={"error": f"无法对齐或 pos {pos} 越界"}))
    else:
        results.append(CheckResult(
            name=f"logits_pos{pos}",
            metrics=_vec_metrics(vecs[0], vecs[1])))
    return results


# ══════════════════════════════════════════════════════════════════
#  Post-RoPE Q/K compare: per-layer + packed_token
# ══════════════════════════════════════════════════════════════════

# RoPE 对比阶段：**先 pre（旋转前，rope_preqk.pt）后 post（旋转后，rope_postqk.pt）**。
# (stage, fname, label) — label 用作结果名前缀与打印 section 头。
_ROPE_STAGES: list[tuple[str, str, str]] = [
    ("pre", "rope_preqk.pt", "rope_preqk"),
    ("post", "rope_postqk.pt", "rope_postqk"),
]


def _load_rope_postqk(dir_path: str, layer: int, filename: str = "rope_postqk.pt"
                   ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load Q/K for a single layer from ``filename`` (rope_postqk.pt=post, rope_preqk.pt=pre).

    Returns ``(query, key)`` or ``(None, None)``.
    """
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        return None, None
    rope_dict = torch.load(filepath, weights_only=True)
    if not isinstance(rope_dict, dict):
        return None, None
    entry = rope_dict.get(layer)
    if entry is None:
        return None, None
    return entry.get("query"), entry.get("key")


def _rope_postqk_cos_for_layer(query_on: torch.Tensor, key_on: torch.Tensor,
                             query_off: torch.Tensor, key_off: torch.Tensor,
                             align_mask: torch.Tensor | None = None) -> dict:
    """Single layer Q/K suffix alignment + per-token cosine (Q and K separately)."""
    # Q/K shape: [T, H, D] → flatten head*dim dimensions for cosine
    query_on_flat = query_on.reshape(query_on.shape[0], -1)
    query_off_flat = query_off.reshape(query_off.shape[0], -1)
    key_on_flat = key_on.reshape(key_on.shape[0], -1)
    key_off_flat = key_off.reshape(key_off.shape[0], -1)

    if align_mask is not None and query_on.shape[0] != query_off.shape[0]:
        query_on_flat, query_off_flat = _align_packed(query_on_flat, query_off_flat, align_mask)
        key_on_flat, key_off_flat = _align_packed(key_on_flat, key_off_flat, align_mask)

    query_cosine = _cosine_sim(query_on_flat, query_off_flat, dim=-1)
    key_cosine = _cosine_sim(key_on_flat, key_off_flat, dim=-1)
    return {
        "n_tokens": query_on_flat.shape[0],
        "Q_cos_avg": float(query_cosine.mean()), "Q_cos_min": float(query_cosine.min()),
        "K_cos_avg": float(key_cosine.mean()), "K_cos_min": float(key_cosine.min()),
        "Q_max_diff": float((query_on_flat - query_off_flat).abs().max()),
        "K_max_diff": float((key_on_flat - key_off_flat).abs().max()),
    }


def _cmp_rope_stage_layer(dir_on: str, dir_off: str, layer: int | None,
                           filename: str, label: str) -> CheckResult | None:
    """Single stage Q/K per-layer cosine (suffix aligned)."""
    align_mask = _build_attn_align_mask(dir_on, dir_off)

    if layer is not None:
        query_on, key_on = _load_rope_postqk(dir_on, layer, filename)
        query_off, key_off = _load_rope_postqk(dir_off, layer, filename)
        if query_on is None or query_off is None:
            return None
        needs_alignment = align_mask is not None and query_on.shape[0] != query_off.shape[0]
        try:
            layer_metrics = _rope_postqk_cos_for_layer(query_on, key_on, query_off, key_off,
                                         align_mask if needs_alignment else None)
        except ValueError as exc:
            return CheckResult(name=f"{label}_L{layer}", passed=False,
                               metrics={"error": str(exc)})
        layer_metrics["layer"] = layer
        passes_threshold = (layer_metrics["Q_cos_avg"] > _COS_AVG_PASS and layer_metrics["Q_cos_min"] > _COS_MIN_PASS
              and layer_metrics["K_cos_avg"] > _COS_AVG_PASS and layer_metrics["K_cos_min"] > _COS_MIN_PASS)
        return CheckResult(name=f"{label}_L{layer}", passed=passes_threshold, metrics=layer_metrics)

    # All layers
    on_filepath = os.path.join(dir_on, filename)
    off_filepath = os.path.join(dir_off, filename)
    if not os.path.exists(on_filepath) or not os.path.exists(off_filepath):
        return None
    on_dict = torch.load(on_filepath, weights_only=True)
    off_dict = torch.load(off_filepath, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None

    results = {}
    for layer_idx in sorted(set(on_dict.keys()) & set(off_dict.keys())):
        on_entry, off_entry = on_dict[layer_idx], off_dict[layer_idx]
        query_on, key_on = on_entry.get("query"), on_entry.get("key")
        query_off, key_off = off_entry.get("query"), off_entry.get("key")
        if query_on is None or query_off is None:
            continue
        needs_alignment = align_mask is not None and query_on.shape[0] != query_off.shape[0]
        try:
            results[layer_idx] = _rope_postqk_cos_for_layer(query_on, key_on, query_off, key_off,
                                                    align_mask if needs_alignment else None)
        except ValueError as exc:
            results[layer_idx] = {"error": str(exc)}
    return CheckResult(name=f"{label}_per_layer", passed=True,
                       metrics={"layers": results})


def cmp_rope_postqk_layer(dir_on: str, dir_off: str, layer: int | None,
                        stage: str = "post") -> CheckResult | None:
    """Q/K per-layer cosine（suffix 对齐），单 stage。

    stage="pre" → rope_preqk.pt（旋转前），stage="post" → rope_postqk.pt（旋转后）。
    调用方按 pre → rope_freqs → post 顺序分别调用，便于定位分歧出现在 RoPE 哪一步。
    """
    if stage == "pre":
        return _cmp_rope_stage_layer(dir_on, dir_off, layer, "rope_preqk.pt", "rope_preqk")
    return _cmp_rope_stage_layer(dir_on, dir_off, layer, "rope_postqk.pt", "rope_postqk")


def _rope_postqk_vec_at_pos(query_on: torch.Tensor | None, key_on: torch.Tensor | None,
                          query_off: torch.Tensor | None, key_off: torch.Tensor | None,
                          pos: int,
                          align_mask: torch.Tensor | None
                          ) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor] | None:
    """Q/K suffix-aligned at [pos], returns (query_on, query_off, key_on, key_off) flat vectors.

    Each vector flattened to [H*D], directly comparable via vec_metrics.
    """
    if query_on is None or query_off is None:
        return None
    query_on_flat = query_on.reshape(query_on.shape[0], -1)
    query_off_flat = query_off.reshape(query_off.shape[0], -1)
    key_on_flat = key_on.reshape(key_on.shape[0], -1) if key_on is not None else None
    key_off_flat = key_off.reshape(key_off.shape[0], -1) if key_off is not None else None

    if align_mask is not None and query_on.shape[0] != query_off.shape[0]:
        try:
            query_on_flat, query_off_flat = _align_packed(query_on_flat, query_off_flat, align_mask)
            if key_on_flat is not None:
                key_on_flat, key_off_flat = _align_packed(key_on_flat, key_off_flat, align_mask)
        except ValueError:
            return None
    aligned_count = min(query_on_flat.shape[0], query_off_flat.shape[0])
    if pos < 0 or pos >= aligned_count:
        return None
    query_on_vec = query_on_flat[pos].contiguous()
    query_off_vec = query_off_flat[pos].contiguous()
    key_on_vec = key_on_flat[pos].contiguous() if key_on_flat is not None else None
    key_off_vec = key_off_flat[pos].contiguous() if key_off_flat is not None else None
    return query_on_vec, query_off_vec, key_on_vec, key_off_vec


def _diag_rope_pos_fail(q_on: torch.Tensor | None, q_off: torch.Tensor | None,
                        pos: int, align_mask: torch.Tensor | None) -> str:
    """rope_postqk packed_token 取 [pos] 失败时的诊断串：区分 缺失 / 对齐失败 / pos 越界。"""
    if q_on is None or q_off is None:
        return f"rope_postqk 该层在 {'ON' if q_on is None else 'OFF'} 侧缺失"
    n_on, n_off = q_on.shape[0], q_off.shape[0]
    if align_mask is not None and n_on != n_off:
        msum = int(align_mask.sum())
        return (f"对齐失败: n_on={n_on} n_off={n_off} "
                f"align_mask(len={align_mask.shape[0]}, sum={msum}); "
                f"需 ON tokens==sum({msum}) 且 mask_len==n_off({n_off})")
    post = min(n_on, n_off)
    return f"pos {pos} 越界: 对齐后 token 数={post} (n_on={n_on}, n_off={n_off})"


def _cmp_rope_stage_token(dir_on: str, dir_off: str, pos: int, layer: int | None,
                           align_mask: torch.Tensor | None, filename: str,
                           label: str) -> list[CheckResult]:
    """Single stage Q/K packed[pos] (suffix aligned)."""
    if align_mask is None:
        align_mask = _build_attn_align_mask(dir_on, dir_off)
    results: list[CheckResult] = []
    rope_layer = layer if layer is not None else (
        _get_num_layers(dir_on) or _get_num_layers(dir_off))
    if rope_layer:
        query_on, key_on = _load_rope_postqk(dir_on, rope_layer, filename)
        query_off, key_off = _load_rope_postqk(dir_off, rope_layer, filename)
        vecs = _rope_postqk_vec_at_pos(query_on, key_on, query_off, key_off, pos, align_mask)
        if vecs is None:
            results.append(CheckResult(
                name=f"{label}_L{rope_layer}_pos{pos}",
                metrics={"error": _diag_rope_pos_fail(query_on, query_off, pos, align_mask)}))
        else:
            query_on_vec, query_off_vec, key_on_vec, key_off_vec = vecs
            results.append(CheckResult(
                name=f"{label}_L{rope_layer}_Q_pos{pos}",
                metrics=_vec_metrics(query_on_vec, query_off_vec)))
            if key_on_vec is not None and key_off_vec is not None:
                results.append(CheckResult(
                    name=f"{label}_L{rope_layer}_K_pos{pos}",
                    metrics=_vec_metrics(key_on_vec, key_off_vec)))
    return results


def cmp_rope_postqk_token(dir_on: str, dir_off: str,
                        pos: int = 0, layer: int | None = None,
                        align_mask: torch.Tensor | None = None,
                        stage: str = "post") -> list[CheckResult]:
    """Q/K packed[pos] 对比（**suffix 对齐后**），单 stage。

    stage="pre" → rope_preqk.pt（旋转前），stage="post" → rope_postqk.pt（旋转后）。
    对 Q、K 分别输出 {label}_L{layer_idx}_Q_pos{pos} / {label}_L{layer_idx}_K_pos{pos}。
    调用方按 pre → rope_freqs → post 顺序分别调用。
    """
    if stage == "pre":
        return _cmp_rope_stage_token(dir_on, dir_off, pos, layer, align_mask,
                                      "rope_preqk.pt", "rope_preqk")
    return _cmp_rope_stage_token(dir_on, dir_off, pos, layer, align_mask,
                                  "rope_postqk.pt", "rope_postqk")


def cmp_logits_packed(dir_on: str, dir_off: str) -> CheckResult | None:
    """全 packed logits suffix 对齐 + per-token cosine。"""
    logits_on = _load_logits(dir_on)
    logits_off = _load_logits(dir_off)
    if logits_on is None or logits_off is None:
        return None
    logits_on, logits_off = _logits_ensure_token_major(logits_on, logits_off)

    meta_on = _load_packed_meta(dir_on, "cu_seqlens_q_logits.pt")
    meta_off = _load_packed_meta(dir_off, "cu_seqlens_q_logits.pt")
    if meta_on is None or meta_off is None:
        return None
    total_off_tokens = int(meta_off["cu_seqlens"][-1]) if meta_off["cu_seqlens"].numel() > 0 else 0
    if total_off_tokens == 0 or logits_on.shape[0] == 0 or logits_off.shape[0] == 0:
        return None

    align_mask = _build_alignment_mask(meta_off["cu_seqlens"], meta_on["prefix_lens"], total_off_tokens)
    try:
        on_aligned, off_aligned = _align_packed(logits_on, logits_off, align_mask)
    except ValueError as exc:
        return CheckResult(name="logits", passed=False,
                           metrics={"error": str(exc),
                                    "n_on": logits_on.shape[0], "n_off": logits_off.shape[0]})

    cosine_values = _cosine_sim(on_aligned, off_aligned, dim=-1)
    cos_avg, cos_min = float(cosine_values.mean()), float(cosine_values.min())
    return CheckResult(name="logits",
                       passed=cos_avg > _COS_AVG_PASS and cos_min > _COS_MIN_PASS,
                       metrics={"n_tokens": on_aligned.shape[0],
                                "cos_avg": cos_avg, "cos_min": cos_min})


def _align_rope_freqs_layer(on_freqs: torch.Tensor, off_freqs: torch.Tensor,
                             align_mask: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """单层 rope_freqs（per-token [T,1,1,D]）suffix 对齐。

    返回 (on_aligned, off_aligned) [N,1,1,D]；对齐失败返回 None。
    供 cmp_rope_freqs（per-layer max_diff）与 cmp_rope_freqs_token（[pos] 角度向量）复用。
    ON/OFF 现在都是 per-token，直接对齐即可（不再从 raw 表重建）。
    """
    try:
        return _align_packed(on_freqs, off_freqs, align_mask)
    except ValueError:
        return None


def _load_rope_freqs_vec_at_pos(dir_on: str, dir_off: str, layer: int, pos: int,
                                align_mask: torch.Tensor | None = None
                                ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """加载 rope_freqs 对齐后 [pos] 的角度向量 [D]，返回 (on_vec, off_vec) 或 (None, None)。

    供 top-K 跨 stage 对齐用（freqs dim = Q/K dim % D，角度按 head_dim 共享）。
    """
    filepath_on = os.path.join(dir_on, "rope_freqs.pt")
    filepath_off = os.path.join(dir_off, "rope_freqs.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None, None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None, None
    if layer not in on_dict or layer not in off_dict:
        return None, None
    if align_mask is None:
        align_mask = _build_attn_align_mask(dir_on, dir_off)
    if align_mask is None:
        return None, None
    aligned_result = _align_rope_freqs_layer(on_dict[layer], off_dict[layer], align_mask)
    if aligned_result is None:
        return None, None
    on_aligned, off_aligned = aligned_result
    if pos < 0 or pos >= on_aligned.shape[0]:
        return None, None
    return on_aligned[pos].reshape(-1), off_aligned[pos].reshape(-1)


def cmp_rope_freqs(dir_on: str, dir_off: str,
                    layer: int | None = None) -> CheckResult | None:
    """对比 per-token RoPE 角度 — suffix 对齐，应精确相等 max_diff==0。

    ON/OFF 都存 per-token 角度 ``rope_freqs.pt`` {layer: [T,1,1,D]}（cos/sin 之前），
    suffix 对齐后逐元素比。角度是 RoPE 输入，应精确相等（max_diff==0）。
    ``layer`` 给定则只比该层。
    """
    filepath_on = os.path.join(dir_on, "rope_freqs.pt")
    filepath_off = os.path.join(dir_off, "rope_freqs.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None

    layers = sorted(set(on_dict.keys()) & set(off_dict.keys()))
    if layer is not None:
        layers = [layer_val for layer_val in layers if layer_val == layer]
    result_name = f"rope_freqs_L{layer}" if layer is not None else "rope_freqs"
    if not layers:
        return CheckResult(name=result_name, passed=False,
                           metrics={"error": f"layer {layer} 不在双方 rope_freqs 中"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    if align_mask is None:
        return CheckResult(name=result_name, passed=False,
                           metrics={"error": "cu_seqlens/prefix_lens 缺失"})

    max_diff_value = 0.0
    mismatches: list[dict] = []
    for layer_idx in layers:
        aligned_result = _align_rope_freqs_layer(on_dict[layer_idx], off_dict[layer_idx], align_mask)
        if aligned_result is None:
            continue
        on_aligned, off_aligned = aligned_result
        diff = (on_aligned - off_aligned).abs()
        layer_max_diff = float(diff.max())
        max_diff_value = max(max_diff_value, layer_max_diff)
        if layer_max_diff > 0:
            token_diff = diff.squeeze(1).squeeze(1).max(dim=-1)
            bad_mask = token_diff.values > 0
            for token_pos in bad_mask.nonzero(as_tuple=True)[0].tolist():
                token_pos = int(token_pos)
                dim_idx = int(token_diff.indices[token_pos])
                mismatches.append({
                    "layer": layer_idx, "token_idx": token_pos, "dim": dim_idx,
                    "on_val": float(on_aligned[token_pos, 0, 0, dim_idx]),
                    "off_val": float(off_aligned[token_pos, 0, 0, dim_idx]),
                    "diff": float(token_diff.values[token_pos]),
                })

    result_metrics: dict = {"max_diff": max_diff_value, "num_layers": len(layers)}
    if mismatches:
        result_metrics["mismatches"] = mismatches[:20]
        result_metrics["total_mismatches"] = len(mismatches)
    return CheckResult(name=result_name, passed=max_diff_value == 0.0, metrics=result_metrics)


def cmp_rope_freqs_token(dir_on: str, dir_off: str, pos: int,
                          layer: int | None = None,
                          align_mask: torch.Tensor | None = None) -> CheckResult | None:
    """rope_freqs 在对齐后 suffix-packed 位置 [pos] 的角度向量对比（应精确相等）。

    取 ``layer``（默认最后一层）对齐后第 ``pos`` 个 token 的角度向量 [D]，比 ON/OFF。
    角度是 RoPE 输入，应逐元素相等 → max_abs 应为 0。
    """
    filepath_on = os.path.join(dir_on, "rope_freqs.pt")
    filepath_off = os.path.join(dir_off, "rope_freqs.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None
    common = set(on_dict.keys()) & set(off_dict.keys())
    rf_layer = layer if layer is not None else (max(common) if common else 0)
    result_name = f"rope_freqs_L{rf_layer}_pos{pos}"
    if rf_layer not in on_dict or rf_layer not in off_dict:
        return CheckResult(name=result_name, metrics={"error": f"layer {rf_layer} 缺失"})

    if align_mask is None:
        align_mask = _build_attn_align_mask(dir_on, dir_off)
    if align_mask is None:
        return CheckResult(name=result_name, metrics={"error": "cu_seqlens/prefix_lens 缺失"})

    aligned_result = _align_rope_freqs_layer(on_dict[rf_layer], off_dict[rf_layer], align_mask)
    if aligned_result is None:
        return CheckResult(name=result_name, metrics={"error": "对齐失败"})
    on_a, off_a = aligned_result
    n = on_a.shape[0]
    if pos < 0 or pos >= n:
        return CheckResult(name=result_name,
                           metrics={"error": f"pos {pos} 越界: 对齐后 token 数={n}"})
    on_vec = on_aligned[pos].reshape(-1)
    off_vec = off_aligned[pos].reshape(-1)
    vec_result = _vec_metrics(on_vec, off_vec)
    return CheckResult(name=result_name, passed=vec_result["max_abs"] == 0.0, metrics=vec_result)


# ════════════════════════════════════════════════════════════════
#  Attention KV: ON expanded_kv vs OFF full_kv（prefix 复用校验）
# ════════════════════════════════════════════════════════════════

def _load_attn_kv(dir_path: str, layer: int,
                  filename: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load {key, value} for a layer from filename. Returns (key, value) or (None, None)."""
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        return None, None
    kv_dict = torch.load(filepath, weights_only=True)
    if not isinstance(kv_dict, dict):
        return None, None
    entry = kv_dict.get(layer)
    if entry is None:
        return None, None
    return entry.get("key"), entry.get("value")


def cmp_attn_kv(dir_on: str, dir_off: str,
                layer: int | None = None) -> CheckResult | None:
    """对比 ON expanded_kv vs OFF full_kv（K/V 分别），逐元素 max_diff + cos。

    两者都应是 full（prefix+suffix）且**逐元素相同**（prefix-sharing 的 KV 展开应精确还原
    完整 KV）。相同 → attention 输入一致，attention_output 差异必来自 attention 计算/mask；
    不同 → bug 在 build_kv 的 prefix 复用（store/expand）。
    """
    filepath_on = os.path.join(dir_on, "expanded_kv.pt")
    filepath_off = os.path.join(dir_off, "full_kv.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None
    layers = sorted(set(on_dict.keys()) & set(off_dict.keys()))
    if layer is not None:
        layers = [layer_val for layer_val in layers if layer_val == layer]
    result_name = f"attn_kv_L{layer}" if layer is not None else "attn_kv"
    if not layers:
        return CheckResult(name=result_name, passed=False,
                           metrics={"error": f"layer {layer} 不在双方 attn_kv 中"})

    per_layer: dict = {}
    worst = {"max_diff": 0.0, "cos_min": 1.0}
    for layer_idx in layers:
        on_key = on_dict[layer_idx].get("key")
        on_value = on_dict[layer_idx].get("value")
        off_key = off_dict[layer_idx].get("key")
        off_value = off_dict[layer_idx].get("value")
        entry_result: dict = {}
        for kv_type, (on_kv, off_kv) in [("K", (on_key, off_key)), ("V", (on_value, off_value))]:
            if on_kv is None or off_kv is None:
                entry_result[kv_type] = {"error": "缺失"}
                continue
            if on_kv.shape != off_kv.shape:
                entry_result[kv_type] = {
                    "error": f"shape mismatch ON{tuple(on_kv.shape)} vs OFF{tuple(off_kv.shape)}"}
                continue
            on_flat = on_kv.reshape(on_kv.shape[0], -1).float()
            off_flat = off_kv.reshape(off_kv.shape[0], -1).float()
            element_diff = (on_flat - off_flat).abs()
            token_cos = _cosine_sim(on_flat, off_flat, dim=-1)
            max_elem_diff = float(element_diff.max())
            entry_result[kv_type] = {
                "max_diff": max_elem_diff,
                "cos_avg": float(token_cos.mean()), "cos_min": float(token_cos.min()),
                "n_tokens": on_flat.shape[0]}
            worst["max_diff"] = max(worst["max_diff"], max_elem_diff)
            worst["cos_min"] = min(worst["cos_min"], float(token_cos.min()))
        per_layer[layer_idx] = entry_result
    # expanded 应精确等于 full → 阈值极严
    passed = worst["max_diff"] < 1e-5 and worst["cos_min"] > 0.9999
    return CheckResult(name=result_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst["max_diff"],
                                "cos_min": worst["cos_min"], "num_layers": len(layers)})


def _print_attn_kv(check_result: CheckResult):
    print(_SEP_SINGLE + f"\n  [{check_result.name}]  ON expanded_kv vs OFF full_kv (K/V element-wise)")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n"); return
    layers = metrics.get("layers", {})
    print(f"  {'LAYER':>6s}  {'K_MAXDIFF':>12s} {'K_COS':>10s}  "
          f"{'V_MAXDIFF':>12s} {'V_COS':>10s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 12} {'─' * 10}  {'─' * 8}")
    failing_layers = []
    for layer_idx in sorted(layers):
        layer_data = layers[layer_idx]
        key_data, value_data = layer_data.get("K", {}), layer_data.get("V", {})
        if "error" in key_data or "error" in value_data:
            print(f"  {layer_idx:>6d}  K:{key_data.get('error','')}  V:{value_data.get('error','')}")
            failing_layers.append(layer_idx); continue
        key_max_diff, key_cos = key_data["max_diff"], key_data["cos_avg"]
        value_max_diff, value_cos = value_data["max_diff"], value_data["cos_avg"]
        passes_threshold = key_max_diff < 1e-5 and value_max_diff < 1e-5
        if not passes_threshold:
            failing_layers.append(layer_idx)
        print(f"  {layer_idx:>6d}  {key_max_diff:>12.3e} {key_cos:>10.6f}  "
              f"{value_max_diff:>12.3e} {value_cos:>10.6f}  {'OK' if passes_threshold else 'DIFF':>8s}")
    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if check_result.passed else _CROSS} "
          f"{'PASS' if check_result.passed else 'FAIL（KV mismatch → build_kv prefix reuse）'}")
    if failing_layers:
        print(f"  ⚠ First KV mismatched layer: {failing_layers[0]}")
    print()


def cmp_build_kv_input_v(dir_on: str, dir_off: str,
                         layer: int | None = None) -> CheckResult | None:
    """对比 ON build_kv_input_v vs OFF build_kv_input_v（suffix 对齐）。

    两边都存 get_qkv 后、build_kv/RoPE 前的 raw V（``{layer: tensor}``）。同源对比，
    应逐元素相同——若不同则问题在 QKV 投影阶段（hidden_states / QKV 权重）。
    ON_T vs OFF_T 还能看出 ON 有没有把 hidden_states 裁成 suffix-only。
    """
    filepath_on = os.path.join(dir_on, "build_kv_input_v.pt")
    filepath_off = os.path.join(dir_off, "build_kv_input_v.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None
    layers = sorted(set(on_dict.keys()) & set(off_dict.keys()))
    if layer is not None:
        layers = [layer_val for layer_val in layers if layer_val == layer]
    result_name = f"build_kv_input_v_L{layer}" if layer is not None else "build_kv_input_v"
    if not layers:
        return CheckResult(name=result_name, passed=False, metrics={"error": "no layers"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    per_layer: dict = {}
    worst_max_diff = 0.0
    worst_cos = 1.0
    for layer_idx in layers:
        on_values = on_dict[layer_idx]
        off_values = off_dict[layer_idx]
        if on_values is None or off_values is None:
            per_layer[layer_idx] = {"error": "missing"}; continue
        on_flat = on_values.reshape(on_values.shape[0], -1).float()
        off_flat = off_values.reshape(off_values.shape[0], -1).float()
        on_token_count, off_token_count = int(on_values.shape[0]), int(off_values.shape[0])
        if align_mask is not None and on_flat.shape[0] != off_flat.shape[0]:
            try:
                on_flat, off_flat = _align_packed(on_flat, off_flat, align_mask)
            except ValueError as exc:
                per_layer[layer_idx] = {"error": str(exc), "on_T": on_token_count, "off_T": off_token_count}
                continue
        diff = (on_flat - off_flat).abs()
        cosine_values = _cosine_sim(on_flat, off_flat, dim=-1)
        layer_max_diff = float(diff.max())
        per_layer[layer_idx] = {"max_diff": layer_max_diff, "cos_avg": float(cosine_values.mean()),
                          "cos_min": float(cosine_values.min()), "n_tokens": on_flat.shape[0],
                          "on_T": on_token_count, "off_T": off_token_count}
        worst_max_diff = max(worst_max_diff, layer_max_diff)
        worst_cos = min(worst_cos, float(cosine_values.min()))
    passed = worst_max_diff < 1e-5
    return CheckResult(name=result_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_max_diff, "cos_min": worst_cos})


def _print_build_kv_input_v(check_result: CheckResult):
    print(_SEP_SINGLE + f"\n  [{check_result.name}]  ON build_kv_input_v vs OFF build_kv_input_v (suffix aligned)")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n"); return
    layers = metrics.get("layers", {})
    print(f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS':>10s}  "
          f"{'ON_T':>8s} {'OFF_T':>8s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 8} {'─' * 8}  {'─' * 8}")
    for layer_idx in sorted(layers):
        layer_data = layers[layer_idx]
        if "max_diff" not in layer_data:
            print(f"  {layer_idx:>6d}  {layer_data.get('error', '')}  ON_T={layer_data.get('on_T')} OFF_T={layer_data.get('off_T')}")
            continue
        layer_max_diff, layer_cos = layer_data["max_diff"], layer_data["cos_avg"]
        passes_threshold = layer_max_diff < 1e-5
        cropped_label = " (cropped)" if layer_data.get("on_T") != layer_data.get("off_T") else ""
        print(f"  {layer_idx:>6d}  {layer_max_diff:>12.3e} {layer_cos:>10.6f}  "
              f"{layer_data.get('on_T', '—'):>8} {layer_data.get('off_T', '—'):>8}  "
              f"{'OK' if passes_threshold else 'DIFF':>8s}{cropped_label}")
    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if check_result.passed else _CROSS} "
          f"{'PASS' if check_result.passed else 'FAIL（V diverged before build_kv → root cause in get_qkv/hidden_states）'}")
    print()


def cmp_hidden_states(dir_on: str, dir_off: str,
                      layer: int | None = None) -> CheckResult | None:
    """Compare ON vs OFF hidden_states (suffix aligned, attention entrance).

    This is the INPUT to QKV projection. If hidden_states match but V differs → GEMM precision;
    if hidden_states already differ → root cause is upstream (embedding / input_layernorm).
    """
    on_filepath = os.path.join(dir_on, "hidden_states.pt")
    off_filepath = os.path.join(dir_off, "hidden_states.pt")
    if not os.path.exists(on_filepath) or not os.path.exists(off_filepath):
        return None
    on_dict = torch.load(on_filepath, weights_only=True)
    off_dict = torch.load(off_filepath, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None
    layers = sorted(set(on_dict.keys()) & set(off_dict.keys()))
    if layer is not None:
        layers = [layer_val for layer_val in layers if layer_val == layer]
    result_name = f"hidden_states_L{layer}" if layer is not None else "hidden_states"
    if not layers:
        return CheckResult(name=result_name, passed=False, metrics={"error": "no layers"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    per_layer: dict = {}
    worst_max_diff = 0.0
    worst_cos = 1.0
    for layer_idx in layers:
        on_hidden = on_dict[layer_idx]
        off_hidden = off_dict[layer_idx]
        if on_hidden is None or off_hidden is None:
            per_layer[layer_idx] = {"error": "missing"}; continue
        on_flat = on_hidden.reshape(on_hidden.shape[0], -1).float()
        off_flat = off_hidden.reshape(off_hidden.shape[0], -1).float()
        on_token_count, off_token_count = int(on_hidden.shape[0]), int(off_hidden.shape[0])
        if align_mask is not None and on_flat.shape[0] != off_flat.shape[0]:
            try:
                on_flat, off_flat = _align_packed(on_flat, off_flat, align_mask)
            except ValueError as exc:
                per_layer[layer_idx] = {"error": str(exc), "on_T": on_token_count, "off_T": off_token_count}
                continue
        diff = (on_flat - off_flat).abs()
        cosine_values = _cosine_sim(on_flat, off_flat, dim=-1)
        layer_max_diff = float(diff.max())
        per_layer[layer_idx] = {"max_diff": layer_max_diff, "cos_avg": float(cosine_values.mean()),
                          "cos_min": float(cosine_values.min()), "n_tokens": on_flat.shape[0],
                          "on_T": on_token_count, "off_T": off_token_count}
        worst_max_diff = max(worst_max_diff, layer_max_diff)
        worst_cos = min(worst_cos, float(cosine_values.min()))
    passed = worst_max_diff < 1e-5
    return CheckResult(name=result_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_max_diff, "cos_min": worst_cos})


def _print_hidden_states(check_result: CheckResult):
    print(_SEP_SINGLE + f"\n  [{check_result.name}]  ON vs OFF hidden_states (suffix aligned, attention entrance)")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n"); return
    layers = metrics.get("layers", {})
    print(f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS':>10s}  "
          f"{'ON_T':>8s} {'OFF_T':>8s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 8} {'─' * 8}  {'─' * 8}")
    for layer_idx in sorted(layers):
        layer_data = layers[layer_idx]
        if "max_diff" not in layer_data:
            print(f"  {layer_idx:>6d}  {layer_data.get('error', '')}  ON_T={layer_data.get('on_T')} OFF_T={layer_data.get('off_T')}")
            continue
        layer_max_diff, layer_cos = layer_data["max_diff"], layer_data["cos_avg"]
        passes_threshold = layer_max_diff < 1e-5
        print(f"  {layer_idx:>6d}  {layer_max_diff:>12.3e} {layer_cos:>10.6f}  "
              f"{layer_data.get('on_T', '—'):>8} {layer_data.get('off_T', '—'):>8}  "
              f"{'OK' if passes_threshold else 'DIFF':>8s}")
    print(f"\n  max_diff={metrics.get('max_diff')}  cos_min={metrics.get('cos_min')}  "
          f"{_CHECK if check_result.passed else _CROSS} "
          f"{'PASS（hidden_states match → V difference in GEMM）' if check_result.passed else 'FAIL（hidden_states mismatch → root cause upstream）'}")
    print()


# ════════════════════════════════════════════════════════════════
#  2D mask loading
# ════════════════════════════════════════════════════════════════

def _load_mask_2d(dir_path: str, mask_kind: str, tag: str) -> torch.Tensor | None:
    """加载 2D mask：``label_mask_{tag}.pt`` / ``attention_mask_{tag}.pt``。"""
    if mask_kind == "none":
        return None
    fname = f"{mask_kind}_mask_{tag}.pt"  # label_mask_{tag} / attention_mask_{tag}
    filepath = os.path.join(dir_path, fname)
    if not os.path.exists(filepath):
        return None
    return torch.load(filepath, weights_only=True).to(torch.bool)


def _resolve_mask(dir_off: str, mask_kind: str, tag: str,
                  ref_shape: tuple[int, ...]) -> torch.Tensor | None:
    """加载 mask（取 OFF 侧 = ground truth 坐标系）并校验 shape。

    若 mask 与 logprobs shape 不一致，打印警告并返回 None（回退到全位置对比）。
    """
    if mask_kind == "none":
        return None
    mask = _load_mask_2d(dir_off, mask_kind, tag)
    if mask is None:
        print(f"  {_CROSS} {mask_kind}_mask_{tag}.pt not found in OFF, "
              f"comparing all positions\n")
        return None
    if tuple(mask.shape) != ref_shape:
        print(f"  {_CROSS} {mask_kind}_mask shape {tuple(mask.shape)} != "
              f"logprobs {ref_shape}, comparing all positions\n")
        return None
    return mask


# ════════════════════════════════════════════════════════════════
#  2D comparison（logprobs / entropy）
# ════════════════════════════════════════════════════════════════

def cmp_2d(dir_on: str, dir_off: str, filename: str, name: str,
           mask: torch.Tensor | None, atol: float
           ) -> tuple[CheckResult, torch.Tensor | None, torch.Tensor | None]:
    """Compare ON/OFF 2D tensors (logprobs / entropy).

    Returns ``(result, on_tensor, off_tensor)`` — the latter two for top-K print reuse.
    """
    on_tensor = _load_tensor(dir_on, filename)
    off_tensor = _load_tensor(dir_off, filename)
    if on_tensor is None or off_tensor is None:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "file missing"}), on_tensor, off_tensor)
    if on_tensor.shape != off_tensor.shape:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "shape mismatch",
                                     "on_shape": tuple(on_tensor.shape),
                                     "off_shape": tuple(off_tensor.shape)}), on_tensor, off_tensor)
    mask_tensor = mask.to(on_tensor.device) if mask is not None else None
    if mask_tensor is not None:
        mask_tensor = mask_tensor & ~torch.isnan(on_tensor) & ~torch.isnan(off_tensor)
    error_metrics = _error_abs_rel(on_tensor, off_tensor, mask_tensor)
    active_count = int(mask_tensor.sum()) if mask_tensor is not None else on_tensor.numel()
    return (CheckResult(name=name,
                        passed=active_count == 0 or error_metrics["abs_max"] <= atol,
                        metrics={"shape": tuple(on_tensor.shape),
                                 "active": active_count,
                                 "abs_max": error_metrics["abs_max"],
                                 "abs_mean": error_metrics["abs_mean"],
                                 "rel_max": error_metrics["rel_max"],
                                 "rel_mean": error_metrics["rel_mean"],
                                 "pearson_r": _pearson_r(on_tensor, off_tensor, mask_tensor),
                                 "atol": atol}), on_tensor, off_tensor)


# ════════════════════════════════════════════════════════════════
#  Shape diagnostics
# ════════════════════════════════════════════════════════════════

def _shape_of(dir_path: str, filename: str) -> str:
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        return "(missing)"
    try:
        obj = torch.load(filepath, weights_only=True)
        if isinstance(obj, dict):
            # per-layer dict（attn_outputs / rope_freqs_*）：显示层数 + 首层 shape
            sample = next(iter(obj.values())) if obj else None
            # rope_postqk.pt：每层值是 {"query","key"[,"positions"]} dict，取 query 的 shape 代表
            if isinstance(sample, dict):
                query_tensor = sample.get("query")
                sample_shape = f",Q{tuple(query_tensor.shape)}" if query_tensor is not None else ""
            elif sample is not None:
                sample_shape = f",{tuple(sample.shape)}"
            else:
                sample_shape = ""
            return f"(dict,{len(obj)}L{sample_shape})"
        return str(tuple(obj.shape))
    except Exception:
        return "(error)"




def _print_shapes(dir_on: str, dir_off: str, tag: str):
    """Print ON/OFF .pt file shapes — first-pass shape mismatch diagnosis."""
    print(_SEP_SINGLE + "\n  [shapes]  ON vs OFF dump shapes")
    print(_SEP_SINGLE)
    dump_files = [
        f"logprobs_{tag}.pt",
        f"entropy_{tag}.pt",
        f"label_mask_{tag}.pt",
        f"attention_mask_{tag}.pt",
        "logits.pt",
        "attn_outputs.pt",
        "rope_postqk.pt",
        "rope_preqk.pt",
        "rope_freqs.pt",
        "expanded_kv.pt",
        "full_kv.pt",
        "build_kv_input_v.pt",
        "hidden_states.pt",
        "prefix_lens.pt",
        "cu_seqlens_q.pt",
    ]
    print(f"  {'FILE':<28s} {'ON':<16s} {'OFF':<16s} {'STATUS'}")
    print(f"  {'─' * 28} {'─' * 16} {'─' * 16} {'─' * 10}")
    for filename in dump_files:
        on_shape, off_shape = _shape_of(dir_on, filename), _shape_of(dir_off, filename)
        if on_shape == "(missing)" or off_shape == "(missing)":
            status = "—"
        elif on_shape == off_shape:
            status = "OK"
        else:
            status = f"{_CROSS} DIFF"
        print(f"  {filename:<28s} {on_shape:<16s} {off_shape:<16s} {status}")
    print()


# ════════════════════════════════════════════════════════════════
#  Output
# ════════════════════════════════════════════════════════════════

def _print_header(dir_on, dir_off, dir_off2, tag, mask_kind, layer):
    print(_SEP_DOUBLE)
    print("  verl080 Prefix-Sharing Diag Report")
    print(f"  ON :  {dir_on}\n  OFF:  {dir_off}")
    if dir_off2:
        print(f"  OFF2: {dir_off2}")
    detail = f"TAG: {tag}    MASK: {mask_kind}"
    if layer is not None:
        detail += f"    LAYER: {layer}"
    print(f"  {detail}")
    print(_SEP_DOUBLE + "\n")


def _print_rope_freqs(check_result: CheckResult):
    print(_SEP_SINGLE + f"\n  [rope_freqs]  {_CHECK if check_result.passed else _CROSS} "
          f"{'PASS' if check_result.passed else 'FAIL'}")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {metrics['error']}")
    else:
        print(f"  layers: {metrics.get('num_layers', '—')}  "
              f"max_diff: {metrics.get('max_diff', '—')}")
        mismatch_total = metrics.get("total_mismatches", 0)
        if mismatch_total > 0:
            print(f"  mismatched tokens: {mismatch_total}"
                  f"{' (showing first 20)' if mismatch_total > 20 else ''}")
            print(f"  {'Layer':>6s} {'Token':>6s} {'Dim':>6s}  "
                  f"{'ON_val':>14s}  {'OFF_val':>14s}  {'Abs_err':>12s}")
            for mismatch in metrics.get("mismatches", []):
                print(f"  {mismatch['layer']:>6d} {mismatch['token_idx']:>6d} {mismatch['dim']:>6d}  "
                      f"{mismatch['on_val']:>14.6e}  {mismatch['off_val']:>14.6e}  "
                      f"{mismatch['diff']:>12.6e}")
    if not check_result.passed:
        print(f"  {_CROSS} CRITICAL — STOP.")
    print()


def _print_per_layer(check_result: CheckResult):
    print(_SEP_SINGLE + "\n  [attn_output]  Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = check_result.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'COS_AVG':>14s}  {'COS_MIN':>14s}  "
              f"{'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─' * 6}  {'─' * 14}  {'─' * 14}  {'─' * 8}  {'─' * 8}")
        failing_layers = []
        for layer_idx in sorted(layers.keys()):
            layer_metrics = layers[layer_idx]
            if "error" in layer_metrics:
                print(f"  {layer_idx:>6d}  {layer_metrics['error']}")
                failing_layers.append(layer_idx)
                continue
            passes_threshold = layer_metrics["cos_avg"] > _COS_AVG_PASS and layer_metrics["cos_min"] > _COS_MIN_PASS
            print(f"  {layer_idx:>6d}  {layer_metrics['cos_avg']:>14.6e}  {layer_metrics['cos_min']:>14.6e}  "
                  f"{layer_metrics['n_tokens']:>8d}  {'PASS' if passes_threshold else 'WARN':>8s}")
            if not passes_threshold:
                failing_layers.append(layer_idx)
        if failing_layers:
            print(f"\n  ⚠ First deviating layer: {failing_layers[0]}")
    elif "cos_avg" in check_result.metrics:
        layer_metrics = check_result.metrics
        passes_threshold = layer_metrics["cos_avg"] > _COS_AVG_PASS and layer_metrics["cos_min"] > _COS_MIN_PASS
        print(f"  L{layer_metrics['layer']}  cos_avg={layer_metrics['cos_avg']:.6e}  "
              f"cos_min={layer_metrics['cos_min']:.6e}  {'PASS' if passes_threshold else 'WARN'}")
    elif "error" in check_result.metrics:
        print(f"  {_CROSS} {check_result.metrics['error']}")
    print()


def _print_rope_postqk_per_layer(check_result: CheckResult):
    section_label = "rope_preqk" if "preqk" in check_result.name else "rope_postqk"
    stage_label = "Pre-RoPE" if "preqk" in check_result.name else "Post-RoPE"
    print(_SEP_SINGLE + f"\n  [{section_label}]  {stage_label} Q/K Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = check_result.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'Q_MAXDIFF':>12s}  {'Q_COS_AVG':>12s}  "
              f"{'K_MAXDIFF':>12s}  {'K_COS_AVG':>12s}  "
              f"{'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}  "
              f"{'─' * 8}  {'─' * 8}")
        failing_layers = []
        for layer_idx in sorted(layers.keys()):
            layer_metrics = layers[layer_idx]
            if "error" in layer_metrics:
                print(f"  {layer_idx:>6d}  {layer_metrics['error']}")
                failing_layers.append(layer_idx)
                continue
            passes_threshold = (layer_metrics["Q_cos_avg"] > _COS_AVG_PASS and layer_metrics["Q_cos_min"] > _COS_MIN_PASS
                  and layer_metrics["K_cos_avg"] > _COS_AVG_PASS and layer_metrics["K_cos_min"] > _COS_MIN_PASS)
            print(f"  {layer_idx:>6d}  {layer_metrics.get('Q_max_diff', 0.0):>12.3e}  "
                  f"{layer_metrics['Q_cos_avg']:>12.6e}  "
                  f"{layer_metrics.get('K_max_diff', 0.0):>12.3e}  {layer_metrics['K_cos_avg']:>12.6e}  "
                  f"{layer_metrics['n_tokens']:>8d}  {'PASS' if passes_threshold else 'WARN':>8s}")
            if not passes_threshold:
                failing_layers.append(layer_idx)
        if failing_layers:
            print(f"\n  ⚠ First deviating layer: {failing_layers[0]}")
    elif "Q_cos_avg" in check_result.metrics:
        layer_metrics = check_result.metrics
        passes_threshold = (layer_metrics["Q_cos_avg"] > _COS_AVG_PASS and layer_metrics["Q_cos_min"] > _COS_MIN_PASS
              and layer_metrics["K_cos_avg"] > _COS_AVG_PASS and layer_metrics["K_cos_min"] > _COS_MIN_PASS)
        print(f"  L{layer_metrics['layer']}  Q_maxdiff={layer_metrics.get('Q_max_diff', 0.0):.3e}  "
              f"Q_cos_avg={layer_metrics['Q_cos_avg']:.6e}  "
              f"K_maxdiff={layer_metrics.get('K_max_diff', 0.0):.3e}  "
              f"K_cos_avg={layer_metrics['K_cos_avg']:.6e}  {'PASS' if passes_threshold else 'WARN'}")
    elif "error" in check_result.metrics:
        print(f"  {_CROSS} {check_result.metrics['error']}")
    print()


def _print_packed_token(check_result: CheckResult):
    print(_SEP_SINGLE + f"\n  [packed_token]  {check_result.name}")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}\n")
        return
    for metric_name in ["mean_abs", "max_abs", "rel_max", "rel_mean", "cos", "pearson"]:
        metric_value = metrics.get(metric_name)
        if metric_value is not None:
            print(f"  {metric_name:>12s}  {metric_value:>14.6e}")
    print()


def _print_logits_packed(check_result: CheckResult):
    print(_SEP_SINGLE + "\n  [logits]  packed alignment (suffix aligned)")
    print(_SEP_SINGLE)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  {_CROSS} {metrics['error']}")
    else:
        print(f"  n_tokens={metrics.get('n_tokens', '—')}  "
              f"cos_avg={metrics.get('cos_avg', 0):.6e}  cos_min={metrics.get('cos_min', 0):.6e}  "
              f"{_CHECK if check_result.passed else _CROSS} {'PASS' if check_result.passed else 'FAIL'}")
    print()


def _print_2d_result(check_result: CheckResult):
    print(_SEP_THIN)
    metrics = check_result.metrics
    if "error" in metrics:
        print(f"  [{check_result.name}]  {_CROSS} {metrics['error']}")
        if metrics.get("on_shape") or metrics.get("off_shape"):
            print(f"    ON: {metrics.get('on_shape')}   OFF: {metrics.get('off_shape')}")
        print()
        return
    print(f"  [{check_result.name}]  shape={metrics.get('shape')}  active={metrics.get('active')}  "
          f"abs_max={metrics.get('abs_max', 0):.6e}  rel_max={metrics.get('rel_max', 0):.6e}  "
          f"pearson={metrics.get('pearson_r', float('nan')):.8f}")
    print(f"  abs_mean={metrics.get('abs_mean', 0):.6e}  rel_mean={metrics.get('rel_mean', 0):.6e}  "
          f"{_CHECK if check_result.passed else _CROSS} {'PASS' if check_result.passed else 'FAIL'}")
    print()


def _print_topk_vec(on_vec: torch.Tensor, off_vec: torch.Tensor,
                    topk: int, sort_by: str, label: str, show_rel: bool = True):
    """1D 向量 top-K（packed_token per-dim）。

    show_rel=False 时省略 REL_ERR 列（用于 logits 表，只看 val/abs）。
    """
    abs_err = (on_vec - off_vec).abs()
    rel_err = abs_err / torch.maximum(on_vec.abs(), off_vec.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    else:  # "val" —— 带符号的实际值，不是绝对值
        # 对 logits：绝对值大但符号为负的 logit，softmax 后概率极低、不会被选中。
        # 按 abs 排会把这种"必不选"的 token 顶到表头，掩盖真正的高 logit 候选。
        # 改用 max(on, off) 带符号值，让真正的高 logit（候选 token）排前面。
        sort_key = torch.maximum(on_vec, off_vec)
    _, idx = sort_key.topk(min(topk, sort_key.numel()))
    idx = idx.to(torch.long)
    print(f"\n  [{label}]  top-{topk} dims (sort by {sort_by})")
    if show_rel:
        print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}  {'REL_ERR':>12s}")
        for i in idx.tolist():
            print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
                  f"  {float(abs_err[i]):>12.6e}  {float(rel_err[i]):>12.6e}")
    else:
        print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}")
        for i in idx.tolist():
            print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
                  f"  {float(abs_err[i]):>12.6e}")
    return idx.tolist()


def _print_vec_at_dims(on_vec: torch.Tensor, off_vec: torch.Tensor,
                       dims, label: str, show_rel: bool = True):
    """在指定 dims 上打印 ON/OFF/ABS_ERR（不排序），跨 stage 对齐同一批 dim。

    供 rope 流水线 top-K 对齐：dims 取自 rope_postqk 的 sort-err top-K，
    在 rope_preqk / rope_freqs 上显示同样的 dim，逐 dim 追溯误差来源。
    """
    abs_err = (on_vec - off_vec).abs()
    rel_err = abs_err / torch.maximum(on_vec.abs(), off_vec.abs()).clamp(min=1e-8)
    print(f"\n  [{label}]  at {len(dims)} dims")
    if show_rel:
        print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}  {'REL_ERR':>12s}")
        for i in dims:
            print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
                  f"  {float(abs_err[i]):>12.6e}  {float(rel_err[i]):>12.6e}")
    else:
        print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}")
        for i in dims:
            print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
                  f"  {float(abs_err[i]):>12.6e}")


def _print_topk_2d(on_t: torch.Tensor, off_t: torch.Tensor,
                   mask: torch.Tensor | None, topk: int, sort_by: str,
                   label: str):
    """2D [B, L_max] top-K 位置（logp/entropy）。"""
    abs_err = (on_t - off_t).abs()
    rel_err = abs_err / torch.maximum(on_t.abs(), off_t.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    else:  # "val"
        sort_key = torch.maximum(on_t.abs(), off_t.abs())
    if mask is not None:
        sort_key = sort_key.clone()
        sort_key[~mask.to(sort_key.device)] = float("-inf")
    flat, idx = sort_key.flatten().topk(min(topk, sort_key.numel()))
    rows = idx // sort_key.shape[1]
    cols = idx % sort_key.shape[1]
    print(f"\n  [{label}]  top-{topk} positions (sort by {sort_by})")
    print(f"  {'Seq_idx':>7s} {'POS':>6s}  {'ON':>14s}  {'OFF':>14s}  "
          f"{'ABS_ERR':>12s}  {'REL_ERR':>12s}")
    for k in range(len(flat)):
        r, c = int(rows[k]), int(cols[k])
        print(f"  {r:>7d} {c:>6d}  {float(on_t[r, c]):>14.6e}  "
              f"{float(off_t[r, c]):>14.6e}  {float(abs_err[r, c]):>12.6e}  "
              f"{float(rel_err[r, c]):>12.6e}")


def _print_summary(results: list[CheckResult]):
    print(_SEP_DOUBLE + "\n  SUMMARY")
    print(_SEP_DOUBLE)
    hdr = (f"  {'NAME':<20s} {'SHAPE':<16s} {'ABS_MAX':>12s}  {'REL_MAX':>10s}  "
           f"{'PEARSON_R':>10s}  {'STATUS':>8s}")
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for result in results:
        result_metrics = result.metrics
        shape = str(result_metrics.get("shape", result_metrics.get("error", "—")))
        abs_max_str = f"{result_metrics['abs_max']:.6e}" if "abs_max" in result_metrics else "—"
        rel_max_str = f"{result_metrics['rel_max']:.6e}" if "rel_max" in result_metrics else "—"
        pearson_str = f"{result_metrics['pearson_r']:.6f}" if result_metrics.get("pearson_r") is not None else "—"
        status_str = f"  {_CHECK} PASS" if result.passed else f"  {_CROSS} FAIL"
        print(f"  {result.name:<20s} {shape:<16s} {abs_max_str:>12s}  {rel_max_str:>10s}  {pearson_str:>10s}  {status_str}")
    print(_SEP_DOUBLE + "\n")


def _dump_json(results: list[CheckResult], path: str,
               dir_on: str, dir_off: str, tag: str, dir_off2: str | None):
    record: dict = {"dir_on": dir_on, "dir_off": dir_off, "tag": tag}
    if dir_off2:
        record["dir_off2"] = dir_off2
    record["results"] = [{**result.metrics, "name": result.name, "passed": result.passed}
                         for result in results]
    record["all_passed"] = all(result.passed for result in results)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {path}\n")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    arg_parser = argparse.ArgumentParser(
        description="verl080 precision comparison — ON vs OFF (packed + 2D, self-contained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    arg_parser.add_argument("--dir-on", required=True, help="ON dump directory")
    arg_parser.add_argument("--dir-off", required=True, help="OFF dump directory")
    arg_parser.add_argument("--dir-off2", default=None,
                    help="Second OFF dir for OFF-vs-OFF baseline")
    arg_parser.add_argument("--tag", required=True,
                    help="2D file tag (old / train) — logprobs_{tag}.pt, mask_{tag}.pt")
    arg_parser.add_argument("--mask", choices=["label", "attention", "none"],
                    default="label", help="2D mask type (default: label)")
    arg_parser.add_argument("--layer", type=int, default=None,
                    help="Compare specific attn layer 1-indexed (default: all). "
                         "Also used by packed_token attn (default: last layer).")
    arg_parser.add_argument("--token", type=int, default=0,
                    help="Packed token position for packed_token compare "
                         "(single int index, default: 0)")
    arg_parser.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D (default: 1e-5)")
    arg_parser.add_argument("--topk", type=int, default=0,
                    help="top-K worst positions (0 = disabled)")
    arg_parser.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="top-K sort: abs / rel / val (default: abs)")
    arg_parser.add_argument("--output", "-o", default=None, help="Save report as JSON")
    args = arg_parser.parse_args()

    _print_header(args.dir_on, args.dir_off, args.dir_off2,
                  args.tag, args.mask, args.layer)

    # ── shape diagnostics ──
    _print_shapes(args.dir_on, args.dir_off, args.tag)

    # ── resolve 2D mask ──
    reference_tensor = _load_tensor(args.dir_off, f"logprobs_{args.tag}.pt")
    mask = None
    if reference_tensor is not None:
        mask = _resolve_mask(args.dir_off, args.mask, args.tag, tuple(reference_tensor.shape))
        if mask is not None:
            print(f"  mask: {args.mask}  shape={tuple(mask.shape)}  "
                  f"active={int(mask.sum())}\n")

    all_results: list[CheckResult] = []

    # ── RoPE pipeline per-layer: pre Q/K → rope freqs → post Q/K ──
    check_result = cmp_rope_postqk_layer(args.dir_on, args.dir_off, args.layer, stage="pre")
    if check_result:
        all_results.append(check_result)
        _print_rope_postqk_per_layer(check_result)

    check_result = cmp_rope_freqs(args.dir_on, args.dir_off, layer=args.layer)
    if check_result:
        all_results.append(check_result)
        _print_rope_freqs(check_result)

    check_result = cmp_rope_postqk_layer(args.dir_on, args.dir_off, args.layer, stage="post")
    if check_result:
        all_results.append(check_result)
        _print_rope_postqk_per_layer(check_result)

    # ── packed: attention KV (ON expanded vs OFF full) ──
    check_result = cmp_attn_kv(args.dir_on, args.dir_off, args.layer)
    if check_result:
        all_results.append(check_result)
        _print_attn_kv(check_result)

    # ── packed: build_kv input V — ON vs OFF same-source ──
    check_result = cmp_build_kv_input_v(args.dir_on, args.dir_off, args.layer)
    if check_result:
        all_results.append(check_result)
        _print_build_kv_input_v(check_result)

    # ── packed: hidden_states (attention entrance) ──
    check_result = cmp_hidden_states(args.dir_on, args.dir_off, args.layer)
    if check_result:
        all_results.append(check_result)
        _print_hidden_states(check_result)

    # ── packed: attention_output per-layer cos ──
    check_result = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
    if check_result:
        all_results.append(check_result)
        _print_per_layer(check_result)

    # ── packed: attn_grad per-layer cos ──
    check_result = cmp_attn_grads(args.dir_on, args.dir_off, args.layer)
    if check_result:
        all_results.append(check_result)
        _print_per_layer(check_result)

    # ── packed: packed_token（attn[pos] + logits[pos]，suffix 对齐后） ──
    # pos 由 --token 指定（默认 0，索引对齐后的 suffix-packed 空间）；
    # attn 用 --layer 指定的层（默认最后一层）；logits 永远最后一层。
    pos = args.token
    align_mask = _build_attn_align_mask(args.dir_on, args.dir_off)
    packed_token_results = cmp_packed_token(args.dir_on, args.dir_off, pos, args.layer,
                                  align_mask=align_mask)
    for packed_result in packed_token_results:
        all_results.append(packed_result)
        _print_packed_token(packed_result)
    if args.topk > 0 and packed_token_results:
        attn_layer = args.layer if args.layer is not None else (
            _get_num_layers(args.dir_on) or _get_num_layers(args.dir_off))
        if attn_layer:
            on_tensor = _load_attn_output(args.dir_on, attn_layer)
            off_tensor = _load_attn_output(args.dir_off, attn_layer)
            vecs = _aligned_vec_at_pos(on_tensor, off_tensor, True, pos, align_mask)
            if vecs is not None:
                _print_topk_vec(vecs[0].cpu(), vecs[1].cpu(), args.topk, "val",
                                f"attn_L{attn_layer}_pos{pos}")
        logits_on = _load_logits(args.dir_on)
        logits_off = _load_logits(args.dir_off)
        vecs = _aligned_vec_at_pos(logits_on, logits_off, False, pos, align_mask)
        if vecs is not None:
            _print_topk_vec(vecs[0].cpu(), vecs[1].cpu(), args.topk,
                            "val", f"logits_pos{pos}", show_rel=False)

    # ── packed: logits (suffix aligned) ──
    check_result = cmp_logits_packed(args.dir_on, args.dir_off)
    if check_result:
        all_results.append(check_result)
        _print_logits_packed(check_result)

    # ── RoPE pipeline packed_token: pre Q/K → rope_freqs → post Q/K ──
    rope_packed_token_results: list[CheckResult] = []
    for rope_result in cmp_rope_postqk_token(args.dir_on, args.dir_off, pos, args.layer,
                                align_mask=align_mask, stage="pre"):
        all_results.append(rope_result); rope_packed_token_results.append(rope_result); _print_packed_token(rope_result)
    freqs_token_result = cmp_rope_freqs_token(args.dir_on, args.dir_off, pos, args.layer,
                               align_mask=align_mask)
    if freqs_token_result is not None:
        all_results.append(freqs_token_result); rope_packed_token_results.append(freqs_token_result); _print_packed_token(freqs_token_result)
    for rope_result in cmp_rope_postqk_token(args.dir_on, args.dir_off, pos, args.layer,
                                align_mask=align_mask, stage="post"):
        all_results.append(rope_result); rope_packed_token_results.append(rope_result); _print_packed_token(rope_result)
    # rope packed_token top-K —— dim 跨 stage 对齐：以 rope_postqk 的 sort-err top-K dim 为基准，
    # rope_preqk 显示同样 dim，rope_freqs 显示 dim%D（角度按 head_dim 共享），逐 dim 追溯误差。
    if args.topk > 0 and rope_packed_token_results:
        rope_layer = args.layer if args.layer is not None else (
            _get_num_layers(args.dir_on) or _get_num_layers(args.dir_off))
        if rope_layer:
            pre_q_on, pre_k_on = _load_rope_postqk(args.dir_on, rope_layer, "rope_preqk.pt")
            pre_q_off, pre_k_off = _load_rope_postqk(args.dir_off, rope_layer, "rope_preqk.pt")
            post_q_on, post_k_on = _load_rope_postqk(args.dir_on, rope_layer, "rope_postqk.pt")
            post_q_off, post_k_off = _load_rope_postqk(args.dir_off, rope_layer, "rope_postqk.pt")
            pre_vecs = _rope_postqk_vec_at_pos(pre_q_on, pre_k_on, pre_q_off, pre_k_off, pos, align_mask)
            post_vecs = _rope_postqk_vec_at_pos(post_q_on, post_k_on, post_q_off, post_k_off, pos, align_mask)
            freq_on, freq_off = _load_rope_freqs_vec_at_pos(
                args.dir_on, args.dir_off, rope_layer, pos, align_mask)
            if post_vecs is not None:
                pqo, pqf, pko, pkf = post_vecs
                # Q: postqk sort-err top-K → preqk / freqs 同 dim
                q_dims = _print_topk_vec(pqo.cpu(), pqf.cpu(), args.topk, args.sort_err,
                                         f"rope_postqk_L{rope_layer}_Q_pos{pos}")
                if pre_vecs is not None:
                    _print_vec_at_dims(pre_vecs[0].cpu(), pre_vecs[1].cpu(), q_dims,
                                       f"rope_preqk_L{rope_layer}_Q_pos{pos} (same dims)")
                if freq_on is not None and freq_off is not None:
                    head_dim = freq_on.numel()
                    _print_vec_at_dims(freq_on.cpu(), freq_off.cpu(),
                                       [dim_idx % head_dim for dim_idx in q_dims],
                                       f"rope_freqs_L{rope_layer}_Q_pos{pos} (dim%D)")
                # K: 同样
                if pko is not None and pkf is not None:
                    k_dims = _print_topk_vec(pko.cpu(), pkf.cpu(), args.topk, args.sort_err,
                                             f"rope_postqk_L{rope_layer}_K_pos{pos}")
                    if pre_vecs is not None and pre_vecs[2] is not None and pre_vecs[3] is not None:
                        _print_vec_at_dims(pre_vecs[2].cpu(), pre_vecs[3].cpu(), k_dims,
                                           f"rope_preqk_L{rope_layer}_K_pos{pos} (same dims)")
                    if freq_on is not None and freq_off is not None:
                        _print_vec_at_dims(freq_on.cpu(), freq_off.cpu(),
                                           [d % _D for dim_idx in k_dims],
                                           f"rope_freqs_L{rope_layer}_K_pos{pos} (dim%D)")

    # ── 2D: logprobs + entropy ──
    for metric_prefix, metric_label in [("logprobs", "logp"), ("entropy", "entropy")]:
        filename = f"{metric_prefix}_{args.tag}.pt"
        check_result, on_tensor, off_tensor = cmp_2d(args.dir_on, args.dir_off, filename,
                           f"{metric_label}_{args.tag}", mask, args.atol)
        all_results.append(check_result)
        _print_2d_result(check_result)
        if (args.topk > 0 and on_tensor is not None and off_tensor is not None
                and on_tensor.shape == off_tensor.shape):
            _print_topk_2d(on_tensor.cpu(), off_tensor.cpu(), mask, args.topk,
                           args.sort_err, check_result.name)

    # ── OFF vs OFF baseline ──
    if args.dir_off2:
        print(_SEP_DOUBLE + "\n  [BASELINE]  OFF vs OFF2 Noise Floor\n" + _SEP_DOUBLE)
        for metric_prefix, metric_label in [("logprobs", "logp"), ("entropy", "entropy")]:
            filename = f"{metric_prefix}_{args.tag}.pt"
            check_result, _, _ = cmp_2d(args.dir_off, args.dir_off2, filename,
                             f"bl_{metric_label}_{args.tag}", mask, args.atol)
            all_results.append(check_result)
            _print_2d_result(check_result)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_on, args.dir_off,
                   args.tag, args.dir_off2)


if __name__ == "__main__":
    main()
