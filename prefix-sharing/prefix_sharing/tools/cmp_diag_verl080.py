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
    """Cosine similarity along ``dim``，promote 到 float32 避免 bf16 噪声。

    Near-zero 向量（两范数都 < sqrt(eps)）视为相同，返回 1.0。
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


def _pearson_r(t1: torch.Tensor, t2: torch.Tensor,
               mask: torch.Tensor | None = None) -> float:
    x = (t1[mask] if mask is not None else t1.flatten()).to(torch.float64)
    y = (t2[mask] if mask is not None else t2.flatten()).to(torch.float64)
    if x.numel() < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx = ((x - mx) ** 2).sum().sqrt()
    sy = ((y - my) ** 2).sum().sqrt()
    return float("nan") if sx == 0 or sy == 0 else float(cov / (sx * sy))


def _vec_metrics(a_vec: torch.Tensor, b_vec: torch.Tensor) -> dict:
    err = _error_abs_rel(a_vec, b_vec)
    cos = float(_cosine_sim(a_vec, b_vec, dim=-1))
    pr = _pearson_r(a_vec, b_vec)
    return {"mean_abs": err["abs_mean"], "max_abs": err["abs_max"],
            "rel_max": err["rel_max"], "rel_mean": err["rel_mean"],
            "cos": cos, "pearson": pr}


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
                      cu_fname: str = "cu_seqlens_q.pt") -> dict | None:
    """加载 cu_seqlens + prefix_lens（suffix 对齐所需）。"""
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
    """加载单层 attn_output（attn_outputs.pt = dict {layer: tensor}）。"""
    filepath = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(filepath):
        return None
    attn_dict = torch.load(filepath, weights_only=True)
    return attn_dict.get(layer) if isinstance(attn_dict, dict) else None


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
    for i in range(cu_seqlens.shape[0] - 1):
        pf = int(prefix_lens[i])
        start = int(cu_seqlens[i]) + pf
        end = int(cu_seqlens[i + 1])
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

def _cos_for_layer(a: torch.Tensor, b: torch.Tensor,
                   align_mask: torch.Tensor | None = None) -> dict:
    """单层 attn_output 对齐 + per-token cosine。"""
    if a.dim() == 3:
        a, b = a.squeeze(1), b.squeeze(1)
    if align_mask is not None and a.shape[0] != b.shape[0]:
        a, b = _align_packed(a, b, align_mask)
    cos = _cosine_sim(a, b, dim=-1)
    return {"cos_avg": float(cos.mean()), "cos_min": float(cos.min()),
            "n_tokens": a.shape[0]}


def _build_attn_align_mask(dir_on: str, dir_off: str) -> torch.Tensor | None:
    """从 OFF cu_seqlens + ON prefix_lens 构建 suffix 对齐 mask（None=无法构建）。"""
    meta_on = _load_packed_meta(dir_on)
    meta_off = _load_packed_meta(dir_off)
    if meta_on is None or meta_off is None:
        return None
    cu_off = meta_off["cu_seqlens"]
    T = int(cu_off[-1]) if cu_off.numel() > 0 else 0
    if T == 0:
        return None
    return _build_alignment_mask(cu_off, meta_on["prefix_lens"], T)


def cmp_attn_layer(dir_on: str, dir_off: str,
                   layer: int | None) -> CheckResult | None:
    """attention_output per-layer cos（suffix 对齐）。

    单层模式（layer 给定）：返回该层 cos。全层模式：返回所有层 cos 汇总。
    """
    align_mask = _build_attn_align_mask(dir_on, dir_off)

    if layer is not None:
        a = _load_attn_output(dir_on, layer)
        b = _load_attn_output(dir_off, layer)
        if a is None or b is None:
            return None
        need = align_mask is not None and a.shape[0] != b.shape[0]
        try:
            d = _cos_for_layer(a, b, align_mask if need else None)
        except ValueError as e:
            return CheckResult(name=f"attn_L{layer}", passed=False,
                               metrics={"error": str(e)})
        d["layer"] = layer
        return CheckResult(name=f"attn_L{layer}",
                           passed=d["cos_avg"] > _COS_AVG_PASS
                           and d["cos_min"] > _COS_MIN_PASS, metrics=d)

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
        a, b = attn_dict_on[layer_idx], attn_dict_off[layer_idx]
        need = align_mask is not None and a.shape[0] != b.shape[0]
        try:
            results[layer_idx] = _cos_for_layer(a, b, align_mask if need else None)
        except ValueError as e:
            results[layer_idx] = {"error": str(e)}
    return CheckResult(name="attn_per_layer", passed=True,
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
        a = _load_attn_output(dir_on, attn_layer)
        b = _load_attn_output(dir_off, attn_layer)
        vecs = _aligned_vec_at_pos(a, b, True, pos, align_mask)
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


def _load_rope_postqk(dir_path: str, layer: int, fname: str = "rope_postqk.pt"
                   ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load Q/K for a single layer from ``fname`` (rope_postqk.pt=post, rope_preqk.pt=pre).

    Returns ``(query, key)`` or ``(None, None)``.
    """
    filepath = os.path.join(dir_path, fname)
    if not os.path.exists(filepath):
        return None, None
    d = torch.load(filepath, weights_only=True)
    if not isinstance(d, dict):
        return None, None
    entry = d.get(layer)
    if entry is None:
        return None, None
    return entry.get("query"), entry.get("key")


def _rope_postqk_cos_for_layer(qa: torch.Tensor, ka: torch.Tensor,
                             qb: torch.Tensor, kb: torch.Tensor,
                             align_mask: torch.Tensor | None = None) -> dict:
    """单层 Q/K suffix 对齐 + per-token cosine（Q 和 K 分别算）。"""
    # Q/K shape: [T, H, D] → 压平 head*dim 维度算 cosine
    qa_flat = qa.reshape(qa.shape[0], -1)
    qb_flat = qb.reshape(qb.shape[0], -1)
    ka_flat = ka.reshape(ka.shape[0], -1)
    kb_flat = kb.reshape(kb.shape[0], -1)

    if align_mask is not None and qa.shape[0] != qb.shape[0]:
        qa_flat, qb_flat = _align_packed(qa_flat, qb_flat, align_mask)
        ka_flat, kb_flat = _align_packed(ka_flat, kb_flat, align_mask)

    q_cos = _cosine_sim(qa_flat, qb_flat, dim=-1)
    k_cos = _cosine_sim(ka_flat, kb_flat, dim=-1)
    return {
        "n_tokens": qa_flat.shape[0],
        "Q_cos_avg": float(q_cos.mean()), "Q_cos_min": float(q_cos.min()),
        "K_cos_avg": float(k_cos.mean()), "K_cos_min": float(k_cos.min()),
        "Q_max_diff": float((qa_flat - qb_flat).abs().max()),
        "K_max_diff": float((ka_flat - kb_flat).abs().max()),
    }


def _cmp_rope_stage_layer(dir_on: str, dir_off: str, layer: int | None,
                           fname: str, label: str) -> CheckResult | None:
    """单 stage（fname/label）的 Q/K per-layer cosine（suffix 对齐）。"""
    align_mask = _build_attn_align_mask(dir_on, dir_off)

    if layer is not None:
        q_on, k_on = _load_rope_postqk(dir_on, layer, fname)
        q_off, k_off = _load_rope_postqk(dir_off, layer, fname)
        if q_on is None or q_off is None:
            return None
        need = align_mask is not None and q_on.shape[0] != q_off.shape[0]
        try:
            d = _rope_postqk_cos_for_layer(q_on, k_on, q_off, k_off,
                                         align_mask if need else None)
        except ValueError as e:
            return CheckResult(name=f"{label}_L{layer}", passed=False,
                               metrics={"error": str(e)})
        d["layer"] = layer
        ok = (d["Q_cos_avg"] > _COS_AVG_PASS and d["Q_cos_min"] > _COS_MIN_PASS
              and d["K_cos_avg"] > _COS_AVG_PASS and d["K_cos_min"] > _COS_MIN_PASS)
        return CheckResult(name=f"{label}_L{layer}", passed=ok, metrics=d)

    # All layers
    filepath_on = os.path.join(dir_on, fname)
    filepath_off = os.path.join(dir_off, fname)
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    attn_dict_on = torch.load(filepath_on, weights_only=True)
    attn_dict_off = torch.load(filepath_off, weights_only=True)
    if not isinstance(attn_dict_on, dict) or not isinstance(attn_dict_off, dict):
        return None

    results = {}
    for layer_idx in sorted(set(attn_dict_on.keys()) & set(attn_dict_off.keys())):
        ea, eb = attn_dict_on[layer_idx], attn_dict_off[layer_idx]
        qa, ka = ea.get("query"), ea.get("key")
        qb, kb = eb.get("query"), eb.get("key")
        if qa is None or qb is None:
            continue
        need = align_mask is not None and qa.shape[0] != qb.shape[0]
        try:
            results[layer_idx] = _rope_postqk_cos_for_layer(qa, ka, qb, kb,
                                                    align_mask if need else None)
        except ValueError as e:
            results[layer_idx] = {"error": str(e)}
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


def _rope_postqk_vec_at_pos(q_on: torch.Tensor | None, k_on: torch.Tensor | None,
                          q_off: torch.Tensor | None, k_off: torch.Tensor | None,
                          pos: int,
                          align_mask: torch.Tensor | None
                          ) -> tuple[torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor] | None:
    """Q/K suffix 对齐后取 [pos]，返回 (q_on, q_off, k_on, k_off) 四向量。

    每个向量压平 [H*D]，可直接做 vec_metrics 对比。
    """
    if q_on is None or q_off is None:
        return None
    q_on_f = q_on.reshape(q_on.shape[0], -1)
    q_off_f = q_off.reshape(q_off.shape[0], -1)
    k_on_f = k_on.reshape(k_on.shape[0], -1) if k_on is not None else None
    k_off_f = k_off.reshape(k_off.shape[0], -1) if k_off is not None else None

    if align_mask is not None and q_on.shape[0] != q_off.shape[0]:
        try:
            q_on_f, q_off_f = _align_packed(q_on_f, q_off_f, align_mask)
            if k_on_f is not None:
                k_on_f, k_off_f = _align_packed(k_on_f, k_off_f, align_mask)
        except ValueError:
            return None
    n = min(q_on_f.shape[0], q_off_f.shape[0])
    if pos < 0 or pos >= n:
        return None
    qo = q_on_f[pos].contiguous()
    qf = q_off_f[pos].contiguous()
    ko = k_on_f[pos].contiguous() if k_on_f is not None else None
    kf = k_off_f[pos].contiguous() if k_off_f is not None else None
    return qo, qf, ko, kf


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
                           align_mask: torch.Tensor | None, fname: str,
                           label: str) -> list[CheckResult]:
    """单 stage（fname/label）的 Q/K packed[pos]（suffix 对齐后）。"""
    if align_mask is None:
        align_mask = _build_attn_align_mask(dir_on, dir_off)
    results: list[CheckResult] = []
    rope_layer = layer if layer is not None else (
        _get_num_layers(dir_on) or _get_num_layers(dir_off))
    if rope_layer:
        q_on, k_on = _load_rope_postqk(dir_on, rope_layer, fname)
        q_off, k_off = _load_rope_postqk(dir_off, rope_layer, fname)
        vecs = _rope_postqk_vec_at_pos(q_on, k_on, q_off, k_off, pos, align_mask)
        if vecs is None:
            results.append(CheckResult(
                name=f"{label}_L{rope_layer}_pos{pos}",
                metrics={"error": _diag_rope_pos_fail(q_on, q_off, pos, align_mask)}))
        else:
            qo, qf, ko, kf = vecs
            results.append(CheckResult(
                name=f"{label}_L{rope_layer}_Q_pos{pos}",
                metrics=_vec_metrics(qo, qf)))
            if ko is not None and kf is not None:
                results.append(CheckResult(
                    name=f"{label}_L{rope_layer}_K_pos{pos}",
                    metrics=_vec_metrics(ko, kf)))
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
    except ValueError as e:
        return CheckResult(name="logits", passed=False,
                           metrics={"error": str(e),
                                    "n_on": logits_on.shape[0], "n_off": logits_off.shape[0]})

    cos = _cosine_sim(on_aligned, off_aligned, dim=-1)
    cos_avg, cos_min = float(cos.mean()), float(cos.min())
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
    on_a, off_a = aligned_result
    if pos < 0 or pos >= on_a.shape[0]:
        return None, None
    return on_a[pos].reshape(-1), off_a[pos].reshape(-1)


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
        layers = [l for l in layers if l == layer]
    result_name = f"rope_freqs_L{layer}" if layer is not None else "rope_freqs"
    if not layers:
        return CheckResult(name=result_name, passed=False,
                           metrics={"error": f"layer {layer} 不在双方 rope_freqs 中"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    if align_mask is None:
        return CheckResult(name=result_name, passed=False,
                           metrics={"error": "cu_seqlens/prefix_lens 缺失"})

    max_diff = 0.0
    mismatches: list[dict] = []
    for layer_idx in layers:
        aligned_result = _align_rope_freqs_layer(on_dict[layer_idx], off_dict[layer_idx], align_mask)
        if aligned_result is None:
            continue
        on_a, off_a = aligned_result
        diff = (on_a - off_a).abs()                                  # [N,1,1,D]
        md = float(diff.max())
        max_diff = max(max_diff, md)
        if md > 0:
            token_diff = diff.squeeze(1).squeeze(1).max(dim=-1)      # values [N], indices [N]
            bad_mask = token_diff.values > 0
            for t in bad_mask.nonzero(as_tuple=True)[0].tolist():
                t = int(t)
                d = int(token_diff.indices[t])
                mismatches.append({
                    "layer": layer_idx, "token_idx": t, "dim": d,
                    "on_val": float(on_a[t, 0, 0, d]),
                    "off_val": float(off_a[t, 0, 0, d]),
                    "diff": float(token_diff.values[t]),
                })

    metrics: dict = {"max_diff": max_diff, "num_layers": len(layers)}
    if mismatches:
        metrics["mismatches"] = mismatches[:20]
        metrics["total_mismatches"] = len(mismatches)
    return CheckResult(name=result_name, passed=max_diff == 0.0, metrics=metrics)


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
    on_vec = on_a[pos].reshape(-1)
    off_vec = off_a[pos].reshape(-1)
    m = _vec_metrics(on_vec, off_vec)
    return CheckResult(name=result_name, passed=m["max_abs"] == 0.0, metrics=m)


# ════════════════════════════════════════════════════════════════
#  Attention KV: ON expanded_kv vs OFF full_kv（prefix 复用校验）
# ════════════════════════════════════════════════════════════════

def _load_attn_kv(dir_path: str, layer: int,
                  fname: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """load {key, value} for a layer from fname. Returns (key, value) or (None, None)."""
    filepath = os.path.join(dir_path, fname)
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
        layers = [l for l in layers if l == layer]
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


def _print_attn_kv(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  ON expanded_kv vs OFF full_kv（K/V 逐元素）")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    print(f"  {'LAYER':>6s}  {'K_MAXDIFF':>12s} {'K_COS':>10s}  "
          f"{'V_MAXDIFF':>12s} {'V_COS':>10s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 12} {'─' * 10}  {'─' * 8}")
    bad = []
    for layer_idx in sorted(layers):
        d = layers[layer_idx]
        kd, vd = d.get("K", {}), d.get("V", {})
        if "error" in kd or "error" in vd:
            print(f"  {layer_idx:>6d}  K:{kd.get('error','')}  V:{vd.get('error','')}")
            bad.append(layer_idx); continue
        kmd, kcos = kd["max_diff"], kd["cos_avg"]
        vmd, vcos = vd["max_diff"], vd["cos_avg"]
        ok = kmd < 1e-5 and vmd < 1e-5
        if not ok:
            bad.append(layer_idx)
        print(f"  {layer_idx:>6d}  {kmd:>12.3e} {kcos:>10.6f}  "
              f"{vmd:>12.3e} {vcos:>10.6f}  {'OK' if ok else 'DIFF':>8s}")
    print(f"\n  max_diff={m.get('max_diff')}  cos_min={m.get('cos_min')}  "
          f"{_CHECK if r.passed else _CROSS} "
          f"{'PASS（KV 一致）' if r.passed else 'FAIL（KV 不一致 → build_kv prefix 复用）'}")
    if bad:
        print(f"  ⚠ 首个 KV 不一致层: {bad[0]}")
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
        layers = [l for l in layers if l == layer]
    result_name = f"build_kv_input_v_L{layer}" if layer is not None else "build_kv_input_v"
    if not layers:
        return CheckResult(name=result_name, passed=False, metrics={"error": "no layers"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    per_layer: dict = {}
    worst_md = 0.0
    worst_cos = 1.0
    for layer_idx in layers:
        on_v = on_dict[layer_idx]
        off_v = off_dict[layer_idx]
        if on_v is None or off_v is None:
            per_layer[layer_idx] = {"error": "缺失"}; continue
        on_f = on_v.reshape(on_v.shape[0], -1).float()
        off_f = off_v.reshape(off_v.shape[0], -1).float()
        on_T, off_T = int(on_v.shape[0]), int(off_v.shape[0])
        if align_mask is not None and on_f.shape[0] != off_f.shape[0]:
            try:
                on_f, off_f = _align_packed(on_f, off_f, align_mask)
            except ValueError as e:
                per_layer[layer_idx] = {"error": str(e), "on_T": on_T, "off_T": off_T}
                continue
        diff = (on_f - off_f).abs()
        cos = _cosine_sim(on_f, off_f, dim=-1)
        md = float(diff.max())
        per_layer[layer_idx] = {"max_diff": md, "cos_avg": float(cos.mean()),
                          "cos_min": float(cos.min()), "n_tokens": on_f.shape[0],
                          "on_T": on_T, "off_T": off_T}
        worst_md = max(worst_md, md)
        worst_cos = min(worst_cos, float(cos.min()))
    passed = worst_md < 1e-5
    return CheckResult(name=result_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_md, "cos_min": worst_cos})


def _print_build_kv_input_v(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  ON build_kv_input_v vs OFF build_kv_input_v（suffix 对齐）")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    print(f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS':>10s}  "
          f"{'ON_T':>8s} {'OFF_T':>8s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 8} {'─' * 8}  {'─' * 8}")
    for layer_idx in sorted(layers):
        d = layers[layer_idx]
        if "max_diff" not in d:
            print(f"  {layer_idx:>6d}  {d.get('error', '')}  ON_T={d.get('on_T')} OFF_T={d.get('off_T')}")
            continue
        md, cos = d["max_diff"], d["cos_avg"]
        ok = md < 1e-5
        _crop = " (cropped)" if d.get("on_T") != d.get("off_T") else ""
        print(f"  {layer_idx:>6d}  {md:>12.3e} {cos:>10.6f}  "
              f"{d.get('on_T', '—'):>8} {d.get('off_T', '—'):>8}  "
              f"{'OK' if ok else 'DIFF':>8s}{_crop}")
    print(f"\n  max_diff={m.get('max_diff')}  cos_min={m.get('cos_min')}  "
          f"{_CHECK if r.passed else _CROSS} "
          f"{'PASS（build_kv 前 V 一致 → 偏由 build_kv 引入）' if r.passed else 'FAIL（build_kv 前 V 已偏 → 根因在 get_qkv/hidden_states）'}")
    print()


def cmp_hidden_states(dir_on: str, dir_off: str,
                      layer: int | None = None) -> CheckResult | None:
    """对比 ON vs OFF hidden_states（suffix 对齐，注意力层入口）。

    这是 QKV 投影的 INPUT。如果 hidden_states 一致但 V 不一致 → GEMM 精度差异；
    如果 hidden_states 就不一致 → 根因在上游（embedding / input_layernorm）。
    """
    filepath_on = os.path.join(dir_on, "hidden_states.pt")
    filepath_off = os.path.join(dir_off, "hidden_states.pt")
    if not os.path.exists(filepath_on) or not os.path.exists(filepath_off):
        return None
    on_dict = torch.load(filepath_on, weights_only=True)
    off_dict = torch.load(filepath_off, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None
    layers = sorted(set(on_dict.keys()) & set(off_dict.keys()))
    if layer is not None:
        layers = [l for l in layers if l == layer]
    result_name = f"hidden_states_L{layer}" if layer is not None else "hidden_states"
    if not layers:
        return CheckResult(name=result_name, passed=False, metrics={"error": "no layers"})

    align_mask = _build_attn_align_mask(dir_on, dir_off)
    per_layer: dict = {}
    worst_md = 0.0
    worst_cos = 1.0
    for layer_idx in layers:
        on_hs = on_dict[layer_idx]
        off_hs = off_dict[layer_idx]
        if on_hs is None or off_hs is None:
            per_layer[layer_idx] = {"error": "缺失"}; continue
        on_f = on_hs.reshape(on_hs.shape[0], -1).float()
        off_f = off_hs.reshape(off_hs.shape[0], -1).float()
        on_T, off_T = int(on_hs.shape[0]), int(off_hs.shape[0])
        if align_mask is not None and on_f.shape[0] != off_f.shape[0]:
            try:
                on_f, off_f = _align_packed(on_f, off_f, align_mask)
            except ValueError as e:
                per_layer[layer_idx] = {"error": str(e), "on_T": on_T, "off_T": off_T}
                continue
        diff = (on_f - off_f).abs()
        cos = _cosine_sim(on_f, off_f, dim=-1)
        md = float(diff.max())
        per_layer[layer_idx] = {"max_diff": md, "cos_avg": float(cos.mean()),
                          "cos_min": float(cos.min()), "n_tokens": on_f.shape[0],
                          "on_T": on_T, "off_T": off_T}
        worst_md = max(worst_md, md)
        worst_cos = min(worst_cos, float(cos.min()))
    passed = worst_md < 1e-5
    return CheckResult(name=result_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_md, "cos_min": worst_cos})


def _print_hidden_states(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  ON vs OFF hidden_states（suffix 对齐，注意力入口）")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    print(f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS':>10s}  "
          f"{'ON_T':>8s} {'OFF_T':>8s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10}  {'─' * 8} {'─' * 8}  {'─' * 8}")
    for layer_idx in sorted(layers):
        d = layers[layer_idx]
        if "max_diff" not in d:
            print(f"  {layer_idx:>6d}  {d.get('error', '')}  ON_T={d.get('on_T')} OFF_T={d.get('off_T')}")
            continue
        md, cos = d["max_diff"], d["cos_avg"]
        ok = md < 1e-5
        print(f"  {layer_idx:>6d}  {md:>12.3e} {cos:>10.6f}  "
              f"{d.get('on_T', '—'):>8} {d.get('off_T', '—'):>8}  "
              f"{'OK' if ok else 'DIFF':>8s}")
    print(f"\n  max_diff={m.get('max_diff')}  cos_min={m.get('cos_min')}  "
          f"{_CHECK if r.passed else _CROSS} "
          f"{'PASS（hidden_states 一致 → V 差异在 GEMM）' if r.passed else 'FAIL（hidden_states 不一致 → 根因上游）'}")
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
    """对比 ON/OFF 的 2D 张量（logprobs / entropy）。

    Returns ``(result, on_tensor, off_tensor)`` —— 后两者供 top-K 打印复用。
    """
    t1 = _load_tensor(dir_on, filename)
    t2 = _load_tensor(dir_off, filename)
    if t1 is None or t2 is None:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "file missing"}), t1, t2)
    if t1.shape != t2.shape:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "shape mismatch",
                                     "s_on": tuple(t1.shape),
                                     "s_off": tuple(t2.shape)}), t1, t2)
    m = mask.to(t1.device) if mask is not None else None
    if m is not None:
        m = m & ~torch.isnan(t1) & ~torch.isnan(t2)
    err = _error_abs_rel(t1, t2, m)
    n_act = int(m.sum()) if m is not None else t1.numel()
    return (CheckResult(name=name,
                        passed=n_act == 0 or err["abs_max"] <= atol,
                        metrics={"shape": tuple(t1.shape),
                                 "active": n_act,
                                 "abs_max": err["abs_max"],
                                 "abs_mean": err["abs_mean"],
                                 "rel_max": err["rel_max"],
                                 "rel_mean": err["rel_mean"],
                                 "pearson_r": _pearson_r(t1, t2, m),
                                 "atol": atol}), t1, t2)


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




def _logits_shape(dir_path: str, manifest: dict | None) -> str:
    """Shape string for logits, manifest-aware: tp>1 → show shard shape tagged.

    Under TP the file is sharded (``logits_tp{r}.pt``); report one shard's shape
    prefixed with ``tp{N}×`` so the shapes table still flags mismatches without
    pretending a plain ``logits.pt`` exists.
    """
    tp_size = (manifest or {}).get("tp_size", 1)
    if tp_size <= 1:
        return _shape_of(dir_path, "logits.pt")
    s0 = _shape_of(dir_path, "logits_tp0.pt")
    if s0 in ("(missing)", "(error)"):
        return s0
    return f"tp{tp_size}×{s0}"


def _print_topology(manifest_on: dict | None, manifest_off: dict | None) -> None:
    """Print ON/OFF parallel topology from manifests; warn on mismatch."""
    def _topo(m):
        if not m:
            return "single-card (no manifest)"
        return f"tp={m.get('tp_size', 1)} pp={m.get('pp_size', 1)} cp={m.get('cp_size', 1)}"
    print(_SEP_SINGLE + "\n  [topology]  ON vs OFF parallel config")
    print(_SEP_SINGLE)
    print(f"  ON : {_topo(manifest_on)}")
    print(f"  OFF: {_topo(manifest_off)}")
    if manifest_on and manifest_off:
        for key in ("tp_size", "pp_size", "cp_size"):
            if manifest_on.get(key) != manifest_off.get(key):
                print(f"  {_CROSS} MISMATCH on {key}: ON={manifest_on.get(key)} "
                      f"OFF={manifest_off.get(key)} — comparison may be invalid")
    print()


def _print_shapes(dir_on: str, dir_off: str, tag: str,
                  manifest_on: dict | None = None,
                  manifest_off: dict | None = None):
    """打印 ON/OFF 各 .pt 文件 shape —— 定位 shape mismatch 根因的第一手信息。"""
    print(_SEP_SINGLE + "\n  [shapes]  ON vs OFF dump shapes")
    print(_SEP_SINGLE)
    files = [
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
    for fname in files:
        if fname == "logits.pt":
            s_on = _logits_shape(dir_on, manifest_on)
            s_off = _logits_shape(dir_off, manifest_off)
        else:
            s_on, s_off = _shape_of(dir_on, fname), _shape_of(dir_off, fname)
        if s_on == "(missing)" or s_off == "(missing)":
            status = "—"
        elif s_on == s_off:
            status = "OK"
        else:
            status = f"{_CROSS} DIFF"
        print(f"  {fname:<28s} {s_on:<16s} {s_off:<16s} {status}")
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


def _print_rope_freqs(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [rope_freqs]  {_CHECK if r.passed else _CROSS} "
          f"{'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {m['error']}")
    else:
        print(f"  layers: {m.get('num_layers', '—')}  "
              f"max_diff: {m.get('max_diff', '—')}")
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
        print(f"  {_CROSS} CRITICAL — STOP.")
    print()


def _print_per_layer(r: CheckResult):
    print(_SEP_SINGLE + "\n  [attn_output]  Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = r.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'COS_AVG':>14s}  {'COS_MIN':>14s}  "
              f"{'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─' * 6}  {'─' * 14}  {'─' * 14}  {'─' * 8}  {'─' * 8}")
        bad = []
        for layer_idx in sorted(layers.keys()):
            d = layers[layer_idx]
            if "error" in d:
                print(f"  {layer_idx:>6d}  {d['error']}")
                bad.append(layer_idx)
                continue
            ok = d["cos_avg"] > _COS_AVG_PASS and d["cos_min"] > _COS_MIN_PASS
            print(f"  {layer_idx:>6d}  {d['cos_avg']:>14.6e}  {d['cos_min']:>14.6e}  "
                  f"{d['n_tokens']:>8d}  {'PASS' if ok else 'WARN':>8s}")
            if not ok:
                bad.append(layer_idx)
        if bad:
            print(f"\n  ⚠ First deviating layer: {bad[0]}")
    elif "cos_avg" in r.metrics:
        d = r.metrics
        ok = d["cos_avg"] > _COS_AVG_PASS and d["cos_min"] > _COS_MIN_PASS
        print(f"  L{d['layer']}  cos_avg={d['cos_avg']:.6e}  "
              f"cos_min={d['cos_min']:.6e}  {'PASS' if ok else 'WARN'}")
    elif "error" in r.metrics:
        print(f"  {_CROSS} {r.metrics['error']}")
    print()


def _print_rope_postqk_per_layer(r: CheckResult):
    _sec = "rope_preqk" if "preqk" in r.name else "rope_postqk"
    _stage = "Pre-RoPE" if "preqk" in r.name else "Post-RoPE"
    print(_SEP_SINGLE + f"\n  [{_sec}]  {_stage} Q/K Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = r.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'Q_MAXDIFF':>12s}  {'Q_COS_AVG':>12s}  "
              f"{'K_MAXDIFF':>12s}  {'K_COS_AVG':>12s}  "
              f"{'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}  "
              f"{'─' * 8}  {'─' * 8}")
        bad = []
        for layer_idx in sorted(layers.keys()):
            d = layers[layer_idx]
            if "error" in d:
                print(f"  {layer_idx:>6d}  {d['error']}")
                bad.append(layer_idx)
                continue
            ok = (d["Q_cos_avg"] > _COS_AVG_PASS and d["Q_cos_min"] > _COS_MIN_PASS
                  and d["K_cos_avg"] > _COS_AVG_PASS and d["K_cos_min"] > _COS_MIN_PASS)
            print(f"  {layer_idx:>6d}  {d.get('Q_max_diff', 0.0):>12.3e}  "
                  f"{d['Q_cos_avg']:>12.6e}  "
                  f"{d.get('K_max_diff', 0.0):>12.3e}  {d['K_cos_avg']:>12.6e}  "
                  f"{d['n_tokens']:>8d}  {'PASS' if ok else 'WARN':>8s}")
            if not ok:
                bad.append(layer_idx)
        if bad:
            print(f"\n  ⚠ First deviating layer: {bad[0]}")
        print(f"  （Q/K max_diff 与 build_kv_input_v 的 V max_diff 同口径，可直接对比）")
    elif "Q_cos_avg" in r.metrics:
        d = r.metrics
        ok = (d["Q_cos_avg"] > _COS_AVG_PASS and d["Q_cos_min"] > _COS_MIN_PASS
              and d["K_cos_avg"] > _COS_AVG_PASS and d["K_cos_min"] > _COS_MIN_PASS)
        print(f"  L{d['layer']}  Q_maxdiff={d.get('Q_max_diff', 0.0):.3e}  "
              f"Q_cos_avg={d['Q_cos_avg']:.6e}  "
              f"K_maxdiff={d.get('K_max_diff', 0.0):.3e}  "
              f"K_cos_avg={d['K_cos_avg']:.6e}  {'PASS' if ok else 'WARN'}")
    elif "error" in r.metrics:
        print(f"  {_CROSS} {r.metrics['error']}")
    print()


def _print_packed_token(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [packed_token]  {r.name}")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n")
        return
    for k in ["mean_abs", "max_abs", "rel_max", "rel_mean", "cos", "pearson"]:
        v = m.get(k)
        if v is not None:
            print(f"  {k:>12s}  {v:>14.6e}")
    print()


def _print_logits_packed(r: CheckResult):
    print(_SEP_SINGLE + "\n  [logits]  packed alignment (suffix aligned)")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}")
    else:
        print(f"  n_tokens={m.get('n_tokens', '—')}  "
              f"cos_avg={m.get('cos_avg', 0):.6e}  cos_min={m.get('cos_min', 0):.6e}  "
              f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print()


def _print_2d_result(r: CheckResult):
    print(_SEP_THIN)
    m = r.metrics
    if "error" in m:
        print(f"  [{r.name}]  {_CROSS} {m['error']}")
        if m.get("s_on") or m.get("s_off"):
            print(f"    ON: {m.get('s_on')}   OFF: {m.get('s_off')}")
        print()
        return
    print(f"  [{r.name}]  shape={m.get('shape')}  active={m.get('active')}  "
          f"abs_max={m.get('abs_max', 0):.6e}  rel_max={m.get('rel_max', 0):.6e}  "
          f"pearson={m.get('pearson_r', float('nan')):.8f}")
    print(f"  abs_mean={m.get('abs_mean', 0):.6e}  rel_mean={m.get('rel_mean', 0):.6e}  "
          f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
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
    for r in results:
        m = r.metrics
        shape = str(m.get("shape", m.get("error", "—")))
        am = f"{m['abs_max']:.6e}" if "abs_max" in m else "—"
        rm = f"{m['rel_max']:.6e}" if "rel_max" in m else "—"
        pr = f"{m['pearson_r']:.6f}" if m.get("pearson_r") is not None else "—"
        s = f"  {_CHECK} PASS" if r.passed else f"  {_CROSS} FAIL"
        print(f"  {r.name:<20s} {shape:<16s} {am:>12s}  {rm:>10s}  {pr:>10s}  {s}")
    print(_SEP_DOUBLE + "\n")


def _dump_json(results: list[CheckResult], path: str,
               dir_on: str, dir_off: str, tag: str, dir_off2: str | None):
    record: dict = {"dir_on": dir_on, "dir_off": dir_off, "tag": tag}
    if dir_off2:
        record["dir_off2"] = dir_off2
    record["results"] = [{**r.metrics, "name": r.name, "passed": r.passed}
                         for r in results]
    record["all_passed"] = all(r.passed for r in results)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {path}\n")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="verl080 precision comparison — ON vs OFF (packed + 2D, self-contained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--dir-off2", default=None,
                    help="Second OFF dir for OFF-vs-OFF baseline")
    ap.add_argument("--tag", required=True,
                    help="2D file tag (old / train) — logprobs_{tag}.pt, mask_{tag}.pt")
    ap.add_argument("--mask", choices=["label", "attention", "none"],
                    default="label", help="2D mask type (default: label)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific attn layer 1-indexed (default: all). "
                         "Also used by packed_token attn (default: last layer).")
    ap.add_argument("--token", type=int, default=0,
                    help="Packed token position for packed_token compare "
                         "(single int index, default: 0)")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D (default: 1e-5)")
    ap.add_argument("--topk", type=int, default=0,
                    help="top-K worst positions (0 = disabled)")
    ap.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="top-K sort: abs / rel / val (default: abs)")
    ap.add_argument("--output", "-o", default=None, help="Save report as JSON")
    args = ap.parse_args()

    _print_header(args.dir_on, args.dir_off, args.dir_off2,
                  args.tag, args.mask, args.layer)

    # ── topology diagnostics ──
    manifest_on = _load_manifest(args.dir_on)
    manifest_off = _load_manifest(args.dir_off)
    _print_topology(manifest_on, manifest_off)

    # ── shape diagnostics ──
    _print_shapes(args.dir_on, args.dir_off, args.tag,
                  manifest_on=manifest_on, manifest_off=manifest_off)

    # ── resolve 2D mask ──
    ref = _load_tensor(args.dir_off, f"logprobs_{args.tag}.pt")
    mask = None
    if ref is not None:
        mask = _resolve_mask(args.dir_off, args.mask, args.tag, tuple(ref.shape))
        if mask is not None:
            print(f"  mask: {args.mask}  shape={tuple(mask.shape)}  "
                  f"active={int(mask.sum())}\n")

    all_results: list[CheckResult] = []

    # ── RoPE pipeline per-layer：pre Q/K → rope 角度 → post Q/K ──
    # 按计算顺序串联：旋转前 Q/K → 每token旋转角度(freqs) → 旋转后 Q/K，
    # 定位分歧出现在 RoPE 哪一步（pre 就偏=上游；freqs 偏=角度表；post 才偏=旋转应用）。
    r = cmp_rope_postqk_layer(args.dir_on, args.dir_off, args.layer, stage="pre")
    if r:
        all_results.append(r)
        _print_rope_postqk_per_layer(r)

    r = cmp_rope_freqs(args.dir_on, args.dir_off, layer=args.layer)
    if r:
        all_results.append(r)
        _print_rope_freqs(r)

    r = cmp_rope_postqk_layer(args.dir_on, args.dir_off, args.layer, stage="post")
    if r:
        all_results.append(r)
        _print_rope_postqk_per_layer(r)

    # ── packed: attention KV（ON expanded vs OFF full，prefix 复用校验）──
    r = cmp_attn_kv(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_attn_kv(r)

    # ── packed: build_kv 输入 V — ON vs OFF 同源对比 ──
    r = cmp_build_kv_input_v(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_build_kv_input_v(r)

    # ── packed: hidden_states（注意力入口）— 隔离 QKV 投影 vs 上游 ──
    r = cmp_hidden_states(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_hidden_states(r)

    # ── packed: attention_output per-layer cos（RoPE 下游）──
    r = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_per_layer(r)

    # ── packed: packed_token（attn[pos] + logits[pos]，suffix 对齐后） ──
    # pos 由 --token 指定（默认 0，索引对齐后的 suffix-packed 空间）；
    # attn 用 --layer 指定的层（默认最后一层）；logits 永远最后一层。
    pos = args.token
    align_mask = _build_attn_align_mask(args.dir_on, args.dir_off)
    pt_results = cmp_packed_token(args.dir_on, args.dir_off, pos, args.layer,
                                  align_mask=align_mask)
    for r in pt_results:
        all_results.append(r)
        _print_packed_token(r)
    # packed_token top-K（per-dim，同样先对齐再取 [pos]）
    if args.topk > 0 and pt_results:
        attn_layer = args.layer if args.layer is not None else (
            _get_num_layers(args.dir_on) or _get_num_layers(args.dir_off))
        if attn_layer:
            a = _load_attn_output(args.dir_on, attn_layer)
            b = _load_attn_output(args.dir_off, attn_layer)
            vecs = _aligned_vec_at_pos(a, b, True, pos, align_mask)
            if vecs is not None:
                _print_topk_vec(vecs[0].cpu(), vecs[1].cpu(), args.topk, "val",
                                f"attn_L{attn_layer}_pos{pos}")
        logits_on = _load_logits(args.dir_on)
        logits_off = _load_logits(args.dir_off)
        vecs = _aligned_vec_at_pos(logits_on, logits_off, False, pos, align_mask)
        if vecs is not None:
            _print_topk_vec(vecs[0].cpu(), vecs[1].cpu(), args.topk,
                            "val", f"logits_pos{pos}", show_rel=False)

    # ── packed: logits（suffix 对齐） ──
    r = cmp_logits_packed(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_logits_packed(r)

    # ── RoPE pipeline packed_token：pre Q/K → rope_freqs → post Q/K（指定 pos）──
    rope_pt_results: list[CheckResult] = []
    for r in cmp_rope_postqk_token(args.dir_on, args.dir_off, pos, args.layer,
                                align_mask=align_mask, stage="pre"):
        all_results.append(r); rope_pt_results.append(r); _print_packed_token(r)
    _rf = cmp_rope_freqs_token(args.dir_on, args.dir_off, pos, args.layer,
                               align_mask=align_mask)
    if _rf is not None:
        all_results.append(_rf); rope_pt_results.append(_rf); _print_packed_token(_rf)
    for r in cmp_rope_postqk_token(args.dir_on, args.dir_off, pos, args.layer,
                                align_mask=align_mask, stage="post"):
        all_results.append(r); rope_pt_results.append(r); _print_packed_token(r)
    # rope packed_token top-K —— dim 跨 stage 对齐：以 rope_postqk 的 sort-err top-K dim 为基准，
    # rope_preqk 显示同样 dim，rope_freqs 显示 dim%D（角度按 head_dim 共享），逐 dim 追溯误差。
    if args.topk > 0 and rope_pt_results:
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
                    _D = freq_on.numel()
                    _print_vec_at_dims(freq_on.cpu(), freq_off.cpu(),
                                       [d % _D for d in q_dims],
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
                                           [d % _D for d in k_dims],
                                           f"rope_freqs_L{rope_layer}_K_pos{pos} (dim%D)")

    # ── 2D: logprobs + entropy ──
    for fname, cname in [("logprobs", "logp"), ("entropy", "entropy")]:
        fn = f"{fname}_{args.tag}.pt"
        r, t1, t2 = cmp_2d(args.dir_on, args.dir_off, fn,
                           f"{cname}_{args.tag}", mask, args.atol)
        all_results.append(r)
        _print_2d_result(r)
        if (args.topk > 0 and t1 is not None and t2 is not None
                and t1.shape == t2.shape):
            _print_topk_2d(t1.cpu(), t2.cpu(), mask, args.topk,
                           args.sort_err, r.name)

    # ── OFF vs OFF baseline ──
    if args.dir_off2:
        print(_SEP_DOUBLE + "\n  [BASELINE]  OFF vs OFF2 Noise Floor\n" + _SEP_DOUBLE)
        for fname, cname in [("logprobs", "logp"), ("entropy", "entropy")]:
            fn = f"{fname}_{args.tag}.pt"
            r, _, _ = cmp_2d(args.dir_off, args.dir_off2, fn,
                             f"bl_{cname}_{args.tag}", mask, args.atol)
            all_results.append(r)
            _print_2d_result(r)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_on, args.dir_off,
                   args.tag, args.dir_off2)


if __name__ == "__main__":
    main()
