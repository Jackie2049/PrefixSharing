"""verl080 精度比较 — ON vs OFF（自包含，不依赖 v070 cmp_diag）。

本文件**完全自包含**，不从 ``cmp_diag`` / ``diagnostic_dump`` import 任何东西，
便于将来独立维护（verl070 对应文件可能被废弃）。

对比项分两类：

  **packed（suffix 对齐）**
    - attention_output per-layer cos   每层 attention 输出余弦相似度
    - provider_first_token              packed[0] 的 provider 哨兵检查
    - reuser_first_suffix_token         每个 reuser 的首个 suffix token（restore 边界）
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
    rope_freqs_on.pt        dict {layer: [T_on,1,1,D]} ON per-token RoPE 角度
    rope_freqs_off.pt       dict {layer: [L0,1,1,D]}   OFF raw 角度表（freqs[p]=p*inv_freq）
    prefix_lens.pt          [B]              ON=plan.prefix_lens / OFF=全0
    cu_seqlens_q.pt         [B+1]            NestedTensor offsets（ON 裁剪后 / OFF 完整）
    cu_seqlens_q_logits.pt  [B+1]            logits packed 边界（同上）

Usage:
    # 完整对比（attn per-layer + first_token + logits + logprobs + entropy）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off --tag old

    # 只看某一层 attention（1-indexed）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off \\
        --tag old --layer 12

    # top-K 误差最大位置（2D + first_token）
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


def _first_token_metrics(a_vec: torch.Tensor, b_vec: torch.Tensor) -> dict:
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
    fp = os.path.join(dir_path, filename)
    return torch.load(fp, weights_only=True).float() if os.path.exists(fp) else None


def _load_manifest(dir_path: str) -> dict | None:
    """Load ``parallel_info.json`` written by the dump layer (topology + scopes).

    Returns None when absent (single-card or pre-manifest dumps) → callers fall
    back to tp_size==1 behavior (plain filenames, single-card compatible).
    """
    fp = os.path.join(dir_path, "parallel_info.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_logits(dir_path: str, manifest: dict | None = None) -> torch.Tensor | None:
    """Load packed logits, gathering tp vocab shards to full vocab when tp>1.

    tp_size==1 (or no manifest) → single ``logits.pt`` (single-card compatible).
    tp_size>1  → concat ``logits_tp{0..tp-1}.pt`` on the vocab (last) dim,
                 reconstructing ``[N, V]`` so ON-vs-OFF compares on the same
                 full-vocab coordinate system as single-card.  A missing shard
                 aborts the reconstruction (returns None) rather than silently
                 comparing partial vocab.
    """
    if manifest is None:
        manifest = _load_manifest(dir_path)
    tp_size = (manifest or {}).get("tp_size", 1)
    if tp_size <= 1:
        return _load_tensor(dir_path, "logits.pt")
    shards = []
    for t in range(tp_size):
        s = _load_tensor(dir_path, f"logits_tp{t}.pt")
        if s is None:
            return None
        shards.append(s)
    return torch.cat(shards, dim=-1)


def _load_packed_meta(dir_path: str,
                      cu_fname: str = "cu_seqlens_q.pt") -> dict | None:
    """加载 cu_seqlens + prefix_lens（suffix 对齐所需）。"""
    fp = os.path.join(dir_path, cu_fname)
    if not os.path.exists(fp):
        fp = os.path.join(dir_path, "cu_seqlens_q.pt")
        if not os.path.exists(fp):
            return None
    pl_fp = os.path.join(dir_path, "prefix_lens.pt")
    if not os.path.exists(pl_fp):
        return None
    return {"cu_seqlens": torch.load(fp, weights_only=True),
            "prefix_lens": torch.load(pl_fp, weights_only=True)}


def _load_attn_output(dir_path: str, layer: int) -> torch.Tensor | None:
    """加载单层 attn_output（attn_outputs.pt = dict {layer: tensor}）。"""
    fp = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(fp):
        return None
    d = torch.load(fp, weights_only=True)
    return d.get(layer) if isinstance(d, dict) else None


def _get_num_layers(dir_path: str) -> int:
    fp = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(fp):
        return 0
    d = torch.load(fp, weights_only=True)
    return max(d.keys()) if isinstance(d, dict) and d else 0


def cmp_input_ids(dir_on: str, dir_off: str, tag: str) -> CheckResult:
    """Verify that ON/OFF consumed exactly the same original token batch."""
    on_ids = _load_tensor(dir_on, f"input_ids_{tag}.pt")
    off_ids = _load_tensor(dir_off, f"input_ids_{tag}.pt")
    if on_ids is None or off_ids is None:
        return CheckResult(
            name="input_ids",
            passed=False,
            metrics={"error": f"input_ids_{tag}.pt missing in ON or OFF dump"},
        )
    if on_ids.shape != off_ids.shape:
        return CheckResult(
            name="input_ids",
            passed=False,
            metrics={"error": "shape mismatch", "s_on": tuple(on_ids.shape),
                     "s_off": tuple(off_ids.shape)},
        )
    different = int((on_ids != off_ids).sum())
    return CheckResult(
        name="input_ids",
        passed=different == 0,
        metrics={"shape": tuple(on_ids.shape), "different_tokens": different},
    )


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

def _logits_first_token(lo: torch.Tensor, lf: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """取 packed[0] 的 full-vocab 向量（logits 词表恒在最后一维）。"""
    lo_2d = lo.reshape(-1, lo.size(-1))
    lf_2d = lf.reshape(-1, lf.size(-1))
    return lo_2d[0, :].contiguous(), lf_2d[0, :].contiguous()


def _logits_ensure_token_major(lo: torch.Tensor, lf: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """确保 logits 为 2D [N, V]（token-major），vocab 在最后一维。"""
    return (lo.reshape(-1, lo.size(-1)).contiguous(),
            lf.reshape(-1, lf.size(-1)).contiguous())


# ════════════════════════════════════════════════════════════════
#  Packed compare: attention_output / first_token / logits
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
    ma = _load_packed_meta(dir_on)
    mb = _load_packed_meta(dir_off)
    if ma is None or mb is None:
        return None
    cu_off = mb["cu_seqlens"]
    T = int(cu_off[-1]) if cu_off.numel() > 0 else 0
    if T == 0:
        return None
    return _build_alignment_mask(cu_off, ma["prefix_lens"], T)


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

    fa = os.path.join(dir_on, "attn_outputs.pt")
    fb = os.path.join(dir_off, "attn_outputs.pt")
    if not os.path.exists(fa) or not os.path.exists(fb):
        return None
    da = torch.load(fa, weights_only=True)
    db = torch.load(fb, weights_only=True)
    if not isinstance(da, dict) or not isinstance(db, dict):
        return None

    on_layers, off_layers = set(da.keys()), set(db.keys())
    if on_layers != off_layers:
        return CheckResult(name="attn_per_layer", passed=False,
                           metrics={"error": "layer set mismatch",
                                    "on_layers": sorted(on_layers),
                                    "off_layers": sorted(off_layers)})

    results = {}
    for lyr in sorted(on_layers):
        a, b = da[lyr], db[lyr]
        need = align_mask is not None and a.shape[0] != b.shape[0]
        try:
            results[lyr] = _cos_for_layer(a, b, align_mask if need else None)
        except ValueError as e:
            results[lyr] = {"error": str(e)}
    passed = bool(results) and all(
        "error" not in metrics
        and metrics["cos_avg"] > _COS_AVG_PASS
        and metrics["cos_min"] > _COS_MIN_PASS
        for metrics in results.values()
    )
    return CheckResult(name="attn_per_layer", passed=passed,
                       metrics={"layers": results})


def _token_major(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize HF [B,H,L,D] or packed [T,H,D] tensors to [T,H,D]."""
    if tensor.dim() == 4:
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1], tensor.shape[-1])
    return tensor


def _load_tensor_dict(dir_path: str, filename: str) -> dict | None:
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return None
    result = torch.load(fp, weights_only=True)
    return result if isinstance(result, dict) else None


def cmp_attention_inputs(dir_on: str, dir_off: str) -> CheckResult | None:
    """Compare post-RoPE Q/K/V inputs and report the first diverging layer."""
    on_dict = _load_tensor_dict(dir_on, "attn_inputs.pt")
    off_dict = _load_tensor_dict(dir_off, "attn_inputs.pt")
    if on_dict is None or off_dict is None:
        return None
    if set(on_dict) != set(off_dict):
        return CheckResult(name="attn_inputs", passed=False,
                           metrics={"error": "layer set mismatch"})
    align_mask = _build_attn_align_mask(dir_on, dir_off)
    layers: dict[int, dict] = {}
    for layer in sorted(on_dict):
        layer_metrics = {}
        for name in ("query", "key", "value"):
            on_tensor = _token_major(on_dict[layer][name])
            off_tensor = _token_major(off_dict[layer][name])
            try:
                metrics = _cos_for_layer(on_tensor, off_tensor, align_mask)
            except ValueError as error:
                return CheckResult(name="attn_inputs", passed=False,
                                   metrics={"error": f"L{layer} {name}: {error}"})
            layer_metrics[name] = metrics
        layers[layer] = layer_metrics
    passed = all(
        metric["cos_avg"] > _COS_AVG_PASS and metric["cos_min"] > _COS_MIN_PASS
        for values in layers.values() for metric in values.values()
    )
    return CheckResult(name="attn_inputs", passed=passed, metrics={"layers": layers})


def cmp_expanded_kv(dir_on: str, dir_off: str) -> CheckResult | None:
    """Compare ON store/load-expanded KV with OFF's full baseline KV."""
    expanded = _load_tensor_dict(dir_on, "expanded_kv.pt")
    baseline = _load_tensor_dict(dir_off, "attn_inputs.pt")
    if expanded is None or baseline is None:
        return None
    if set(expanded) != set(baseline):
        return CheckResult(name="expanded_kv", passed=False,
                           metrics={"error": "layer set mismatch"})
    layers: dict[int, dict] = {}
    for layer in sorted(expanded):
        layer_metrics = {}
        for name in ("key", "value"):
            on_tensor = _token_major(expanded[layer][name])
            off_tensor = _token_major(baseline[layer][name])
            try:
                layer_metrics[name] = _cos_for_layer(on_tensor, off_tensor)
            except ValueError as error:
                return CheckResult(name="expanded_kv", passed=False,
                                   metrics={"error": f"L{layer} {name}: {error}"})
        layers[layer] = layer_metrics
    passed = all(
        metric["cos_avg"] > _COS_AVG_PASS and metric["cos_min"] > _COS_MIN_PASS
        for values in layers.values() for metric in values.values()
    )
    return CheckResult(name="expanded_kv", passed=passed, metrics={"layers": layers})


def cmp_first_token(dir_on: str, dir_off: str) -> list[CheckResult]:
    """packed[0] 对比：最后一层 attn[0] + logits[0]。

    第一个序列（row 0）永远是 provider（完整序列），packed[0] 是完整 suffix token，
    可直接对比无需对齐。
    """
    results: list[CheckResult] = []
    last = _get_num_layers(dir_on) or _get_num_layers(dir_off)

    if last:
        a = _load_attn_output(dir_on, last)
        b = _load_attn_output(dir_off, last)
        if a is not None and b is not None:
            a0 = a.squeeze(1) if a.dim() == 3 else a
            b0 = b.squeeze(1) if b.dim() == 3 else b
            metrics = _first_token_metrics(a0[0], b0[0])
            results.append(CheckResult(
                name="provider_first_token_attn",
                passed=metrics["cos"] > _COS_AVG_PASS,
                metrics=metrics))

    lo = _load_logits(dir_on)
    lf = _load_logits(dir_off)
    if lo is not None and lf is not None:
        lo_first, lf_first = _logits_first_token(lo, lf)
        metrics = _first_token_metrics(lo_first, lf_first)
        results.append(CheckResult(
            name="provider_first_token_logits",
            passed=metrics["cos"] > _COS_AVG_PASS,
            metrics=metrics))
    return results


def _reuser_first_suffix_positions(
    on_meta: dict, off_meta: dict,
) -> list[tuple[int, int, int]]:
    """Return ``(row, on_index, off_index)`` at every reuser restore boundary."""
    on_cu, off_cu = on_meta["cu_seqlens"], off_meta["cu_seqlens"]
    prefix_lens = on_meta["prefix_lens"]
    if on_cu.numel() != off_cu.numel() or prefix_lens.numel() != on_cu.numel() - 1:
        raise ValueError("incompatible ON/OFF packed metadata")
    positions = []
    for row, prefix_len in enumerate(prefix_lens.tolist()):
        if prefix_len > 0:
            positions.append((row, int(on_cu[row]), int(off_cu[row]) + int(prefix_len)))
    return positions


def cmp_reuser_first_suffix_token(dir_on: str, dir_off: str) -> list[CheckResult]:
    """Compare every reuser's first suffix token, including the restore boundary."""
    on_meta = _load_packed_meta(dir_on, "cu_seqlens_q_logits.pt")
    off_meta = _load_packed_meta(dir_off, "cu_seqlens_q_logits.pt")
    if on_meta is None or off_meta is None:
        return [CheckResult(name="reuser_first_suffix", passed=False,
                            metrics={"error": "packed metadata missing"})]
    try:
        positions = _reuser_first_suffix_positions(on_meta, off_meta)
    except ValueError as error:
        return [CheckResult(name="reuser_first_suffix", passed=False,
                            metrics={"error": str(error)})]
    if not positions:
        return []

    results: list[CheckResult] = []
    last = _get_num_layers(dir_on) or _get_num_layers(dir_off)
    if last:
        on_attn, off_attn = _load_attn_output(dir_on, last), _load_attn_output(dir_off, last)
        if on_attn is not None and off_attn is not None:
            on_attn = on_attn.squeeze(1) if on_attn.dim() == 3 else on_attn
            off_attn = off_attn.squeeze(1) if off_attn.dim() == 3 else off_attn
            for row, on_index, off_index in positions:
                metrics = _first_token_metrics(on_attn[on_index], off_attn[off_index])
                metrics.update({"row": row, "on_packed_index": on_index,
                                "off_packed_index": off_index})
                results.append(CheckResult(
                    name=f"reuser_first_suffix_attn_row{row}",
                    passed=metrics["cos"] > _COS_AVG_PASS, metrics=metrics))

    on_logits, off_logits = _load_logits(dir_on), _load_logits(dir_off)
    if on_logits is not None and off_logits is not None:
        on_logits, off_logits = _logits_ensure_token_major(on_logits, off_logits)
        for row, on_index, off_index in positions:
            metrics = _first_token_metrics(on_logits[on_index], off_logits[off_index])
            metrics.update({"row": row, "on_packed_index": on_index,
                            "off_packed_index": off_index})
            results.append(CheckResult(
                name=f"reuser_first_suffix_logits_row{row}",
                passed=metrics["cos"] > _COS_AVG_PASS, metrics=metrics))
    return results


def cmp_logits_packed(dir_on: str, dir_off: str) -> CheckResult | None:
    """全 packed logits suffix 对齐 + per-token cosine。"""
    lo = _load_logits(dir_on)
    lf = _load_logits(dir_off)
    if lo is None or lf is None:
        return None
    lo, lf = _logits_ensure_token_major(lo, lf)

    ma = _load_packed_meta(dir_on, "cu_seqlens_q_logits.pt")
    mb = _load_packed_meta(dir_off, "cu_seqlens_q_logits.pt")
    if ma is None or mb is None:
        return None
    T_off = int(mb["cu_seqlens"][-1]) if mb["cu_seqlens"].numel() > 0 else 0
    if T_off == 0 or lo.shape[0] == 0 or lf.shape[0] == 0:
        return None

    align_mask = _build_alignment_mask(mb["cu_seqlens"], ma["prefix_lens"], T_off)
    try:
        on_aligned, off_aligned = _align_packed(lo, lf, align_mask)
    except ValueError as e:
        return CheckResult(name="logits", passed=False,
                           metrics={"error": str(e),
                                    "n_on": lo.shape[0], "n_off": lf.shape[0]})

    cos = _cosine_sim(on_aligned, off_aligned, dim=-1)
    cos_avg, cos_min = float(cos.mean()), float(cos.min())
    return CheckResult(name="logits",
                       passed=cos_avg > _COS_AVG_PASS and cos_min > _COS_MIN_PASS,
                       metrics={"n_tokens": on_aligned.shape[0],
                                "cos_avg": cos_avg, "cos_min": cos_min})


def cmp_rope_freqs(dir_on: str, dir_off: str) -> CheckResult | None:
    """对比 pre-RoPE 角度表（angle table，非 cos/sin）— suffix 对齐。

    ON  ``rope_freqs_on.pt``:  per-token 角度 dict {layer: [T_on, 1, 1, D]}
        （已 index_select 到 packed_position_ids，每 token 实际旋转角度）
    OFF ``rope_freqs_off.pt``: raw 角度表 dict {layer: [L0, 1, 1, D]}
        （freqs[p] = p * inv_freq，未切片）

    OFF per-token 角度从 raw 表按 cu_seqlens_off 重建（每段取 ``[:seg_len]``），
    再与 ON 用同一 suffix 对齐（cu_seqlens + prefix_lens）后逐元素比 max_diff。
    角度是 RoPE 的输入，应精确相等（``max_diff == 0``）。
    """
    fa = os.path.join(dir_on, "rope_freqs_on.pt")
    fb = os.path.join(dir_off, "rope_freqs_off.pt")
    if not os.path.exists(fa) or not os.path.exists(fb):
        return None
    on_dict = torch.load(fa, weights_only=True)
    off_dict = torch.load(fb, weights_only=True)
    if not isinstance(on_dict, dict) or not isinstance(off_dict, dict):
        return None

    la, lb = set(on_dict.keys()), set(off_dict.keys())
    if la != lb:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "layer set mismatch",
                                    "on_layers": sorted(la),
                                    "off_layers": sorted(lb)})

    mb = _load_packed_meta(dir_off)
    if mb is None:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "OFF cu_seqlens missing"})
    cu_off = mb["cu_seqlens"]
    T_off = int(cu_off[-1]) if cu_off.numel() > 0 else 0

    ma = _load_packed_meta(dir_on)
    if ma is None:
        return CheckResult(name="rope_freqs", passed=False,
                           metrics={"error": "ON prefix_lens missing"})
    align_mask = _build_alignment_mask(cu_off, ma["prefix_lens"], T_off)

    seqlens = (cu_off[1:] - cu_off[:-1]).tolist()

    max_diff = 0.0
    mismatches: list[dict] = []
    for lyr in sorted(la):
        on_freqs = on_dict[lyr]                            # [T_on, 1, 1, D]
        # 从 raw 表重建 OFF per-token：每段 [:seg_len]
        off_freqs = torch.cat(
            [off_dict[lyr][:s, :, :, :] for s in seqlens], dim=0)  # [T_off, 1, 1, D]

        try:
            on_aligned, off_aligned = _align_packed(
                on_freqs, off_freqs, align_mask)
        except ValueError as e:
            return CheckResult(name="rope_freqs", passed=False,
                               metrics={"error": f"align failed L{lyr}: {e}"})

        diff = (on_aligned - off_aligned).abs()            # [N, 1, 1, D]
        md = float(diff.max())
        max_diff = max(max_diff, md)

        if md > 0:
            token_diff = diff.squeeze(1).squeeze(1).max(dim=-1)  # values [N], indices [N]
            bad_mask = token_diff.values > 0
            for t in bad_mask.nonzero(as_tuple=True)[0].tolist():
                t = int(t)
                d = int(token_diff.indices[t])
                mismatches.append({
                    "layer": lyr, "token_idx": t, "dim": d,
                    "on_val": float(on_aligned[t, 0, 0, d]),
                    "off_val": float(off_aligned[t, 0, 0, d]),
                    "diff": float(token_diff.values[t]),
                })

    metrics: dict = {"max_diff": max_diff, "num_layers": len(la)}
    if mismatches:
        metrics["mismatches"] = mismatches[:20]
        metrics["total_mismatches"] = len(mismatches)
    return CheckResult(name="rope_freqs", passed=max_diff == 0.0, metrics=metrics)


# ════════════════════════════════════════════════════════════════
#  2D mask loading
# ════════════════════════════════════════════════════════════════

def _load_mask_2d(dir_path: str, mask_kind: str, tag: str) -> torch.Tensor | None:
    """加载 2D mask：``label_mask_{tag}.pt`` / ``attention_mask_{tag}.pt``。"""
    if mask_kind == "none":
        return None
    fname = f"{mask_kind}_mask_{tag}.pt"  # label_mask_{tag} / attention_mask_{tag}
    fp = os.path.join(dir_path, fname)
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True).to(torch.bool)


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
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return "(missing)"
    try:
        obj = torch.load(fp, weights_only=True)
        if isinstance(obj, dict):
            # per-layer dict（attn_outputs / rope_freqs_*）：显示层数 + 首层 shape
            sample = next(iter(obj.values())) if obj else None
            sample_shape = f",{tuple(sample.shape)}" if sample is not None else ""
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
        "prefix_lens.pt",
        "cu_seqlens_q.pt",
    ]
    print(f"  {'FILE':<28s} {'ON':<16s} {'OFF':<16s} {'STATUS'}")
    print(f"  {'─' * 28} {'─' * 16} {'─' * 16} {'─' * 10}")
    for fname in files:
        if fname == "logits.pt":
            # TP-sharded: per-rank logits_tp{r}.pt, not a plain logits.pt
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
        for lyr in sorted(layers.keys()):
            d = layers[lyr]
            if "error" in d:
                print(f"  {lyr:>6d}  {d['error']}")
                bad.append(lyr)
                continue
            ok = d["cos_avg"] > _COS_AVG_PASS and d["cos_min"] > _COS_MIN_PASS
            print(f"  {lyr:>6d}  {d['cos_avg']:>14.6e}  {d['cos_min']:>14.6e}  "
                  f"{d['n_tokens']:>8d}  {'PASS' if ok else 'WARN':>8s}")
            if not ok:
                bad.append(lyr)
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


def _print_first_token(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [token_sentinel]  {r.name}")
    print(_SEP_SINGLE)
    m = r.metrics
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
                    topk: int, sort_by: str, label: str):
    """1D 向量 top-K（first_token per-dim）。"""
    abs_err = (on_vec - off_vec).abs()
    rel_err = abs_err / torch.maximum(on_vec.abs(), off_vec.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    else:  # "val"
        sort_key = torch.maximum(on_vec.abs(), off_vec.abs())
    _, idx = sort_key.topk(min(topk, sort_key.numel()))
    idx = idx.to(torch.long)
    print(f"\n  [{label}]  top-{topk} dims (sort by {sort_by})")
    print(f"  {'DIM':>6s}  {'ON':>14s}  {'OFF':>14s}  {'ABS_ERR':>12s}  {'REL_ERR':>12s}")
    for i in idx.tolist():
        print(f"  {i:>6d}  {float(on_vec[i]):>14.6e}  {float(off_vec[i]):>14.6e}"
              f"  {float(abs_err[i]):>12.6e}  {float(rel_err[i]):>12.6e}")


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
                    help="Compare specific attn layer 1-indexed (default: all)")
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

    # ── parallel topology (manifest-driven: TP shards, future SP/PP) ──
    manifest_on = _load_manifest(args.dir_on)
    manifest_off = _load_manifest(args.dir_off)
    _print_topology(manifest_on, manifest_off)

    # ── shape diagnostics ──
    _print_shapes(args.dir_on, args.dir_off, args.tag, manifest_on, manifest_off)

    # ── resolve 2D mask ──
    ref = _load_tensor(args.dir_off, f"logprobs_{args.tag}.pt")
    mask = None
    if ref is not None:
        mask = _resolve_mask(args.dir_off, args.mask, args.tag, tuple(ref.shape))
        if mask is not None:
            print(f"  mask: {args.mask}  shape={tuple(mask.shape)}  "
                  f"active={int(mask.sum())}\n")

    all_results: list[CheckResult] = []

    # The replay fixture must make ON/OFF consume the identical original batch.
    # Packed lengths intentionally differ, but input IDs must not.
    r = cmp_input_ids(args.dir_on, args.dir_off, args.tag)
    all_results.append(r)

    # ── packed: rope 角度（suffix 对齐，应精确相等 max_diff==0） ──
    r = cmp_rope_freqs(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_rope_freqs(r)

    for compare in (cmp_attention_inputs, cmp_expanded_kv):
        r = compare(args.dir_on, args.dir_off)
        if r:
            all_results.append(r)
            _print_per_layer(r)

    # ── packed: attention_output per-layer cos ──
    r = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_per_layer(r)

    # ── packed: first_token（attn[0] + logits[0]） ──
    ft_results = cmp_first_token(args.dir_on, args.dir_off)
    for r in ft_results:
        all_results.append(r)
        _print_first_token(r)

    # packed[0] is only a provider sentinel.  Check the actual reuser boundary
    # separately: this is where prefix-last logprob restoration matters.
    reuser_results = cmp_reuser_first_suffix_token(args.dir_on, args.dir_off)
    for r in reuser_results:
        all_results.append(r)
        _print_first_token(r)
    # first_token top-K（per-dim）
    if args.topk > 0 and ft_results:
        last = _get_num_layers(args.dir_on) or _get_num_layers(args.dir_off)
        if last:
            a = _load_attn_output(args.dir_on, last)
            b = _load_attn_output(args.dir_off, last)
            if a is not None and b is not None:
                a0 = (a.squeeze(1) if a.dim() == 3 else a)[0].cpu()
                b0 = (b.squeeze(1) if b.dim() == 3 else b)[0].cpu()
                _print_topk_vec(a0, b0, args.topk, "val", "first_token_attn")
        lo = _load_logits(args.dir_on)
        lf = _load_logits(args.dir_off)
        if lo is not None and lf is not None:
            lo_f, lf_f = _logits_first_token(lo, lf)
            _print_topk_vec(lo_f.cpu(), lf_f.cpu(), args.topk,
                            args.sort_err, "first_token_logits")

    # ── packed: logits（suffix 对齐） ──
    r = cmp_logits_packed(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_logits_packed(r)

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
    return 0 if all_results and all(result.passed for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
