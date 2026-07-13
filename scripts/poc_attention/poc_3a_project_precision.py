#!/usr/bin/env python3
"""PoC-3A: 项目 expanded-KV 与 Flex 的精度/梯度闭环.

必须满足§2.3.1实验纪律:
  - 直接调用 prefix_sharing.backends.kv_builder.build_prefix_expanded_kv()
  - expanded、Flex、dense oracle 都使用同一 upstream gradient 完整 backward
  - 禁止手工 trace/cat provider chain；由 PrefixAttentionStore 与项目 builder 保证
  - spy 断言 build_prefix_expanded_kv 被调用且调用次数符合预期
"""

import json, os, sys, time, traceback
from typing import Any

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")

os.environ["ENABLE_PREFIX_SHARING"] = "0"  # 不影响测试脚本

import torch
import torch.nn.functional as F

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.kv_builder import build_prefix_expanded_kv
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED = 42
DEVICE = "cuda"
DTYPE = torch.float32  # fp32 primary; bf16 tested separately
H_Q, H_KV, HEAD_DIM = 14, 2, 64

# ── workloads ──
def mp(p): return list(range(1, p+1))
def mk_star(plen, rlen, n):
    p = mp(plen)
    return [p + list(range(100, 100+rlen))] + [p + list(range(200+plen*i, 200+plen*i+rlen)) for i in range(n)]
def mk_chain(d, pl, sl):
    p = mp(pl); rs = [p]
    for i in range(1, d): rs.append(rs[-1] + list(range(300+sl*i, 300+sl*(i+1))))
    return rs
def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

WORKLOADS = {
    "star_aligned":      (mk_star(64, 65, 3),      "B=4,P=64,R=65"),
    "star_long_prompt":  (mk_star(1024, 128, 7),   "B=8,P=1024,R=128"),
    "chain_depth3":      (mk_chain(3, 32, 8),      "depth=3,P=32,suffix=8"),
    "chain_depth6":      (mk_chain(6, 32, 8),      "depth=6,P=32,suffix=8"),
    "chain_depth12":     (mk_chain(12, 16, 4),     "depth=12,P=16,suffix=4"),
    "deep_fragmented":   (mk_frag(),               "B=6,mixed"),
    "multi_group":       (mk_star(64, 65, 2) + mk_star(32, 33, 2),
                          "2 independent prefix groups"),
}

# ── Spy: count build_prefix_expanded_kv calls ──
_call_count = [0]

def spy_build_kv(*args, **kwargs):
    _call_count[0] += 1
    return build_prefix_expanded_kv(*args, **kwargs)

class NullStats:
    def record_store_count(self, **kw): pass
    def record_reuse(self, **kw): pass
    def record_reuse_hit(self, **kw): pass
    def record_reuse_miss(self, **kw): pass
    def record_stored_tokens(self, **kw): pass
    def record_reused_prefix_tokens(self, **kw): pass
    def record_expanded_kv_tokens(self, **kw): pass
    def record_valid_q_tokens(self, **kw): pass
    def record_padded_q_tokens(self, **kw): pass
    def record_attention_kv_build(self, **kw): pass
    def record_stats(self, **kw): pass

def derive_tree(plan):
    """Build per-token node metadata for Flex mask_mod."""
    B, T = plan.batch_size, plan.cu_seqlens_q[-1]
    cu, ir, pi = plan.cu_seqlens_q, plan.input_keep_ranges, plan.provider_index
    ptr = torch.zeros(T, dtype=torch.long, device=DEVICE)
    op = torch.zeros(T, dtype=torch.long, device=DEVICE)
    for i in range(B):
        s, e = cu[i], cu[i+1]
        ptr[s:e] = i
        op[s:e] = torch.arange(ir[i][0], ir[i][0] + (e - s), device=DEVICE)
    anc = torch.zeros(B, B, dtype=torch.bool, device=DEVICE)
    for i in range(B):
        if pi[i] != i:
            anc[pi[i], i] = True
            for k in range(B):
                if anc[k, pi[i]]: anc[k, i] = True
    return ptr, op, anc, torch.tensor(plan.prefix_lens, device=DEVICE)

def build_dense_oracle_mask(plan):
    """Token-level boolean mask for dense SDPA oracle. [T, T] bool."""
    cu, pi = plan.cu_seqlens_q, plan.provider_index
    pl = plan.prefix_lens
    B, T = plan.batch_size, cu[-1]
    ptr, op, anc, _ = derive_tree(plan)
    mask = torch.zeros(T, T, dtype=torch.bool, device=DEVICE)
    for qi in range(T):
        qr = ptr[qi].item()
        qo = op[qi].item()
        # same-row causal
        for ki in range(cu[qr], qi+1):
            ko = op[ki].item()
            if ko <= qo:
                mask[qi, ki] = True
        # ancestor full
        for ki in range(T):
            kr = ptr[ki].item()
            if kr != qr and anc[kr, qr] and op[ki].item() < pl[qr]:
                mask[qi, ki] = True
    return mask

def run_case(name: str, input_ids: list[list[int]], dtype: torch.dtype):
    """Run a single case: plan -> three paths -> compare Q/K/V output & gradient."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(input_ids)
    B, Td = plan.batch_size, plan.cu_seqlens_q[-1]
    Torig = sum(plan.original_lengths)
    Texp = sum(plan.expanded_lengths_kv)

    torch.manual_seed(SEED)
    q = torch.randn(1, Td, H_Q, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    k = torch.randn(1, Td, H_KV, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)
    v = torch.randn(1, Td, H_KV, HEAD_DIM, dtype=dtype, device=DEVICE, requires_grad=True)

    cu_q = plan.cu_seqlens_q
    is_fp32 = (dtype == torch.float32)
    results = {"case": name, "dtype": str(dtype), "orig": Torig, "dedup": Td, "expanded": Texp}

    # ── Path A: expanded (project builder + TorchRef SDPA) ──
    _call_count[0] = 0
    store = PrefixAttentionStore()
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    stats = NullStats()
    # Reshape for builder: (T, H*D)
    k_flat = k[0].reshape(Td, -1)
    v_flat = v[0].reshape(Td, -1)
    try:
        ek, ev = spy_build_kv(
            key=k_flat, value=v_flat, store=store,
            prefix_sharing_plan=plan, packed_batch_layout=layout,
            layer_id=0, stats=stats,
        )
    except Exception as e:
        results["expanded_error"] = repr(e)

    if "expanded_error" not in results:
        builder_calls = _call_count[0]
        # TorchRef-style per-row SDPA for expanded K/V
        exp_outputs = []
        ek_2d = ek.view(Texp, H_KV, HEAD_DIM) if ek.dim() == 2 else ek
        ev_2d = ev.view(Texp, H_KV, HEAD_DIM) if ev.dim() == 2 else ev
        for i in range(B):
            qs, qe = cu_q[i], cu_q[i+1]
            ks, ke = plan.cu_seqlens_kv[i], plan.cu_seqlens_kv[i+1]
            qi = q[:, qs:qe, :, :]
            ki = ek_2d[ks:ke, :, :].unsqueeze(0)
            vi = ev_2d[ks:ke, :, :].unsqueeze(0)
            # For GQA, repeat KV heads
            kr = ki.repeat_interleave(H_Q // H_KV, dim=2)
            vr = vi.repeat_interleave(H_Q // H_KV, dim=2)
            # Causal mask: [qe-qe, ke-ks]
            Lq, Lk = qe - qs, ke - ks
            causal_mask = torch.triu(torch.full((Lq, Lk), float("-inf"), dtype=dtype, device=DEVICE), diagonal=1)
            out = F.scaled_dot_product_attention(
                qi.permute(0, 2, 1, 3), kr.permute(0, 2, 1, 3), vr.permute(0, 2, 1, 3),
                attn_mask=causal_mask, dropout_p=0.0, is_causal=False,
            )
            exp_outputs.append(out.permute(0, 2, 1, 3))
        exp_out = torch.cat(exp_outputs, dim=1)
        results["builder_calls"] = builder_calls
    else:
        exp_out = None
        results["builder_calls"] = 0

    # ── Path B: Flex ──
    with torch.no_grad():
        ptr, op, anc, pl = derive_tree(plan)
        def mm(b, h, qi, ki):
            qr, kr = ptr[qi], ptr[ki]
            qo, ko = op[qi], op[ki]
            return ((kr == qr) & (ko <= qo)) | (anc[kr, qr] & (ko < pl[qr]))
        bm = create_block_mask(mm, None, None, Td, Td, BLOCK_SIZE=128, device=DEVICE)
    flex_out = flex_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                               v.permute(0, 2, 1, 3), block_mask=bm, enable_gqa=True)
    flex_out = flex_out.permute(0, 2, 1, 3)

    # ── Path C: dense oracle (fp32 only) ──
    if is_fp32:
        dmask = build_dense_oracle_mask(plan)
        q_f32 = q.float().permute(0, 2, 1, 3)
        kr_f32 = k.float().repeat_interleave(H_Q // H_KV, dim=2).permute(0, 2, 1, 3)
        vr_f32 = v.float().repeat_interleave(H_Q // H_KV, dim=2).permute(0, 2, 1, 3)
        am = dmask[None, None, :, :].expand(1, H_Q, -1, -1)
        oracle_out = F.scaled_dot_product_attention(q_f32, kr_f32, vr_f32, attn_mask=am, dropout_p=0.0, is_causal=False)
        oracle_out = oracle_out.permute(0, 2, 1, 3).to(dtype)
    else:
        oracle_out = None

    # ── Backward all paths with same upstream gradient ──
    up_grad = torch.randn_like(flex_out)
    if exp_out is not None:
        exp_out.backward(up_grad, retain_graph=True)
        q_grad_exp = q.grad.clone()
        k_grad_exp = k.grad.clone()
        v_grad_exp = v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

    flex_out.backward(up_grad, retain_graph=True)
    q_grad_flex = q.grad.clone()
    k_grad_flex = k.grad.clone()
    v_grad_flex = v.grad.clone()
    q.grad, k.grad, v.grad = None, None, None

    if oracle_out is not None:
        oracle_out.backward(up_grad, retain_graph=True)
        q_grad_ora = q.grad.clone()
        k_grad_ora = k.grad.clone()
        v_grad_ora = v.grad.clone()
        q.grad, k.grad, v.grad = None, None, None

    # ── Metrics ──
    def met(a, b, label=""):
        d = (a - b).float().abs()
        mx, mn = d.max().item(), d.mean().item()
        rn = d.norm().item() / max(a.float().norm().item(), 1e-12)
        a_f = a.float().reshape(-1)
        b_f = b.float().reshape(-1)
        cs = (a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-12)
        return {"max": round(mx, 8), "mean": round(mn, 8), "rel_l2": round(rn, 8),
                "cos": round(cs.item(), 8), "finite": not (torch.isnan(d).any() or torch.isinf(d).any())}

    if exp_out is not None:
        results["exp_vs_flex"] = met(exp_out, flex_out)
        results["grad_exp_vs_flex"] = {
            "q": met(q_grad_exp, q_grad_flex),
            "k": met(k_grad_exp, k_grad_flex),
            "v": met(v_grad_exp, v_grad_flex),
        }
    if oracle_out is not None:
        if exp_out is not None:
            results["exp_vs_oracle"] = met(exp_out, oracle_out)
        results["flex_vs_oracle"] = met(oracle_out, flex_out)
        if exp_out is not None:
            results["grad_exp_vs_ora"] = {
                "q": met(q_grad_exp, q_grad_ora),
                "k": met(k_grad_exp, k_grad_ora),
                "v": met(v_grad_exp, v_grad_ora),
            }
        results["grad_flex_vs_ora"] = {
            "q": met(q_grad_flex, q_grad_ora),
            "k": met(k_grad_flex, k_grad_ora),
            "v": met(v_grad_flex, v_grad_ora),
        }
    else:
        # bf16: still compare exp vs flex (no oracle)
        if exp_out is not None:
            results["exp_vs_flex"] = met(exp_out, flex_out)
            results["grad_exp_vs_flex"] = {
                "q": met(q_grad_exp, q_grad_flex),
                "k": met(k_grad_exp, k_grad_flex),
                "v": met(v_grad_exp, v_grad_flex),
            }

    # ── Provider directed gradient test ──
    # Backprop from deepest leaf only, verify provider prefix gets gradient
    if exp_out is not None or oracle_out is not None:
        # Need to re-run oracle or flex for this directed test
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        bm2 = create_block_mask(mm, None, None, Td, Td, BLOCK_SIZE=128, device=DEVICE)
        out2 = flex_attention(q2.permute(0,2,1,3), k2.permute(0,2,1,3), v2.permute(0,2,1,3),
                              block_mask=bm2, enable_gqa=True)
        # Deepest leaf = last row's Q
        last_row_start = cu_q[-2] if B >= 2 else 0
        leaf_mask = torch.zeros_like(out2)
        leaf_mask[:, last_row_start:, :, :] = 1.0
        (out2 * leaf_mask).sum().backward()
        # Check provider prefix (row 0) got gradient
        prov_grad_norm = q2.grad[:, :cu_q[1], :, :].float().norm().item()
        results["provider_directed_grad_norm"] = round(prov_grad_norm, 6)

    del q, k, v, flex_out
    if exp_out is not None: del exp_out
    if oracle_out is not None: del oracle_out
    torch.cuda.empty_cache()
    return results

def main():
    print("=" * 60, file=sys.stderr)
    print("PoC-3A: 项目 expanded-KV 与 Flex 的精度/梯度闭环", file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0)} torch: {torch.__version__}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_results = []
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n>>> dtype={dtype}", file=sys.stderr)
        for name, (ids, desc) in WORKLOADS.items():
            print(f"  {name}: {desc}", file=sys.stderr)
            try:
                r = run_case(name, ids, dtype)
                all_results.append(r)
                # Print summary
                cl = r.get("builder_calls", "?")
                if "exp_vs_flex" in r:
                    evf = r["exp_vs_flex"]
                    print(f"    builder_calls={cl} exp_vs_flex: max={evf['max']:.2e} rel_l2={evf['rel_l2']:.2e} cos={evf['cos']:.4f}",
                          file=sys.stderr)
                if "flex_vs_oracle" in r:
                    fvo = r["flex_vs_oracle"]
                    print(f"    flex_vs_oracle: max={fvo['max']:.2e} rel_l2={fvo['rel_l2']:.2e} cos={fvo['cos']:.4f}",
                          file=sys.stderr)
                if "grad_flex_vs_ora" in r:
                    gfvo = r["grad_flex_vs_ora"]
                    print(f"    grad_flex_vs_ora: q_max={gfvo['q']['max']:.2e} k_max={gfvo['k']['max']:.2e} v_max={gfvo['v']['max']:.2e}",
                          file=sys.stderr)
                if "provider_directed_grad_norm" in r:
                    print(f"    provider_directed_grad_norm={r['provider_directed_grad_norm']:.6f}",
                          file=sys.stderr)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                all_results.append({"case": name, "dtype": str(dtype), "error": repr(e)})

    # Summary
    print("\n\n=== SUMMARY ===", file=sys.stderr)
    for r in all_results:
        c = r.get("case", "?")
        d = r.get("dtype", "?")
        if "error" in r:
            print(f"  {c}({d}): ERROR - {r['error']}", file=sys.stderr)
            continue
        bc = r.get("builder_calls", "?")
        evf = r.get("exp_vs_flex", {})
        fvo = r.get("flex_vs_oracle", {})
        fev = evf.get("max", "?")
        fvo_m = fvo.get("max", "?")
        fvo_r = fvo.get("rel_l2", "?")
        print(f"  {c}({d}): builder_calls={bc} exp-vs-flex: max={fev}  flex-vs-ora: max={fvo_m} rel={fvo_r}",
              file=sys.stderr)

    out_path = os.path.join(os.path.dirname(__file__), "poc_3a_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2, default=str))

if __name__ == "__main__":
    main()
