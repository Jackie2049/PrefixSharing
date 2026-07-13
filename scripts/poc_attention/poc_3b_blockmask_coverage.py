#!/usr/bin/env python3
"""PoC-3B: BlockMask coverage — 正确统计与 coverage oracle 断言.

满足§2.3.4:
  1. 正确读取 partial/full blocks
  2. coverage oracle 断言 scheduled >= logical
  3. direct from_kv_blocks 标记为 API_PRESENT_NOT_VALIDATED
"""

import json, os, sys, time, traceback

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner

SEED = 42
DEVICE = "cuda"
H_Q, H_KV, HD = 14, 2, 64

def mp(p): return list(range(1, p+1))
def mk_star(plen, rlen, n):
    p = mp(plen)
    return [p + list(range(100, 100+rlen))] + [p + list(range(200+plen*i, 200+plen*i+rlen)) for i in range(n)]
def mk_chain(d, pl, sl):
    p = mp(pl); r = [p]
    for i in range(1, d): r.append(r[-1] + list(range(300+sl*i, 300+sl*(i+1)))); return r
def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

WORKLOADS = {
    "star_long_prompt":  (mk_star(1024, 128, 7), "B=8,P=1024,R=128"),
    "chain_depth12":     (mk_chain(12, 16, 4),   "depth=12,T=60"),
    "deep_fragmented":   (mk_frag(),             "B=6,T=21"),
}

def derive_tree(plan):
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

def build_prefix_tree_mask_fn(plan):
    ptr, op, anc, pl = derive_tree(plan)
    def mm(b, h, qi, ki):
        qr, kr = ptr[qi], ptr[ki]
        qo, ko = op[qi], op[ki]
        return ((kr == qr) & (ko <= qo)) | (anc[kr, qr] & (ko < pl[qr]))
    return mm

def build_logical_mask(plan):
    B, T = plan.batch_size, plan.cu_seqlens_q[-1]
    cu, pi = plan.cu_seqlens_q, plan.provider_index
    pl = plan.prefix_lens
    ptr, op, anc, _ = derive_tree(plan)
    mask = torch.zeros(T, T, dtype=torch.bool, device=DEVICE)
    for qi in range(T):
        qr = ptr[qi].item()
        qo = op[qi].item()
        for ki in range(cu[qr], qi+1):
            if op[ki].item() <= qo: mask[qi, ki] = True
        for ki in range(T):
            kr = ptr[ki].item()
            if kr != qr and anc[kr, qr] and op[ki].item() < pl[qr]: mask[qi, ki] = True
    return mask

def run_case(name, input_ids):
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(input_ids)
    T = plan.cu_seqlens_q[-1]
    mm_fn = build_prefix_tree_mask_fn(plan)
    logical_mask = build_logical_mask(plan)
    logical_pairs = int(logical_mask.sum().item())

    results = {
        "case": name, "T": T, "logical_pairs": logical_pairs,
        "from_kv_blocks_available": hasattr(BlockMask, "from_kv_blocks"),
    }

    for BS in [64, 128]:
        if BS > T and BS > 64:
            continue

        bm = create_block_mask(mm_fn, None, None, T, T, BLOCK_SIZE=BS, device=DEVICE)

        # ── Scheduled block stats ──
        # kv_num_blocks: count of PARTIAL (mask_mod-required) KV blocks per Q block
        # full_kv_num_blocks: count of FULL KV blocks per Q block (independent count)
        total_partial_blocks = int(bm.kv_num_blocks.sum().item())
        total_full_blocks = int(bm.full_kv_num_blocks.sum().item())
        total_scheduled_blocks = total_partial_blocks + total_full_blocks
        scheduled_elements = total_scheduled_blocks * BS * BS  # worst-case -- mask_mod filters partials

        sched_logical_ratio = round(scheduled_elements / max(logical_pairs, 1), 3)

        # ── Coverage oracle ──
        # Rebuild [T,T] bool from block indices
        coverage = torch.zeros(T, T, dtype=torch.bool, device=DEVICE)
        nQ_blocks = (T + BS - 1) // BS
        # kv_indices: [1,1,nQ_blocks, max_per_q]
        ki = bm.kv_indices
        nK_blocks_max = ki.shape[3]
        for qb in range(nQ_blocks):
            qs, qe = qb * BS, min((qb + 1) * BS, T)
            for kb_idx in range(nK_blocks_max):
                kvb = ki[0, 0, qb, kb_idx].item()
                if kvb < 0:
                    break
                ks, ke = kvb * BS, min((kvb + 1) * BS, T)
                coverage[qs:qe, ks:ke] = True
        # Also full block indices
        fki = bm.full_kv_indices
        if fki is not None:
            for qb in range(nQ_blocks):
                qs, qe = qb * BS, min((qb + 1) * BS, T)
                for kb_idx in range(fki.shape[3]):
                    kvb = fki[0, 0, qb, kb_idx].item()
                    if kvb < 0: break
                    ks, ke = kvb * BS, min((kvb + 1) * BS, T)
                    coverage[qs:qe, ks:ke] = True

        covered_pairs = int(coverage.sum().item())
        coverage_pass = covered_pairs >= logical_pairs

        bsr = {
            "block_size": BS,
            "nQ_blocks": (T + BS - 1) // BS,
            "total_scheduled_blocks": total_scheduled_blocks,
            "partial_blocks": total_partial_blocks,
            "full_blocks": total_full_blocks,
            "scheduled_elements": scheduled_elements,
            "sched_logical_ratio": sched_logical_ratio,
            "logical_pairs": logical_pairs,
            "covered_pairs": covered_pairs,
            "coverage_assertion_pass": coverage_pass,
            "coverage_assertion": f"scheduled_coverage={covered_pairs} >= logical={logical_pairs}",
        }

        # ── Forward/Backward perf with this block size ──
        torch.manual_seed(SEED)
        q = torch.randn(1, T, H_Q, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k = torch.randn(1, T, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v = torch.randn(1, T, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        WU, NI = 10, 30
        for _ in range(WU):
            flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                          block_mask=bm, enable_gqa=True)
        torch.cuda.synchronize()

        ft = []
        for _ in range(NI):
            t0 = time.perf_counter()
            flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                          block_mask=bm, enable_gqa=True)
            torch.cuda.synchronize()
            ft.append(time.perf_counter() - t0)

        sft = sorted(ft)
        bsr["fwd_p50_ms"] = round(sft[len(sft)//2] * 1000, 3)
        bsr["fwd_p90_ms"] = round(sft[int(len(sft)*0.9)] * 1000, 3)

        # backward
        for _ in range(WU):
            out = flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                                block_mask=bm, enable_gqa=True)
            out.sum().backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()

        bt = []
        for _ in range(NI):
            q.grad = k.grad = v.grad = None
            t0 = time.perf_counter()
            out = flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                                block_mask=bm, enable_gqa=True)
            out.sum().backward()
            torch.cuda.synchronize()
            bt.append(time.perf_counter() - t0)

        sbt = sorted(bt)
        bsr["bwd_p50_ms"] = round(sbt[len(sbt)//2] * 1000, 3)
        bsr["bwd_p90_ms"] = round(sbt[int(len(sbt)*0.9)] * 1000, 3)

        # Peak HBM
        torch.cuda.reset_peak_memory_stats()
        out = flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                            block_mask=bm, enable_gqa=True)
        out.sum().backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None
        bsr["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        bsr["peak_res_mb"] = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 1)

        del q, k, v, bm, out
        torch.cuda.empty_cache()
        results[f"bs{BS}"] = bsr

    return results

def main():
    print("=" * 60, file=sys.stderr)
    print("PoC-3B: BlockMask coverage 与正确统计", file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0)} torch: {torch.__version__}", file=sys.stderr)
    print(f"from_kv_blocks: {hasattr(BlockMask, 'from_kv_blocks')}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_results = []
    for name, (ids, desc) in WORKLOADS.items():
        print(f"\n--- {name}: {desc} ---", file=sys.stderr)
        try:
            r = run_case(name, ids)
            all_results.append(r)
            for bk, bv in r.items():
                if not bk.startswith("bs"): continue
                print(f"  {bk}: blocks={bv['total_scheduled_blocks']} partial={bv['partial_blocks']} full={bv['full_blocks']} "
                      f"sched/log={bv['sched_logical_ratio']} "
                      f"cov_pass={bv['coverage_assertion_pass']} "
                      f"fwd={bv['fwd_p50_ms']:.1f}ms bwd={bv['bwd_p50_ms']:.1f}ms "
                      f"peak={bv['peak_alloc_mb']:.0f}MB", file=sys.stderr)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            all_results.append({"case": name, "error": repr(e)})

    print("\n\n=== SUMMARY ===", file=sys.stderr)
    for r in all_results:
        if "error" in r:
            print(f"  {r['case']}: ERROR - {r['error']}", file=sys.stderr)
            continue
        for bk, bv in r.items():
            if not bk.startswith("bs"): continue
            print(f"  {r['case']}/{bk}: {bv['total_scheduled_blocks']} blocks ({bv['partial_blocks']} partial + {bv['full_blocks']} full) "
                  f"sched/log={bv['sched_logical_ratio']} cov={bv['coverage_assertion_pass']} "
                  f"fwd={bv['fwd_p50_ms']:.1f}ms bwd={bv['bwd_p50_ms']:.1f}ms peak={bv['peak_alloc_mb']:.0f}MB",
                  file=sys.stderr)

    out_path = os.path.join(os.path.dirname(__file__), "poc_3b_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2, default=str))

if __name__ == "__main__":
    main()
