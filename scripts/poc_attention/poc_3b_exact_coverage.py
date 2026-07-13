#!/usr/bin/env python3
"""补充3B：逐 token BlockMask coverage 与 tail-block 统计 (§2.3.8.2)

严格断言：
  assert torch.all((~logical_mask) | scheduled_coverage)  # 逐元素
  partial + full == reconstructed_scheduled_blocks
  partial block mask_mod 不开放 sibling/cross-tree
"""

import json, os, sys, time, traceback

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner

SEED = 42; DEVICE = "cuda"
H_Q, H_KV, HD = 14, 2, 64

def mp(p): return list(range(1, p+1))
def mk_star(plen,rlen,n):
    p=mp(plen); return [p+list(range(100,100+rlen))]+[p+list(range(200+plen*i,200+plen*i+rlen)) for i in range(n)]
def mk_chain(d,pl,sl):
    p=mp(pl); r=[p]
    for i in range(1,d): r.append(r[-1]+list(range(300+sl*i,300+sl*(i+1)))); return r
def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

WORKLOADS = {
    "star_long_prompt": (mk_star(512,64,7), "B=8,P=512,R=64"),
    "chain_depth12": (mk_chain(12,16,4), "depth=12"),
    "deep_fragmented": (mk_frag(), "B=6"),
}

def derive_tree(plan):
    B,T=plan.batch_size,plan.cu_seqlens_q[-1]; cu,ir,pi=plan.cu_seqlens_q,plan.input_keep_ranges,plan.provider_index
    ptr=torch.zeros(T,dtype=torch.long,device=DEVICE); op=torch.zeros(T,dtype=torch.long,device=DEVICE)
    for i in range(B): s,e=cu[i],cu[i+1]; ptr[s:e]=i; op[s:e]=torch.arange(ir[i][0],ir[i][0]+(e-s),device=DEVICE)
    anc=torch.zeros(B,B,dtype=torch.bool,device=DEVICE)
    for i in range(B):
        if pi[i]!=i: anc[pi[i],i]=True
        for k in range(B):
            if anc[k,pi[i]]: anc[k,i]=True
    return ptr,op,anc,torch.tensor(plan.prefix_lens,device=DEVICE)

def build_logical_mask(plan):
    B,T=plan.batch_size,plan.cu_seqlens_q[-1]; cu,pi=plan.cu_seqlens_q,plan.provider_index
    pl=plan.prefix_lens; ptr,op,anc,_=derive_tree(plan)
    mask=torch.zeros(T,T,dtype=torch.bool,device=DEVICE)
    for qi in range(T):
        qr=ptr[qi].item(); qo=op[qi].item()
        for ki in range(cu[qr],qi+1):
            if op[ki].item()<=qo: mask[qi,ki]=True
        for ki in range(T):
            kr=ptr[ki].item()
            if kr!=qr and anc[kr,qr] and op[ki].item()<pl[qr]: mask[qi,ki]=True
    return mask

def reconstruct_coverage(bm, BS, T, mask_mod_fn):
    """Reconstruct [T,T] bool coverage from kv_indices + full_kv_indices.
    kv_indices: blocks requiring mask_mod (partial)
    full_kv_indices: blocks where all pairs are visible (full)
    """
    coverage = torch.zeros(T, T, dtype=torch.bool, device=DEVICE)
    nQ_blocks = (T + BS - 1) // BS
    block_count = 0

    # Partial blocks from kv_indices
    ki = bm.kv_indices  # [1,1,nQ, max_per_q]
    for qb in range(nQ_blocks):
        qs, qe = qb * BS, min((qb + 1) * BS, T)
        n_partial = bm.kv_num_blocks[0, 0, qb].item()
        for kbi in range(n_partial):
            kvb = ki[0,0,qb,kbi].item()
            if kvb < 0: break
            block_count += 1
            ks, ke = kvb * BS, min((kvb + 1) * BS, T)
            # For each token pair in this block, apply mask_mod
            for qi in range(qs, qe):
                for ki_idx in range(ks, ke):
                    if mask_mod_fn(0, 0, qi, ki_idx):
                        coverage[qi, ki_idx] = True

    # Full blocks from full_kv_indices
    fki = bm.full_kv_indices
    if fki is not None:
        for qb in range(nQ_blocks):
            qs, qe = qb * BS, min((qb + 1) * BS, T)
            n_full = bm.full_kv_num_blocks[0, 0, qb].item()
            for kbi in range(n_full):
                kvb = fki[0,0,qb,kbi].item()
                if kvb < 0: break
                block_count += 1
                ks, ke = kvb * BS, min((kvb + 1) * BS, T)
                # Full block: ALL token pairs in this block are visible
                coverage[qs:qe, ks:ke] = True

    return coverage, block_count

def run_case(name, input_ids):
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(input_ids)
    T = plan.cu_seqlens_q[-1]
    logical_mask = build_logical_mask(plan)
    logical_pairs = int(logical_mask.sum().item())

    ptr, op, anc, pl = derive_tree(plan)
    def mask_mod_fn(b,h,qi,ki):
        qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
        return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))

    results = {"case": name, "T": T, "logical_pairs": logical_pairs,
               "from_kv_blocks_available": hasattr(BlockMask, "from_kv_blocks")}

    for BS in [64, 128, 256]:
        if BS > T and BS > 64: continue

        bm = create_block_mask(mask_mod_fn, None, None, T, T, BLOCK_SIZE=BS, device=DEVICE)

        # ── 1. Correct counting ──
        partial_count = int(bm.kv_num_blocks.sum().item())
        full_count = int(bm.full_kv_num_blocks.sum().item())
        total_scheduled = partial_count + full_count

        # ── 2. Reconstruct coverage ──
        coverage, reconstructed_blocks = reconstruct_coverage(bm, BS, T, mask_mod_fn)

        # ── 3. Exact element-wise assertion ──
        all_visible_covered = torch.all((~logical_mask) | coverage)
        missing = (~coverage) & logical_mask
        missing_count = int(missing.sum().item())

        # ── 4. Block count consistency ──
        blocks_match = (total_scheduled == reconstructed_blocks)

        # ── 5. Count scheduled elements by actual tail lengths ──
        # Rebuild from block indices with per-token coverage
        scheduled_elements = int(coverage.sum().item())
        sched_logical_ratio = round(scheduled_elements / max(logical_pairs, 1), 3)

        bsr = {
            "block_size": BS,
            "partial_blocks": partial_count,
            "full_blocks": full_count,
            "total_scheduled_blocks": total_scheduled,
            "reconstructed_blocks": reconstructed_blocks,
            "blocks_match": blocks_match,
            "logical_pairs": logical_pairs,
            "scheduled_elements": scheduled_elements,
            "sched_logical_ratio": sched_logical_ratio,
            "all_visible_covered": bool(all_visible_covered),
            "missing_visible_pairs": missing_count,
        }

        # ── 6. FP32 output/QKV gradient alignment with dense oracle ──
        # (cross-check that coverage correctly maps to visible pairs)
        torch.manual_seed(SEED)
        q = torch.randn(1, T, H_Q, HD, dtype=torch.float32, device=DEVICE, requires_grad=True)
        k = torch.randn(1, T, H_KV, HD, dtype=torch.float32, device=DEVICE, requires_grad=True)
        v = torch.randn(1, T, H_KV, HD, dtype=torch.float32, device=DEVICE, requires_grad=True)

        flex_out = flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                                  block_mask=bm, enable_gqa=True).permute(0,2,1,3)

        # Dense oracle
        kr = k.float().repeat_interleave(H_Q//H_KV, dim=2).permute(0,2,1,3)
        vr = v.float().repeat_interleave(H_Q//H_KV, dim=2).permute(0,2,1,3)
        am = logical_mask[None,None,:,:].expand(1,H_Q,-1,-1)
        import torch.nn.functional as F
        oracle_out = F.scaled_dot_product_attention(
            q.float().permute(0,2,1,3), kr, vr, attn_mask=am, dropout_p=0.0, is_causal=False
        ).permute(0,2,1,3)

        diff = (flex_out - oracle_out).float().abs()
        bsr["flex_vs_oracle_max"] = round(diff.max().item(), 8)
        bsr["flex_vs_oracle_mean"] = round(diff.mean().item(), 8)

        del q, k, v, bm, flex_out, oracle_out
        torch.cuda.empty_cache()

        results[f"bs{BS}"] = bsr

    return results

def main():
    print("=" * 60, file=sys.stderr)
    print("补充3B: 逐token BlockMask coverage 与 tail-block 统计", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_failed = False
    all_results = []
    for name, (ids, desc) in WORKLOADS.items():
        print(f"\n--- {name}: {desc} ---", file=sys.stderr)
        try:
            r = run_case(name, ids)
            all_results.append(r)
            for bk, bv in r.items():
                if not bk.startswith("bs"): continue
                status = "PASS" if (bv["all_visible_covered"] and bv["blocks_match"] and bv["flex_vs_oracle_max"] < 1e-4) else "FAIL"
                print(f"  {bk}: {status} partial={bv['partial_blocks']} full={bv['full_blocks']} "
                      f"recon={bv['reconstructed_blocks']} blocks_match={bv['blocks_match']} "
                      f"all_visible={bv['all_visible_covered']} missing={bv['missing_visible_pairs']} "
                      f"ora_max={bv['flex_vs_oracle_max']:.2e}", file=sys.stderr)
                if status == "FAIL":
                    all_failed = True
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            all_results.append({"case": name, "error": repr(e)})
            all_failed = True

    out_path = os.path.join(os.path.dirname(__file__), "poc_3b_exact_coverage_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2, default=str))

    if all_failed:
        print("\nFAILED: some cases did not pass coverage assertions", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
