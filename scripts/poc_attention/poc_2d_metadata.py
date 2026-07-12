#!/usr/bin/env python3
"""PoC-2D: metadata lifecycle — layout/BlockMask 跨层复用验证.

模拟 24-layer attention loop, 验证：
  1. prefix tree layout & BlockMask 是否只构造一次
  2. 所有 layer 复用同一个 immutable 对象
  3. 20 个连续 micro-batch (相同/不同 shape) 的 cache 行为
  4. old-logprob/ref/actor 场景的 reuse 可行性
"""

import json, sys, time, traceback
import torch

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED=42; DEVICE="cuda"
H_Q,H_KV,HD=14,2,64; DTYPE=torch.bfloat16
N_LAYERS=24

def mp(p): return list(range(1,p+1))
def mk_star(plen,rlen,n):
    p=mp(plen)
    return [p+list(range(100,100+rlen))]+[p+list(range(200+plen*i,200+plen*i+rlen)) for i in range(n)]
def mk_chain(d,pl,sl):
    p=mp(pl); r=[p]
    for i in range(1,d): r.append(r[-1]+list(range(300+sl*i,300+sl*(i+1)))); return r

def derive_tree(plan):
    B,T=plan.batch_size,plan.cu_seqlens_q[-1]; cu,ir,pi=plan.cu_seqlens_q,plan.input_keep_ranges,plan.provider_index
    ptr=torch.zeros(T,dtype=torch.long); op=torch.zeros(T,dtype=torch.long)
    for i in range(B):
        s,e=cu[i],cu[i+1]; ptr[s:e]=i; op[s:e]=torch.arange(ir[i][0],ir[i][0]+(e-s))
    anc=torch.zeros(B,B,dtype=torch.bool)
    for i in range(B):
        if pi[i]!=i: anc[pi[i],i]=True
        for k in range(B):
            if anc[k,pi[i]]: anc[k,i]=True
    return {"T":T,"ptr":ptr,"op":op,"anc":anc,"pl":torch.tensor(plan.prefix_lens)}

# Counter for BlockMask builds
blockmask_build_count = [0]
tree_signature_cache = {}

def build_or_reuse_blockmask(plan, tree):
    """Build BlockMask once per unique tree signature, return cached."""
    # Tree signature: (batch_size, prefix_lens, kept_lengths_q, provider_index)
    sig = (plan.batch_size, tuple(plan.prefix_lens), tuple(plan.kept_lengths_q), tuple(plan.provider_index))
    if sig in tree_signature_cache:
        return tree_signature_cache[sig], True  # cached hit
    # Build new
    ptr,op,anc,pl=tree["ptr"].cuda(),tree["op"].cuda(),tree["anc"].cuda(),tree["pl"].cuda()
    def mm(b,h,qi,ki):
        qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
        return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
    bm = create_block_mask(mm,None,None,tree["T"],tree["T"],BLOCK_SIZE=128,device=DEVICE)
    blockmask_build_count[0] += 1
    tree_signature_cache[sig] = bm
    return bm, False  # new build

def simulate_microbatch(micro_batch_id, input_ids):
    """模拟一个 micro-batch: plan → layout → N 层 attention (复用同一个 BlockMask)."""
    config = PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
    t_cpu = time.perf_counter()
    plan = PrefixSharingPlanner(config).plan(input_ids, micro_batch_id=micro_batch_id)
    cpu_ms = (time.perf_counter() - t_cpu) * 1000
    Td = plan.cu_seqlens_q[-1]

    t0 = time.perf_counter()
    tree = derive_tree(plan)
    bm, is_hit = build_or_reuse_blockmask(plan, tree)
    mask_ms = (time.perf_counter() - t0) * 1000

    # Generate QKV and run N-layer attention
    torch.manual_seed(SEED + micro_batch_id)
    q = torch.randn(1,Td,H_Q,HD,dtype=DTYPE,device=DEVICE)
    k = torch.randn(1,Td,H_KV,HD,dtype=DTYPE,device=DEVICE)
    v = torch.randn(1,Td,H_KV,HD,dtype=DTYPE,device=DEVICE)

    t_attn = time.perf_counter()
    # 在第一层做 flex_attention, 其他层重复用同一个 BlockMask
    for layer in range(N_LAYERS):
        out = flex_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),block_mask=bm,enable_gqa=True)
        # 把输出接回 Q 模拟残差
        q = q + out.permute(0,2,1,3)
    torch.cuda.synchronize()
    attn_ms = (time.perf_counter() - t_attn) * 1000

    tree_sig = (plan.batch_size, tuple(plan.prefix_lens), tuple(plan.kept_lengths_q), tuple(plan.provider_index))

    del q,k,v; torch.cuda.empty_cache()
    return {
        "micro_batch": micro_batch_id,
        "tree_signature": str(hash(tree_sig) % 10000),
        "rows": plan.batch_size,
        "dedup_tokens": Td,
        "orig_tokens": sum(plan.original_lengths),
        "expanded_tokens": sum(plan.expanded_lengths_kv),
        "cpu_plan_ms": round(cpu_ms, 2),
        "blockmask_ms": round(mask_ms, 2),
        "blockmask_new_build": not is_hit,
        "layers": N_LAYERS,
        "total_attn_ms": round(attn_ms, 2),
        "per_layer_attn_ms": round(attn_ms / N_LAYERS, 3),
    }

def main():
    print("="*60,file=sys.stderr); print("PoC-2D: metadata lifecycle & cross-layer reuse",file=sys.stderr); print("="*60,file=sys.stderr)

    # 20 micro-batches: mix of star, chain, and repeated shapes
    workloads = []
    for i in range(10):
        plen = 128 + (i % 3) * 32  # 128, 160, 192
        rlen = 64 + (i % 2) * 16    # 64, 80
        workloads.append(("star", mk_star(plen, rlen, 4 + i % 3)))
    for i in range(6):
        workloads.append(("chain", mk_chain(6, 32 + (i % 2) * 8, 8)))
    for i in range(4):
        workloads.append(("star", mk_star(512, 64, 8)))  # 4 identical shape

    print(f"Total micro-batches: {len(workloads)} ({N_LAYERS} layers each)",file=sys.stderr)

    results = []
    for i, (typ, ids) in enumerate(workloads):
        r = simulate_microbatch(i, ids)
        results.append(r)
        if i < 3 or i >= len(workloads) - 2 or r["blockmask_new_build"]:
            print(f"  MB#{i:2d} ({typ:5s}): sig={r['tree_signature']} T={r['dedup_tokens']:4d} "
                  f"BM_build={r['blockmask_new_build']} BM_ms={r['blockmask_ms']:.1f} "
                  f"attn={r['total_attn_ms']:.1f}ms ({r['per_layer_attn_ms']:.2f}ms/layer)",
                  file=sys.stderr)

    # Summary
    builds = [r for r in results if r["blockmask_new_build"]]
    hits = [r for r in results if not r["blockmask_new_build"]]
    unique_sigs = set(r["tree_signature"] for r in results)

    print(f"\n=== SUMMARY ===",file=sys.stderr)
    print(f"  Total micro-batches: {len(results)}",file=sys.stderr)
    print(f"  Unique tree signatures: {len(unique_sigs)}",file=sys.stderr)
    print(f"  BlockMask new builds: {len(builds)}",file=sys.stderr)
    print(f"  BlockMask cache hits: {len(hits)}",file=sys.stderr)
    print(f"  Total BlockMask build time: {sum(r['blockmask_ms'] for r in builds):.0f}ms",file=sys.stderr)
    print(f"  Total attention time across {N_LAYERS} layers * {len(results)} MBs: "
          f"{sum(r['total_attn_ms'] for r in results):.0f}ms",file=sys.stderr)
    print(f"  Per-layer avg: {sum(r['total_attn_ms'] for r in results) / len(results) / N_LAYERS:.3f}ms",file=sys.stderr)

    # Reuse analysis for old-logprob/ref/actor
    reuse_same = [r for r in results[16:20]]  # same shape
    print(f"\n  Reuse analysis (same shape MB#16-19):",file=sys.stderr)
    if reuse_same:
        all_same_sig = all(r["tree_signature"] == reuse_same[0]["tree_signature"] for r in reuse_same)
        print(f"    Same tree signature: {all_same_sig}")
        print(f"    4 identical micro-batches would share 1 BlockMask → 75% build savings")

    print(f"\n  Fallback scenario:",file=sys.stderr)
    small_mbs = [r for r in results if r["dedup_tokens"] < 100]
    print(f"    Micro-batches with T<100 (fallback candidates): {len(small_mbs)}/{len(results)}")

    print("\n"+json.dumps(results,indent=2))

if __name__=="__main__":
    main()
