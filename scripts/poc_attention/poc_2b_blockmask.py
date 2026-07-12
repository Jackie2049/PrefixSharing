#!/usr/bin/env python3
"""PoC-2B: BlockMask 真实调度统计 + from_kv_blocks 探测.

对每个 workload 和 block size，从实际 BlockMask 元数据提取：
  - kv_num_blocks, full_kv_num_blocks
  - 统计 full/partial QK blocks
  - 计算真实 scheduled block elements vs logical attention elements
  - 对比 generic mask_mod vs direct from_kv_blocks (若可用)

Usage:
  cd /path/to/PrefixSharing
  PYTHONPATH=prefix-sharing python scripts/poc_attention/poc_2b_blockmask.py
"""

import json, sys, time, inspect, traceback
import torch

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

SEED=42; DEVICE="cuda"
H_Q,H_KV,HD=14,2,64; DTYPE=torch.bfloat16
WU,NI=20,100

def mp(p): return list(range(1,p+1))
def mk_star(plen,rlen,n):
    p=mp(plen)
    return [p+list(range(100,100+rlen))]+[p+list(range(200+plen*i,200+plen*i+rlen)) for i in range(n)]

def mk_chain(d,pl,sl):
    p=mp(pl); r=[p]
    for i in range(1,d): r.append(r[-1]+list(range(300+sl*i,300+sl*(i+1))))
    return r

def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

def derive_tree(plan):
    B,T=plan.batch_size,plan.cu_seqlens_q[-1]; cu,ir,pi=plan.cu_seqlens_q,plan.input_keep_ranges,plan.provider_index
    ptr=torch.zeros(T,dtype=torch.long); op=torch.zeros(T,dtype=torch.long)
    for i in range(B):
        s,e=cu[i],cu[i+1]; ptr[s:e]=i; op[s:e]=torch.arange(ir[i][0],ir[i][0]+(e-s))
    anc=torch.zeros(B,B,dtype=torch.bool)
    for i in range(B):
        if pi[i]!=i:
            anc[pi[i],i]=True
            for k in range(B):
                if anc[k,pi[i]]: anc[k,i]=True
    return {"T":T,"ptr":ptr,"op":op,"anc":anc,"pl":torch.tensor(plan.prefix_lens)}

def logical_pairs(plan, tree):
    """Count semantically visible QK pairs for prefix-tree."""
    cu=plan.cu_seqlens_q; B=plan.batch_size; T=cu[-1]
    ptr,op,anc,pl=tree["ptr"],tree["op"],tree["anc"],tree["pl"]
    count=0
    for i in range(B):
        s,e=cu[i],cu[i+1]; this_pl=pl[i].item()
        count+=(e-s)*(e-s+1)//2  # same-row causal
        for kk in range(T):
            ki=ptr[kk].item()
            if ki!=i and anc[ki,i] and op[kk].item()<this_pl:
                count+=(e-s)
    return count

def run_blockmask_analysis(name, input_ids):
    config=PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
    plan=PrefixSharingPlanner(config).plan(input_ids)
    tree=derive_tree(plan)
    T=tree["T"]
    total_orig=sum(plan.original_lengths)
    total_exp=sum(plan.expanded_lengths_kv)
    log_pairs=logical_pairs(plan, tree)

    ptr,op,anc,pl=tree["ptr"].cuda(),tree["op"].cuda(),tree["anc"].cuda(),tree["pl"].cuda()

    def mkm():
        def mm(b,h,qi,ki):
            qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
            return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
        return mm

    # Check BlockMask API
    bm_api={
        "from_kv_blocks":hasattr(BlockMask,"from_kv_blocks"),
        "blockmask_attrs":[a for a in dir(BlockMask) if not a.startswith("_")]
    }
    try:
        sig=str(inspect.signature(BlockMask.from_kv_blocks))
        bm_api["from_kv_blocks_sig"]=sig
    except:
        bm_api["from_kv_blocks_sig"]="N/A"

    bs_results={}
    for bs in [64,128,256]:
        if bs > T and bs != 64: continue
        print(f"  bs={bs}...",file=sys.stderr)

        # Cold build
        t0=time.perf_counter()
        bm=create_block_mask(mkm(),None,None,T,T,BLOCK_SIZE=bs,device=DEVICE)
        torch.cuda.synchronize(); cold_ms=(time.perf_counter()-t0)*1000

        # Warm build
        t0=time.perf_counter()
        bm2=create_block_mask(mkm(),None,None,T,T,BLOCK_SIZE=bs,device=DEVICE)
        torch.cuda.synchronize(); warm_ms=(time.perf_counter()-t0)*1000
        del bm2

        # Extract block metadata
        # kv_num_blocks: [n_batch, n_head, total_q_blocks] — number of KV blocks per Q block
        # full_kv_num_blocks: [n_batch, n_head, total_q_blocks] — number of full KV blocks
        try:
            knb=bm.kv_num_blocks; fknb=bm.full_kv_num_blocks
            print(f"  kv_num_blocks: {knb.shape} {knb.dtype}",file=sys.stderr)
            # knb shape: [B, nH, nQ_blocks]
            nQ_blocks=knb.shape[-1]
            total_scheduled=knb.sum().item()  # total KV blocks scheduled
            total_full=fknb.sum().item()      # total full KV blocks
            total_partial=total_scheduled-total_full
            sched_elements=int(total_scheduled)*bs*bs  # block elements
        except Exception as e:
            print(f"  ERROR reading block metadata: {e}",file=sys.stderr)
            nQ_blocks=(T+bs-1)//bs
            total_scheduled=nQ_blocks*nQ_blocks  # worst case
            total_full=total_partial=-1
            sched_elements=total_scheduled*bs*bs

        sched_logical_ratio=sched_elements/max(log_pairs,1)

        # Forward/backward perf with this block size
        # generate QKV quickly
        torch.manual_seed(SEED)
        q=torch.randn(1,T,H_Q,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)
        k=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)
        v=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)

        def fa(qt,kt,vt,bm_):
            return flex_attention(qt.permute(0,2,1,3),kt.permute(0,2,1,3),vt.permute(0,2,1,3),block_mask=bm_,enable_gqa=True)

        # warmup
        for _ in range(WU): fa(q,k,v,bm)
        torch.cuda.synchronize()
        # fwd
        ft=[]
        for _ in range(NI):
            t0=time.perf_counter(); fa(q,k,v,bm); torch.cuda.synchronize()
            ft.append(time.perf_counter()-t0)
        f50,f90=torch.tensor(ft).quantile(torch.tensor([0.5,0.9])).tolist()
        # fwd+bwd
        for _ in range(WU):
            fa(q,k,v,bm).sum().backward(); q.grad=k.grad=v.grad=None
        torch.cuda.synchronize()
        bt=[]
        for _ in range(NI):
            q.grad=k.grad=v.grad=None
            t0=time.perf_counter(); fa(q,k,v,bm).sum().backward(); torch.cuda.synchronize()
            bt.append(time.perf_counter()-t0)
        b50,b90=torch.tensor(bt).quantile(torch.tensor([0.5,0.9])).tolist()

        # Peak HBM (fwd+bwd)
        torch.cuda.reset_peak_memory_stats()
        fa(q,k,v,bm).sum().backward(); torch.cuda.synchronize()
        pa=torch.cuda.max_memory_allocated(); pr=torch.cuda.max_memory_reserved()
        q.grad=k.grad=v.grad=None

        bs_results[f"bs{bs}"]={
            "cold_ms":round(cold_ms,1),"warm_ms":round(warm_ms,1),
            "fwd_p50_ms":round(f50,3),"fwd_p90_ms":round(f90,3),
            "bwd_p50_ms":round(b50,3),"bwd_p90_ms":round(b90,3),
            "peak_alloc_mb":round(pa/1024/1024,1),"peak_res_mb":round(pr/1024/1024,1),
            "nQ_blocks":nQ_blocks,
            "total_scheduled_blocks":int(total_scheduled),
            "full_blocks":int(total_full),"partial_blocks":int(total_partial),
            "scheduled_elements":sched_elements,
            "logical_pairs":log_pairs,
            "sched_logical_ratio":round(sched_logical_ratio,3),
            "dedup_tokens":T,"expanded_tokens":total_exp,"orig_tokens":total_orig,
        }
        print(f"    blocks={int(total_scheduled)} full={int(total_full)} partial={int(total_partial)} "
              f"sched/logical={sched_logical_ratio:.3f}",file=sys.stderr)

        del q,k,v,bm; torch.cuda.empty_cache()

    return {"case":name,"desc":f"orig={total_orig} dedup={T} expanded={total_exp}",
            "blockmask_api":bm_api,"results":bs_results}

def main():
    print("="*60,file=sys.stderr); print("PoC-2B: BlockMask 真实调度统计",file=sys.stderr); print("="*60,file=sys.stderr)
    print(f"GPU:{torch.cuda.get_device_name(0)} torch={torch.__version__}",file=sys.stderr)

    # from_kv_blocks check
    print(f"\nBlockMask API:",file=sys.stderr)
    print(f"  from_kv_blocks: {hasattr(BlockMask,'from_kv_blocks')}",file=sys.stderr)
    if hasattr(BlockMask,"from_kv_blocks"):
        print(f"  signature: {inspect.signature(BlockMask.from_kv_blocks)}",file=sys.stderr)
    print(f"  attrs: {[a for a in dir(BlockMask) if not a.startswith('_')]}",file=sys.stderr)

    workloads={
        "star_long_prompt": (mk_star(1024,128,8), "B=8,P=1024,R=128"),
        "chain_depth12": (mk_chain(12,16,4), "B=12 depth=12"),
        "deep_fragmented": (mk_frag(), "B=6 mixed"),
    }

    all_results=[]
    for name,(ids,desc) in workloads.items():
        print(f"\n--- {name}: {desc} ---",file=sys.stderr)
        try:
            r=run_blockmask_analysis(name,ids); all_results.append(r)
        except Exception as e:
            traceback.print_exc(file=sys.stderr); all_results.append({"case":name,"error":str(e)})

    print("\n=== SUMMARY ===",file=sys.stderr)
    for r in all_results:
        c=r.get("case","?")
        if "error" in r: print(f"  {c}: ERROR - {r['error']}",file=sys.stderr); continue
        print(f"  {c}: {r.get('desc','')}",file=sys.stderr)
        for bsn,br in r.get("results",{}).items():
            print(f"    {bsn}: blocks={br['total_scheduled_blocks']} full={br['full_blocks']} "
                  f"partial={br['partial_blocks']} sched/log={br['sched_logical_ratio']:.3f} "
                  f"fwd={br['fwd_p50_ms']:.1f}ms bwd={br['bwd_p50_ms']:.1f}ms "
                  f"peak={br['peak_alloc_mb']:.0f}MB",file=sys.stderr)

    print("\n"+json.dumps(all_results,indent=2))

if __name__=="__main__":
    main()
