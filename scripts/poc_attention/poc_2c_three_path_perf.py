#!/usr/bin/env python3
"""PoC-2C: 三路径 attention module 速度与完整 HBM.

对 no_sharing、star_long_prompt、chain_depth、deep_fragmented：
  - ps_off_fa: full orig tokens + flash_attn_varlen (bf16 baseline)
  - ps_on_expanded_fa: trimmed Q + build_kv expanded K/V + flash_attn_varlen
  - ps_on_dedup_flex: trimmed Q/K/V + PrefixTree BlockMask + flex_attention

每个 mode 在独立进程级清理后测 HBM 和时间拆分。

Usage:
  PYTHONPATH=prefix-sharing python scripts/poc_attention/poc_2c_three_path_perf.py
"""

import json, sys, time, traceback
import torch

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from flash_attn.flash_attn_interface import flash_attn_varlen_func

SEED=42; DEVICE="cuda"; DTYPE=torch.bfloat16
H_Q,H_KV,HD=14,2,64
WU,NI=20,100

def mp(p): return list(range(1,p+1))
def mk_star(plen,rlen,n):
    p=mp(plen)
    return [p+list(range(100,100+rlen))]+[p+list(range(200+plen*i,200+plen*i+rlen)) for i in range(n)]
def mk_chain(d,pl,sl):
    p=mp(pl); r=[p]
    for i in range(1,d): r.append(r[-1]+list(range(300+sl*i,300+sl*(i+1)))); return r
def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

WORKLOADS={
    "no_sharing":([list(range(1+128*i,1+128*(i+1))) for i in range(8)],"B=8,L=128"),
    "star_long_prompt":(mk_star(1024,128,8),"B=8,P=1024,R=128"),
    "chain_depth6":(mk_chain(6,32,8),"depth=6,P=32,suffix=8"),
    "deep_fragmented":(mk_frag(),"B=6,mixed"),
}

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

def run_mode(mode, q, k, v, plan):
    """Run one mode and return (out, timings_dict)."""
    t={}
    cu_q,cu_kv,pi,pl=plan.cu_seqlens_q,plan.cu_seqlens_kv,plan.provider_index,plan.prefix_lens
    B=plan.batch_size

    if mode=="ps_off_fa":
        # Full original tokens: attention per original sequence
        t0=time.perf_counter()
        outputs=[]
        for i in range(B):
            orig_len=plan.original_lengths[i]
            fs=sum(plan.original_lengths[:i]); fe=fs+orig_len
            qi=q[:,fs:fe,:,:]; ki=k[:,fs:fe,:,:]; vi=v[:,fs:fe,:,:]
            Q=qi.reshape(-1,H_Q,HD); K=ki.reshape(-1,H_KV,HD); V=vi.reshape(-1,H_KV,HD)
            cq=torch.tensor([0,Q.shape[0]],device=DEVICE,dtype=torch.int32)
            ck=torch.tensor([0,K.shape[0]],device=DEVICE,dtype=torch.int32)
            o=flash_attn_varlen_func(Q,K,V,cq,ck,Q.shape[0],K.shape[0],0.0,causal=True)
            outputs.append(o.view(1,-1,H_Q,HD))
        out=torch.cat(outputs,dim=1)
        t["fwd"]=time.perf_counter()-t0; return out,t

    elif mode=="ps_on_expanded_fa":
        t0=time.perf_counter()
        # Build expanded K/V
        ek=torch.zeros(1,sum(plan.expanded_lengths_kv),H_KV,HD,dtype=DTYPE,device=DEVICE)
        ev=torch.zeros_like(ek)
        for i in range(B):
            qs,qe=cu_q[i],cu_q[i+1]; ks,ke=cu_kv[i],cu_kv[i+1]
            if pi[i]==i:
                ek[:,ks:ke,:,:]=k[:,qs:qe,:,:]; ev[:,ks:ke,:,:]=v[:,qs:qe,:,:]
            else:
                parts_k,parts_v=[],[]
                rem=pl[i]; p=pi[i]
                while rem>0:
                    ps,pe=cu_q[p],cu_q[p+1]; take=min(rem,pe-ps)
                    parts_k.append(k[:,ps:ps+take,:,:]); parts_v.append(v[:,ps:ps+take,:,:])
                    rem-=take
                    if p==pi[p]: break; p=pi[p]
                pk=torch.cat(parts_k,dim=1) if parts_k else torch.zeros(1,0,H_KV,HD,device=DEVICE,dtype=DTYPE)
                pv=torch.cat(parts_v,dim=1) if parts_v else torch.zeros_like(pk)
                ek[:,ks:ke,:,:]=torch.cat([pk,k[:,qs:qe,:,:]],dim=1)
                ev[:,ks:ke,:,:]=torch.cat([pv,v[:,qs:qe,:,:]],dim=1)
        t["build_kv"]=time.perf_counter()-t0
        t0=time.perf_counter()
        outputs=[]
        for i in range(B):
            qs,qe=cu_q[i],cu_q[i+1]; ks,ke=cu_kv[i],cu_kv[i+1]
            qi_=q[:,qs:qe,:,:]; ki_=ek[:,ks:ke,:,:]; vi_=ev[:,ks:ke,:,:]
            Q=qi_.reshape(-1,H_Q,HD); K=ki_.reshape(-1,H_KV,HD); V=vi_.reshape(-1,H_KV,HD)
            cq=torch.tensor([0,Q.shape[0]],device=DEVICE,dtype=torch.int32)
            ck=torch.tensor([0,K.shape[0]],device=DEVICE,dtype=torch.int32)
            o=flash_attn_varlen_func(Q,K,V,cq,ck,Q.shape[0],K.shape[0],0.0,causal=True)
            outputs.append(o.view(1,-1,H_Q,HD))
        out=torch.cat(outputs,dim=1)
        t["attn_fwd"]=time.perf_counter()-t0
        t["module"]=t.get("build_kv",0)+t.get("attn_fwd",0)
        return out,t

    elif mode=="ps_on_dedup_flex":
        tree=derive_tree(plan); T=tree["T"]
        ptr,op,anc,pl=tree["ptr"].cuda(),tree["op"].cuda(),tree["anc"].cuda(),tree["pl"].cuda()
        def mm(b,h,qi,ki):
            qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
            return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
        t0=time.perf_counter()
        bm=create_block_mask(mm,None,None,T,T,BLOCK_SIZE=128,device=DEVICE)
        torch.cuda.synchronize(); t["bm_build"]=time.perf_counter()-t0
        t0=time.perf_counter()
        out=flex_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),block_mask=bm,enable_gqa=True)
        torch.cuda.synchronize(); t["attn_fwd"]=time.perf_counter()-t0
        t["module"]=t.get("bm_build",0)+t.get("attn_fwd",0)
        return out.permute(0,2,1,3),t

def bench_mode(name, input_ids, mode):
    config=PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
    plan=PrefixSharingPlanner(config).plan(input_ids)
    T_orig=sum(plan.original_lengths); Td=plan.cu_seqlens_q[-1]; Texp=sum(plan.expanded_lengths_kv)

    # Create QKV once
    torch.manual_seed(SEED)
    if mode=="ps_off_fa":
        q=torch.randn(1,T_orig,H_Q,HD,dtype=DTYPE,device=DEVICE)
        k=torch.randn(1,T_orig,H_KV,HD,dtype=DTYPE,device=DEVICE)
        v=torch.randn(1,T_orig,H_KV,HD,dtype=DTYPE,device=DEVICE)
        kv_tokens=T_orig
    else:
        q=torch.randn(1,Td,H_Q,HD,dtype=DTYPE,device=DEVICE)
        k=torch.randn(1,Td,H_KV,HD,dtype=DTYPE,device=DEVICE)
        v=torch.randn(1,Td,H_KV,HD,dtype=DTYPE,device=DEVICE)
        kv_tokens=Td if mode=="ps_on_dedup_flex" else Texp

    # Use no_grad for perf (no backward for pure speed test)
    # Warmup
    for _ in range(WU):
        if mode in ("ps_off_fa","ps_on_expanded_fa"):
            out,_=run_mode(mode,q,k,v,plan)
        else:
            out,_=run_mode(mode,q,k,v,plan)
        del out
    torch.cuda.synchronize()

    # Timing iterations
    fwd_times=[]
    for _ in range(NI):
        if mode in ("ps_off_fa","ps_on_expanded_fa"):
            t0=time.perf_counter(); out,_=run_mode(mode,q,k,v,plan); torch.cuda.synchronize()
            fwd_times.append(time.perf_counter()-t0)
        else:
            torch.cuda.reset_peak_memory_stats()
            t0=time.perf_counter(); out,_=run_mode(mode,q,k,v,plan); torch.cuda.synchronize()
            fwd_times.append(time.perf_counter()-t0)
        del out

    # Peak HBM (from last iteration, which had reset_peak)
    pa=torch.cuda.max_memory_allocated()
    pr=torch.cuda.max_memory_reserved()

    sft=sorted(fwd_times)
    f50=sft[len(sft)//2]; f90=sft[int(len(sft)*0.9)]

    del q,k,v; torch.cuda.empty_cache()
    return {
        "mode":mode,"orig":T_orig,"dedup":Td,"expanded":Texp,"kv_tokens":kv_tokens,
        "fwd_p50_ms":round(f50*1000,3),"fwd_p90_ms":round(f90*1000,3),
        "peak_alloc_mb":round(pa/1024/1024,1),"peak_res_mb":round(pr/1024/1024,1),
    }

def main():
    print("="*60,file=sys.stderr); print("PoC-2C: 三路径attention module速度+HBM",file=sys.stderr); print("="*60,file=sys.stderr)
    print(f"GPU:{torch.cuda.get_device_name(0)}",file=sys.stderr)

    all_r=[]
    for name,(ids,desc) in WORKLOADS.items():
        print(f"\n--- {name}: {desc} ---",file=sys.stderr)
        for mode in ["ps_off_fa","ps_on_expanded_fa","ps_on_dedup_flex"]:
            try:
                r=bench_mode(name,ids,mode); all_r.append(r)
                print(f"  {mode}: fwd={r['fwd_p50_ms']:.1f}ms peak={r['peak_alloc_mb']:.0f}MB kv_tokens={r['kv_tokens']}",file=sys.stderr)
            except Exception as e:
                traceback.print_exc(file=sys.stderr); all_r.append({"case":name,"mode":mode,"error":str(e)})

    print("\n=== SUMMARY ===",file=sys.stderr)
    for r in all_r:
        print(f"  {r.get('case',r.get('mode','?'))}/{r.get('mode','?')}: fwd={r.get('fwd_p50_ms','?'):.1f}ms peak={r.get('peak_alloc_mb','?'):.0f}MB",file=sys.stderr)

    print("\n"+json.dumps(all_r,indent=2))

if __name__=="__main__":
    main()
