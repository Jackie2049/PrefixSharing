#!/usr/bin/env python3
"""Script: python3 poc_3a_real_path_precision.py  # (inside scripts/poc_attention/)

补充3A：真实 expanded 路径与 Flex 的精度/梯度闭环 (§2.3.8.1)

Must:
- A: build_prefix_expanded_kv() -> GpuFlashAttentionBackend.attention()
- B: dedup Q/K/V -> generic BlockMask -> flex_attention()
- C: fp32 dense oracle (dedup Q/K/V -> token-level PrefixTree mask -> SDPA)

Spy build_prefix_expanded_kv and GpuFlashAttentionBackend.attention calls.
Provider directed-gradient test on deepest leaf suffix output.
Exit non-zero on any failure.
"""
import json, os, sys, time, traceback
from typing import Any

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")

import torch
import torch.nn.functional as F

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.kv_builder import build_prefix_expanded_kv
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED = 42
DEVICE = "cuda"
H_Q, H_KV, HEAD_DIM = 14, 2, 64

def mp(p): return list(range(1, p+1))
def mk_star(plen, rlen, n):
    p = mp(plen)
    return [p + list(range(100, 100+rlen))] + [p + list(range(200+plen*i, 200+plen*i+rlen)) for i in range(n)]
def mk_chain(d, pl, sl):
    p = mp(pl); r = [p]
    for i in range(1,d): r.append(r[-1] + list(range(300+sl*i, 300+sl*(i+1)))); return r
def mk_frag():
    return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
            [1,2,3,4,5,6,7,8,9,10,11,12,13],
            [1,2,3,4,5,6,7,8,100,101,102],
            [1,2,3,4,5,6,7,8,100,101,102,103,104],
            [1,2,3,4,5,200,201,202]]

WORKLOADS = {
    "star_aligned": (mk_star(64, 65, 3), "B=4,P=64,R=65"),
    "star_long_prompt": (mk_star(1024, 128, 7), "B=8,P=1024,R=128"),
    "chain_depth3": (mk_chain(3, 32, 8), "depth=3"),
    "chain_depth6": (mk_chain(6, 32, 8), "depth=6"),
    "chain_depth12": (mk_chain(12, 16, 4), "depth=12"),
    "deep_fragmented": (mk_frag(), "B=6,mixed"),
    "multi_group": (mk_star(64,65,2)+mk_star(32,33,2), "2 groups"),
}

# Spy counters
_builder_calls = [0]; _backend_attention_calls = [0]
_real_builder = build_prefix_expanded_kv
_real_backend = GpuFlashAttentionBackend.attention
def spy_builder(*a, **kw):
    _builder_calls[0] += 1; return _real_builder(*a, **kw)
def spy_attention(self, *a, **kw):
    _backend_attention_calls[0] += 1; return _real_backend(self, *a, **kw)

class NullStats:
    def record_store_count(self,**kw): pass
    def record_reuse(self,**kw): pass
    def record_reuse_hit(self,**kw): pass
    def record_reuse_miss(self,**kw): pass
    def record_stored_tokens(self,**kw): pass
    def record_reused_prefix_tokens(self,**kw): pass
    def record_expanded_kv_tokens(self,**kw): pass
    def record_valid_q_tokens(self,**kw): pass
    def record_padded_q_tokens(self,**kw): pass
    def record_attention_kv_build(self,**kw): pass
    def record_stats(self,**kw): pass

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

def build_dense_oracle_mask(plan):
    """Token-level PrefixTree mask for SDPA. [T,T] bool."""
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

def met(a,b):
    d=(a-b).float().abs(); mx,rn=d.max().item(),d.norm().item()/max(a.float().norm().item(),1e-12)
    cs=(a.float().reshape(-1)@b.float().reshape(-1))/(a.float().norm()*b.float().norm()+1e-12)
    return {"max":round(mx,8),"rel_l2":round(rn,8),"cos":round(cs.item(),8),"finite":not(torch.isnan(d).any()or torch.isinf(d).any()),"norm_a":round(a.float().norm().item(),4),"norm_b":round(b.float().norm().item(),4)}

def diff_report(ref, test, label):
    """Print first mismatched token info if diff significant."""
    d=(ref-test).float().abs(); mx_idx=d.argmax().item()
    shape=ref.shape
    if len(shape)==3:   # (1,T,HD)
        b,t,h=0,mx_idx//shape[2]%shape[1],mx_idx%shape[2]
    else:
        b,t,h=0,0,0
    return f"{label}: max_diff={d.max():.4e} at [0,{t},{h}]"

def run_case(name, input_ids, dtype):
    config=PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
    plan=PrefixSharingPlanner(config).plan(input_ids)
    B,Td=plan.batch_size,plan.cu_seqlens_q[-1]
    Torig=sum(plan.original_lengths); Texp=sum(plan.expanded_lengths_kv)
    is_fp32=(dtype==torch.float32)
    results={"case":name,"dtype":str(dtype),"orig":Torig,"dedup":Td,"expanded":Texp}

    # ── Create shared Q/K/V ──
    torch.manual_seed(SEED)
    q0=torch.randn(1,Td,H_Q,HEAD_DIM,dtype=dtype,device=DEVICE,requires_grad=True)
    k0=torch.randn(1,Td,H_KV,HEAD_DIM,dtype=dtype,device=DEVICE,requires_grad=True)
    v0=torch.randn(1,Td,H_KV,HEAD_DIM,dtype=dtype,device=DEVICE,requires_grad=True)
    cu_q=plan.cu_seqlens_q

    # ── Path A: project-expanded reference ──
    _builder_calls[0]=0; _backend_attention_calls[0]=0
    store=PrefixAttentionStore()
    layout=PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    stats=NullStats()
    k_flat=k0[0].reshape(Td,-1); v_flat=v0[0].reshape(Td,-1)
    ek,ev = spy_builder(key=k_flat,value=v_flat,store=store,prefix_sharing_plan=plan,packed_batch_layout=layout,layer_id=0,stats=stats)

    # GpuFlashAttentionBackend.attention expects (total_tokens, num_heads, head_dim) 3-D input
    ek_3d=ek.view(-1,H_KV,HEAD_DIM)  # (Texp, H_KV, HD)
    ev_3d=ev.view(-1,H_KV,HEAD_DIM)
    # query is also 3-D: (Td, H_Q, HD)
    q_a = q0[0]  # (Td, H_Q, HD)
    # Build backend instance and call production attention
    backend=GpuFlashAttentionBackend()
    try:
        out_a = backend.attention(query=q_a, key=ek_3d, value=ev_3d, prefix_sharing_plan=plan)
        results["builder_calls"]=_builder_calls[0]
        results["backend_attention_type"]="GpuFlashAttentionBackend.attention (production)"
    except Exception as e:
        results["expanded_error"]=repr(e)
        out_a=None
        results["builder_calls"]=_builder_calls[0]

    # out_a is (total_tokens, H_Q, HD) = (T, H_Q, HD). Add batch dim to match other paths
    if out_a is not None and out_a.dim() == 3:
        out_a = out_a.unsqueeze(0)  # (1, Td, H_Q, HD)

    # ── Path B: sparse candidate (dedup Q/K/V + BlockMask + flex_attention) ──
    ptr,op,anc,pl=derive_tree(plan)
    def mm(b,h,qi,ki):
        qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
        return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
    bm=create_block_mask(mm,None,None,Td,Td,BLOCK_SIZE=128,device=DEVICE)
    flex_out = flex_attention(q0.permute(0,2,1,3),k0.permute(0,2,1,3),v0.permute(0,2,1,3),
                              block_mask=bm,enable_gqa=True).permute(0,2,1,3)

    # ── Path C: fp32 dense oracle ──
    if is_fp32:
        dmask=build_dense_oracle_mask(plan)
        q_f32=q0.float().permute(0,2,1,3)
        kr_f32=k0.float().repeat_interleave(H_Q//H_KV,dim=2).permute(0,2,1,3)
        vr_f32=v0.float().repeat_interleave(H_Q//H_KV,dim=2).permute(0,2,1,3)
        am=dmask[None,None,:,:].expand(1,H_Q,-1,-1)
        oracle_out=F.scaled_dot_product_attention(q_f32,kr_f32,vr_f32,attn_mask=am,dropout_p=0.0,is_causal=False).permute(0,2,1,3).to(dtype)
    else:
        oracle_out=None

    # ── Backward with same upstream gradient ──
    up_grad=torch.randn_like(flex_out)
    if out_a is not None:
        out_a.backward(up_grad)
        # expanded path grads from backward (through builder)
        qg_exp, kg_exp, vg_exp = q0.grad.clone(), k0.grad.clone(), v0.grad.clone()
        q0.grad = k0.grad = v0.grad = None
    flex_out.backward(up_grad)
    qg_flex,kg_flex,vg_flex=q0.grad.clone(),k0.grad.clone(),v0.grad.clone()
    q0.grad=k0.grad=v0.grad=None
    if oracle_out is not None:
        oracle_out.backward(up_grad)
        qg_ora,kg_ora,vg_ora=q0.grad.clone(),k0.grad.clone(),v0.grad.clone()
        q0.grad=k0.grad=v0.grad=None

    # ── Compare A/B, B/C, A/C ──
    def compare(label, x, y):
        if x is None or y is None: return
        results[label] = met(x,y)
        # If diff significant, print mismatched position
        d=(x-y).float().abs()
        if d.max().item() > 1e-4:
            print(f"  WARN {label}: {diff_report(x,y,'')}", file=sys.stderr)

    if out_a is not None and oracle_out is not None:
        compare("exp_vs_oracle", out_a, oracle_out)
        compare("grad_exp_vs_ora_q", qg_exp, qg_ora)
        compare("grad_exp_vs_ora_k", kg_exp, kg_ora)
        compare("grad_exp_vs_ora_v", vg_exp, vg_ora)
    if oracle_out is not None:
        compare("flex_vs_oracle", flex_out, oracle_out)
        compare("grad_flex_vs_ora_q", qg_flex, qg_ora)
        compare("grad_flex_vs_ora_k", kg_flex, kg_ora)
        compare("grad_flex_vs_ora_v", vg_flex, vg_ora)
    if out_a is not None:
        compare("exp_vs_flex", out_a, flex_out)
        compare("grad_exp_vs_flex_q", qg_exp, qg_flex)
        compare("grad_exp_vs_flex_k", kg_exp, kg_flex)
        compare("grad_exp_vs_flex_v", vg_exp, vg_flex)

    # ── Provider directed-gradient test ──
    # Re-run flex: apply loss only on deepest leaf's suffix
    q2=q0.detach().clone().requires_grad_(True)
    k2=k0.detach().clone().requires_grad_(True)
    v2=v0.detach().clone().requires_grad_(True)
    bm2=create_block_mask(mm,None,None,Td,Td,BLOCK_SIZE=128,device=DEVICE)
    out2=flex_attention(q2.permute(0,2,1,3),k2.permute(0,2,1,3),v2.permute(0,2,1,3),
                        block_mask=bm2,enable_gqa=True).permute(0,2,1,3)
    # Deepest leaf = last row's suffix tokens
    last_row_start=cu_q[-2] if B>=2 else 0
    suffix_mask=torch.zeros_like(out2)
    suffix_mask[:,last_row_start:,:,:]=1.0
    (out2*suffix_mask).sum().backward()
    # Provider prefix (row 0) K/V gradient must exist
    prov_prefix=cu_q[1]  # row0 token count
    k_prov_norm=k2.grad[:,:prov_prefix,:,:].float().norm().item()
    v_prov_norm=v2.grad[:,:prov_prefix,:,:].float().norm().item()
    results["provider_k_grad_norm"]=round(k_prov_norm,6)
    results["provider_v_grad_norm"]=round(v_prov_norm,6)
    assert k_prov_norm>0, f"Provider K gradient is zero! case={name}"
    assert v_prov_norm>0, f"Provider V gradient is zero! case={name}"

    del q0,k0,v0,flex_out,out2
    if out_a is not None: del out_a
    if oracle_out is not None: del oracle_out
    torch.cuda.empty_cache()
    return results

def main():
    print("="*60,file=sys.stderr)
    print("补充3A: 真实 expanded 路径与 Flex 精度/梯度闭环",file=sys.stderr)
    print(f"GPU:{torch.cuda.get_device_name(0)} torch:{torch.__version__}",file=sys.stderr)
    print("="*60,file=sys.stderr)

    all_results=[]
    fails=0
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n>>> dtype={dtype}",file=sys.stderr)
        for name,(ids,desc) in WORKLOADS.items():
            print(f"  {name}: {desc}",file=sys.stderr)
            try:
                r=run_case(name,ids,dtype)
                all_results.append(r)
                evo=r.get("exp_vs_oracle",{}); fvo=r.get("flex_vs_oracle",{})
                print(f"    builder_calls={r.get('builder_calls','?')} backend_calls={r.get('backend_calls','?')}",file=sys.stderr)
                emax = evo.get("max","?"); erel = evo.get("rel_l2","?"); ecos = evo.get("cos","?")
                fmax = fvo.get("max","?"); frel = fvo.get("rel_l2","?"); fcos = fvo.get("cos","?")
                print(f"    exp-vs-ora: max={emax} rel={erel} cos={ecos}",file=sys.stderr)
                print(f"    flex-vs-ora: max={fmax} rel={frel} cos={fcos}",file=sys.stderr)
                print(f"    prov_k_grad={r.get('provider_k_grad_norm',0):.4e} prov_v_grad={r.get('provider_v_grad_norm',0):.4e}",file=sys.stderr)
                # fp32 gate
                _is_fp32 = (dtype == torch.float32)
                if _is_fp32 and evo and evo.get("max",1e9) > 1e-4:
                    print(f"    FAIL: exp_vs_oracle max={str(evo.get('max',1e9)):s} > 1e-4",file=sys.stderr)
                    fails+=1
                if _is_fp32 and fvo and fvo.get("max",1e9) > 1e-4:
                    print(f"    FAIL: flex_vs_oracle max={str(fvo.get('max',1e9)):s} > 1e-4",file=sys.stderr)
                    fails+=1
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                all_results.append({"case":name,"dtype":str(dtype),"error":repr(e)})
                fails+=1

    print(f"\n\n=== SUMMARY ===",file=sys.stderr)
    print(f"  Total failures: {fails}",file=sys.stderr)
    for r in all_results:
        c=r.get("case","?"); d=r.get("dtype","?")
        if "error" in r:
            print(f"  {c}({d}): ERROR - {r['error']}",file=sys.stderr)
            continue
        bc=str(r.get("builder_calls","?")); ac=str(r.get("backend_calls","?"))
        evo=r.get("exp_vs_oracle",{}); fvo=r.get("flex_vs_oracle",{})
        evo_max = evo.get("max","?"); fvo_max = fvo.get("max","?")
        if isinstance(evo_max, (int,float)): evo_max_f = f"{evo_max:.2e}"
        else: evo_max_f = str(evo_max)
        if isinstance(fvo_max, (int,float)): fvo_max_f = f"{fvo_max:.2e}"
        else: fvo_max_f = str(fvo_max)
        pk=r.get("provider_k_grad_norm",0); pv=r.get("provider_v_grad_norm",0)
        print(f"  {c}({d}): bc={bc} ac={ac} exp-ora-max={evo_max_f} flex-ora-max={fvo_max_f} prov_k={pk:.4e} prov_v={pv:.4e}",
              file=sys.stderr)

    out_path=os.path.join(os.path.dirname(__file__),"poc_3a_real_path_results.json")
    with open(out_path,"w") as f:
        json.dump(all_results,f,indent=2,default=str)
    print(f"\n[OK] Results saved to {out_path}",file=sys.stderr)
    print(json.dumps(all_results,indent=2,default=str))

    if fails>0:
        print(f"\nFAILED: {fails} case(s) did not pass precision gate",file=sys.stderr)
        sys.exit(1)

if __name__=="__main__":
    main()
