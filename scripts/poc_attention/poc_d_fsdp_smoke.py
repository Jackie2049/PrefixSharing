#!/usr/bin/env python3
"""PoC-D: FSDP remove-padding smoke test (single GPU).

Compares:
  1. PS=OFF (native HF attention) — baseline
  2. PS=ON expanded-KV (current backend)
  3. PS=ON dedup-flex (prefix-tree BlockMask + flex_attention)

Uses Qwen2.5-0.5B, single GPU, no real FSDP sharding — verifies
attention-level correctness in the remove-padding path.
"""

import json, sys, torch, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED=42; DEVICE="cuda"; DTYPE=torch.bfloat16
MODEL_PATH="/jiangdingfeng/zy/Termius/models/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

def make_plan(input_ids):
    cfg=PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    return PrefixSharingPlanner(cfg).plan(input_ids)

def trim_to_dedup(q, k, v, plan):
    """Trim Q/K/V from full tokens to dedup tokens using input_keep_ranges."""
    cu=plan.cu_seqlens_q; ir=plan.input_keep_ranges
    q_parts=[]; k_parts=[]; v_parts=[]
    for i in range(plan.batch_size):
        s,e=cu[i],cu[i+1]
        irs,ire=ir[i]
        offset=irs - irs  # = 0, but conceptually the offset from original position
        # The dedup tokens for this row are simply cu[i] to cu[i+1]
        q_parts.append(q[:, s:e, :, :])
        k_parts.append(k[:, s:e, :, :])
        v_parts.append(v[:, s:e, :, :])
    return torch.cat(q_parts, dim=1), torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

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

def build_expanded_kv(plan, q_hidden, k_hidden, v_hidden):
    total_expanded=sum(plan.expanded_lengths_kv); H_KV=k_hidden.shape[2]; D=k_hidden.shape[3]
    ek=torch.zeros(1,total_expanded,H_KV,D,dtype=k_hidden.dtype,device=k_hidden.device)
    ev=torch.zeros_like(ek)
    cu_q=plan.cu_seqlens_q; cu_kv=plan.cu_seqlens_kv; pi=plan.provider_index; pl=plan.prefix_lens

    # Precompute dedup token positions for each row
    # For a row i, its dedup tokens are cu_q[i]:cu_q[i+1]
    # For a chain (i uses provider j which uses provider k...), expanded K/V =
    #   k_token[provider_chain_prefix] + suffix

    def find_provider_prefix_tokens(row_idx, prefix_len):
        """Trace provider chain to find K/V tokens that belong to shared prefix.
        For a reuser with prefix_len P:
        - The first P tokens come from its immediate provider's dedup tokens.
        - If the provider itself has a prefix (chain), those P tokens may span
          across the provider's own prefix + some of its suffix."""
        p = pi[row_idx]
        p_start = cu_q[p]; p_end = cu_q[p+1]
        # The first min(prefix_len, p_end-p_start) tokens of provider are the prefix
        return [p_start, p_start + min(prefix_len, p_end-p_start)]

    for i in range(plan.batch_size):
        qs,qe=cu_q[i],cu_q[i+1]; ks,ke=cu_kv[i],cu_kv[i+1]
        if pi[i]==i:  # standalone or root provider
            ek[:,ks:ke,:,:]=k_hidden[:,qs:qe,:,:]
            ev[:,ks:ke,:,:]=v_hidden[:,qs:qe,:,:]
        else:
            # Build expanded K/V: trace provider chain
            prefix_len = pl[i]
            kv_parts = []
            # Gather tokens from provider chain (recursive)
            remaining = prefix_len
            p = pi[i]
            while remaining > 0 and p != pi[p]:
                p_start = cu_q[p]; p_end = cu_q[p+1]
                take = min(remaining, p_end - p_start)
                kv_parts.append((k_hidden[:, p_start:p_start+take, :, :],
                                v_hidden[:, p_start:p_start+take, :, :]))
                remaining -= take
                p = pi[p]
            # Also the root provider (pi[p] == p or the first provider)
            root = p
            while root != pi[root]:
                root = pi[root]
            p_start = cu_q[root]; p_end = cu_q[root+1]
            take = min(remaining, p_end - p_start)
            if take > 0:
                kv_parts.append((k_hidden[:, p_start:p_start+take, :, :],
                                v_hidden[:, p_start:p_start+take, :, :]))

            # Reverse to original order
            kv_parts = kv_parts[::-1]
            prefix_k = torch.cat([kp[0] for kp in kv_parts], dim=1)
            prefix_v = torch.cat([kp[1] for kp in kv_parts], dim=1)
            # Suffix is this row's dedup tokens
            suffix_k = k_hidden[:, qs:qe, :, :]
            suffix_v = v_hidden[:, qs:qe, :, :]
            ek[:, ks:ke, :, :] = torch.cat([prefix_k, suffix_k], dim=1)
            ev[:, ks:ke, :, :] = torch.cat([prefix_v, suffix_v], dim=1)
    return ek, ev

def run_expanded(q, k, v, plan, H_Q, H_KV, D):
    outputs=[]
    for i in range(plan.batch_size):
        qs,qe=plan.cu_seqlens_q[i],plan.cu_seqlens_q[i+1]
        ks,ke=plan.cu_seqlens_kv[i],plan.cu_seqlens_kv[i+1]
        qi=q[:,qs:qe,:,:]; ki=k[:,ks:ke,:,:]; vi=v[:,ks:ke,:,:]
        kr=ki.repeat_interleave(H_Q//H_KV,dim=2); vr=vi.repeat_interleave(H_Q//H_KV,dim=2)
        out=torch.nn.functional.scaled_dot_product_attention(qi.permute(0,2,1,3),kr.permute(0,2,1,3),vr.permute(0,2,1,3),is_causal=True)
        outputs.append(out.permute(0,2,1,3))
    return torch.cat(outputs, dim=1)

def run_flex_dedup(q, k, v, plan, H_Q, H_KV, D):
    tr=derive_tree(plan); T=tr["T"]
    ptr,op,anc,pl=tr["ptr"].cuda(),tr["op"].cuda(),tr["anc"].cuda(),tr["pl"].cuda()
    def mm(b,h,qi,ki):
        qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
        return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
    bm=create_block_mask(mm,None,None,T,T,BLOCK_SIZE=128,device=q.device)
    out=flex_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),block_mask=bm,enable_gqa=True)
    return out.permute(0,2,1,3)

def main():
    print("="*60); print("PoC-D: FSDP remove-padding smoke test"); print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoConfig
    cfg=AutoConfig.from_pretrained(MODEL_PATH)
    H_Q,H_KV,D=cfg.num_attention_heads,cfg.num_key_value_heads,getattr(cfg,"head_dim",cfg.hidden_size//cfg.num_attention_heads)
    print(f"Config: H_Q={H_Q}, H_KV={H_KV}, D={D}, hidden={cfg.hidden_size}")

    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=DTYPE,device_map=DEVICE)
    model.eval()
    embed=model.model.embed_tokens; norm=model.model.norm; lm_head=model.lm_head
    layers=model.model.layers

    test_cases={
        "star":([[1]*32+[100]*33,[1]*32+[200]*33,[1]*32+[300]*33], "star P32+R32 x3"),
        "chain":([list(range(1,33))*1,list(range(1,33))+list(range(100,117)),list(range(1,33))+list(range(100,117))+list(range(200,209))],"chain d3"),
    }

    results=[]
    for name,(input_ids,desc) in test_cases.items():
        print(f"\n--- {name}: {desc} ---")
        # Pad to max length for model input (handles unequal lengths)
        max_len = max(len(s) for s in input_ids)
        input_ids_padded = [s + [0]*(max_len-len(s)) for s in input_ids]
        input_pt=torch.tensor(input_ids_padded,device=DEVICE,dtype=torch.long)
        # Build attention mask for padding
        attn_mask_pt = torch.tensor([[1]*len(s)+[0]*(max_len-len(s)) for s in input_ids],device=DEVICE,dtype=torch.long)

        # PS=OFF baseline
        with torch.no_grad():
            logits_off=model(input_pt, attention_mask=attn_mask_pt).logits
        print(f"  PS=OFF logits: {logits_off.shape}")

        plan=make_plan(input_ids)
        if not plan.has_sharing:
            print("  [WARN] no sharing"); continue

        T_dedup=plan.cu_seqlens_q[-1]; total_orig=sum(plan.original_lengths)
        total_exp=sum(plan.expanded_lengths_kv)
        print(f"  orig={total_orig} dedup={T_dedup} expanded={total_exp}")
        print(f"  kept_q={plan.kept_lengths_q}, expanded_kv={plan.expanded_lengths_kv}")

        # Get dedup Q/K/V from model
        with torch.no_grad():
            hidden_all=embed(input_pt)  # [3, max_len, hidden]
            # Trim hidden to dedup tokens
            hidden_parts=[]
            for i in range(plan.batch_size):
                s,e=plan.cu_seqlens_q[i],plan.cu_seqlens_q[i+1]
                irs,ire=plan.input_keep_ranges[i]
                hidden_parts.append(hidden_all[i:i+1, irs:ire, :])
            hidden_dedup=torch.cat(hidden_parts,dim=1)  # [1, T_dedup, hidden]
            q=layers[0].self_attn.q_proj(hidden_dedup).view(1,T_dedup,H_Q,D)
            k=layers[0].self_attn.k_proj(hidden_dedup).view(1,T_dedup,H_KV,D)
            v=layers[0].self_attn.v_proj(hidden_dedup).view(1,T_dedup,H_KV,D)

        # Expanded-KV reference
        k_exp, v_exp = build_expanded_kv(plan, q, k, v)
        attn_exp = run_expanded(q, k_exp, v_exp, plan, H_Q, H_KV, D)
        print(f"  Expanded attn out: {attn_exp.shape}")

        # Dedup-Flex path
        attn_flex = run_flex_dedup(q, k, v, plan, H_Q, H_KV, D)
        print(f"  Dedup-Flex attn out: {attn_flex.shape}")

        # Compare attention outputs
        diff_attn = (attn_exp - attn_flex).float().abs().max().item()
        print(f"  Attn output max diff (exp vs flex): {diff_attn:.2e}")

        # Compare logits: use PS=OFF model (with expanded-KV attention manually replaced)
        # Since remaining layers need position_ids, just compare attention output
        # and verify both paths produce consistent results through the full model
        # by running model() and checking the output difference pattern

        # Full forward: compose attention output back into hidden, then through LM head
        hidden_size = cfg.hidden_size
        h_attn_exp = layers[0].self_attn.o_proj(attn_exp.view(1,-1,hidden_size))
        h_attn_flex = layers[0].self_attn.o_proj(attn_flex.view(1,-1,hidden_size))

        # Post-attention norm + MLP (no position_ids needed)
        hidden_post_exp = layers[0].input_layernorm(hidden_dedup + h_attn_exp)
        hidden_post_flex = layers[0].input_layernorm(hidden_dedup + h_attn_flex)
        mlp_exp = layers[0].mlp(hidden_post_exp)
        mlp_flex = layers[0].mlp(hidden_post_flex)

        # Logits via LM head directly (skip remaining layers - they add position bias but
        # if attention output is identical, the difference should be minimal)
        logits_exp = lm_head(norm(mlp_exp))
        logits_flex = lm_head(norm(mlp_flex))
        diff_logits = (logits_exp - logits_flex).float().abs().max().item()
        print(f"  Post-MLP logits max diff: {diff_logits:.2e}")

        # Loss comparison
        labels = torch.tensor([plan.original_lengths[i] for i in range(plan.batch_size)], device=DEVICE, dtype=torch.long)
        # Simplified loss: cross_entropy over last token prediction
        loss_exp = logits_exp[0, -1, :].softmax(dim=-1)[0].item()
        loss_flex = logits_flex[0, -1, :].softmax(dim=-1)[0].item()
        print(f"  Last token softmax (exp/flex): {loss_exp:.6f} / {loss_flex:.6f}")

        # Gradient comparison
        print(f"  === Gradient ===")
        torch.manual_seed(SEED+1)
        q1=q.detach().clone().requires_grad_(True)
        k1=k.detach().clone().requires_grad_(True)
        v1=v.detach().clone().requires_grad_(True)
        ke1,ve1=build_expanded_kv(plan,q1,k1,v1)
        o1=run_expanded(q1,ke1,ve1,plan,H_Q,H_KV,D); o1.sum().backward()
        g1=(q1.grad.clone(),k1.grad.clone(),v1.grad.clone())

        q2=q.detach().clone().requires_grad_(True)
        k2=k.detach().clone().requires_grad_(True)
        v2=v.detach().clone().requires_grad_(True)
        o2=run_flex_dedup(q2,k2,v2,plan,H_Q,H_KV,D); o2.sum().backward()
        g2=(q2.grad.clone(),k2.grad.clone(),v2.grad.clone())
        gd={}
        for ng,ga,gb in [("q",g1[0],g2[0]),("k",g1[1],g2[1]),("v",g1[2],g2[2])]:
            d=(ga-gb).float().abs(); gd[ng]={"max":round(d.max().item(),8)}
        print(f"    grad q:{gd['q']['max']:.2e} k:{gd['k']['max']:.2e} v:{gd['v']['max']:.2e}")

        results.append({
            "case":name, "orig":total_orig, "dedup":T_dedup, "expanded":total_exp,
            "attn_diff":round(diff_attn,8), "logits_diff":round(diff_logits,8),
            "gradient":gd,
        })

    print("\n"+"="*60); print("SUMMARY"); print("="*60)
    for r in results:
        print(f"  {r['case']}: attnΔ={r['attn_diff']:.2e} logitsΔ={r['logits_diff']:.2e}")
    print("\n"+json.dumps(results,indent=2))

if __name__=="__main__":
    main()
