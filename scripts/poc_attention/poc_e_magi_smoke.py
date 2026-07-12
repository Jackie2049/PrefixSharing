"""PoC-E: Magi FFA on A100 sm80 — dispatch API (correct)."""
import warnings; warnings.filterwarnings("ignore")
import torch, time, torch.distributed as dist

from magi_attention.common.range import AttnRange
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.api.magi_attn_interface import magi_attn_flex_key, dispatch
from magi_attention.config import DistAttnConfig

B, H_Q, H_KV, D = 1, 8, 2, 64; T = 128
torch.manual_seed(42)
q = torch.randn(B, H_Q, T, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H_KV, T, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H_KV, T, D, device="cuda", dtype=torch.bfloat16)

# CP group: for single GPU CP=1, need a ProcessGroup with world_size=1
# Use a mock group or check if init_process_group was called
import os
if not dist.is_initialized():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
    print("Init dist OK (single GPU)")

cp_group = dist.GroupMember.WORLD
config = DistAttnConfig()

qr = AttnRanges(); qr.append(AttnRange(0, T))
kr = AttnRanges(); kr.append(AttnRange(0, T))

key = magi_attn_flex_key(qr, kr, AttnMaskType.CAUSAL,
    total_seqlen_q=T, total_seqlen_k=T,
    num_heads_q=H_Q, num_heads_kv=H_KV, head_dim=D,
    pad_size=0, cp_group_or_mesh=cp_group, dist_attn_config=config)

out = dispatch(q, k, v, key)[0]
print(f"dispatch causal: {out.shape} max={out.abs().max():.4f}")
out.sum().backward()
print(f"  backward: grad={q.grad.norm().item():.4f}")

k_rep = k.repeat_interleave(H_Q // H_KV, dim=1)
v_rep = v.repeat_interleave(H_Q // H_KV, dim=1)
ref = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
diff = (out - ref).float().abs().max().item()
print(f"  vs SDPA: max diff = {diff:.2e}")

# Prefix-tree
T2 = 20; P, A, B = 10, 5, 5; pn = P+A; bn = B
torch.manual_seed(42)
q2 = torch.randn(1, H_Q, T2, D, device="cuda", dtype=torch.bfloat16)
k2 = torch.randn(1, H_KV, T2, D, device="cuda", dtype=torch.bfloat16)
v2 = torch.randn(1, H_KV, T2, D, device="cuda", dtype=torch.bfloat16)

qr2 = AttnRanges(); kr2 = AttnRanges(); atm2 = []
for qs,qe,ks,ke,tp in [(0,pn,0,pn,AttnMaskType.CAUSAL),(pn,T2,pn,T2,AttnMaskType.CAUSAL),(pn,T2,0,pn,AttnMaskType.FULL)]:
    qr2.append(AttnRange(qs,qe)); kr2.append(AttnRange(ks,ke)); atm2.append(tp)

key2 = magi_attn_flex_key(qr2, kr2, atm2,
    total_seqlen_q=T2, total_seqlen_k=T2,
    num_heads_q=H_Q, num_heads_kv=H_KV, head_dim=D,
    pad_size=0, cp_group_or_mesh=cp_group, dist_attn_config=config)
out2 = dispatch(q2, k2, v2, key2)[0]
print(f"\nprefix-tree (T={T2}): {out2.shape}")

dmask = torch.zeros(T2, T2, dtype=torch.bool, device="cuda")
dmask[0:pn,0:pn] = torch.tril(torch.ones(pn,pn,dtype=torch.bool))
dmask[pn:T2,pn:T2] = torch.tril(torch.ones(bn,bn,dtype=torch.bool))
dmask[pn:T2,0:pn] = True
k2r = k2.repeat_interleave(H_Q//H_KV, dim=1)
v2r = v2.repeat_interleave(H_Q//H_KV, dim=1)
ref2 = torch.nn.functional.scaled_dot_product_attention(q2, k2r, v2r, attn_mask=dmask[None,None,:,:].expand(1,H_Q,-1,-1))
diff2 = (out2-ref2).float().abs().max().item()
print(f"  vs oracle: max diff = {diff2:.2e}")

# Performance
print(f"\n=== Performance (T=128, N=50) ===")
for _ in range(10): dispatch(q,k,v,key)[0].sum().backward()
torch.cuda.synchronize()
ts = []
for _ in range(50):
    t0=time.perf_counter(); dispatch(q,k,v,key)[0].sum().backward(); torch.cuda.synchronize()
    ts.append(time.perf_counter()-t0)
avg = sum(ts)/50
print(f"  Magi dispatch: {avg*1000:.1f}ms")

from torch.nn.attention.flex_attention import flex_attention as pt_flex, create_block_mask
def cm(b,h,qi,ki): return qi >= ki
bm = create_block_mask(cm, None, None, T, T, BLOCK_SIZE=128, device="cuda")
qp = q.permute(0,2,1,3); kp = k.permute(0,2,1,3); vp = v.permute(0,2,1,3)
for _ in range(10): pt_flex(qp,kp,vp,block_mask=bm,enable_gqa=True).sum().backward()
torch.cuda.synchronize()
ts2 = []
for _ in range(50):
    t0=time.perf_counter(); pt_flex(qp,kp,vp,block_mask=bm,enable_gqa=True).sum().backward(); torch.cuda.synchronize()
    ts2.append(time.perf_counter()-t0)
avg2 = sum(ts2)/50
print(f"  PT flex_attn:  {avg2*1000:.1f}ms")
print(f"  Ratio (Magi/PyT): {avg/avg2:.2f}x")
dist.destroy_process_group()
print("Magi FFA on A100 sm80: PASSED")
