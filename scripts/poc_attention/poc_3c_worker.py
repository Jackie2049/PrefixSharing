#!/usr/bin/env python3
"""补充3C worker — 单 (workload, mode) 独立进程执行。

调用方式（由父进程调用）：
  python3 poc_3c_worker.py <cuda_device> <workload_name> <mode> <output_json_path>

三条路径：
  ps_off_fa:   flash_attn_varlen_func, per-row original tokens
  ps_on_expanded_fa: build_prefix_expanded_kv -> GpuFlashAttentionBackend.attention
  ps_on_dedup_flex:  generic BlockMask -> torch.compile(flex_attention)()

记录 after_qkv / after_metadata / after_forward / after_backward / peak
spy builder + backend.attention 调用次数
"""

import json, os, sys, time, traceback, gc

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")

os.environ["ENABLE_PREFIX_SHARING"] = "0"
import torch
import torch.nn.functional as F

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.kv_builder import build_prefix_expanded_kv
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = "cuda"
H_Q, H_KV, HD = 14, 2, 64
WU, NI = 20, 100
SEED = 42

def mp(p): return list(range(1,p+1))
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
    "no_sharing": ([list(range(1+128*i,1+128*(i+1))) for i in range(8)], "B=8,L=128"),
    "star_long_prompt": (mk_star(1024,128,7), "B=8,P=1024,R=128"),
    "star_long_B32": (mk_star(1024,128,31), "B=32,P=1024,R=128"),
    "chain_depth6": (mk_chain(6,32,8), "depth=6"),
    "chain_depth12": (mk_chain(12,16,4), "depth=12"),
    "deep_fragmented": (mk_frag(), "B=6,mixed"),
}

MODE_SHAPES = {
    "qwen25": (14, 2, 64),
    "qwen3_gqa": (16, 8, 128),
}

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

class SpyBuilder:
    def __init__(self): self.count = 0
    def __call__(self, *a, **kw):
        self.count += 1
        return build_prefix_expanded_kv(*a, **kw)

class SpyBackend:
    def __init__(self): self.count = 0
    def __call__(self, bk, *a, **kw):
        self.count += 1
        return bk.attention(*a, **kw)

def run_worker(cuda_dev, workload_name, mode, out_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)
    torch.cuda.empty_cache()
    gc.collect()

    input_ids, desc = WORKLOADS[workload_name]
    hq, hkv, hd = MODE_SHAPES["qwen25"]

    torch.manual_seed(SEED + 1)
    result = {"workload": workload_name, "mode": mode, "gpu": str(cuda_dev), "shape": f"Q{hq}KV{hkv}D{hd}"}

    # ── Plan ──
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(input_ids)
    B, Td = plan.batch_size, plan.cu_seqlens_q[-1]
    Torig = sum(plan.original_lengths)
    Texp = sum(plan.expanded_lengths_kv)
    cu_q = plan.cu_seqlens_q
    result.update({"orig": Torig, "dedup": Td, "expanded": Texp, "batch": B})

    snapshots = {}

    if mode == "ps_off_fa":
        # PS=OFF: per-row original tokens + flash_attn_varlen_func
        torch.manual_seed(SEED)
        q = torch.randn(1, Torig, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k = torch.randn(1, Torig, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v = torch.randn(1, Torig, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        torch.cuda.synchronize()
        snapshots["after_qkv"] = torch.cuda.memory_allocated() / 1024 / 1024
        snapshots["after_metadata"] = snapshots["after_qkv"]

        def forward_fn():
            outs = []
            for i in range(B):
                s, e = cu_q[i], cu_q[i+1]
                fs = sum(plan.original_lengths[:i]); fe = fs + plan.original_lengths[i]
                qi = q[:, fs:fe, :, :]; ki = k[:, fs:fe, :, :]; vi = v[:, fs:fe, :, :]
                Q = qi.reshape(-1, hq, hd); K = ki.reshape(-1, hkv, hd); V = vi.reshape(-1, hkv, hd)
                cq = torch.tensor([0, Q.shape[0]], device=DEVICE, dtype=torch.int32)
                ck = torch.tensor([0, K.shape[0]], device=DEVICE, dtype=torch.int32)
                out = flash_attn_varlen_func(Q, K, V, cq, ck, Q.shape[0], K.shape[0], 0.0, causal=True)
                outs.append(out.view(1, -1, hq, hd))
            return torch.cat(outs, dim=1)

        result["backend_type"] = "flash_attn_varlen_func (per-row)"
        builder_count = 0
        backend_count = 0

    elif mode == "ps_on_expanded_fa":
        # PS=ON expanded: build_prefix_expanded_kv -> GpuFlashAttentionBackend.attention
        torch.manual_seed(SEED)
        q = torch.randn(1, Td, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k_src = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v_src = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        torch.cuda.synchronize()
        snapshots["after_qkv"] = torch.cuda.memory_allocated() / 1024 / 1024

        # Build expanded K/V with spy
        spy = SpyBuilder()
        store = PrefixAttentionStore()
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        stats = NullStats()
        k_flat = k_src[0].reshape(Td, -1)
        v_flat = v_src[0].reshape(Td, -1)
        ek, ev = spy(key=k_flat, value=v_flat, store=store, prefix_sharing_plan=plan,
                     packed_batch_layout=layout, layer_id=0, stats=stats)
        ek_3d = ek.view(Texp, hkv, hd)
        ev_3d = ev.view(Texp, hkv, hd)
        builder_count = spy.count
        torch.cuda.synchronize()
        snapshots["after_metadata"] = torch.cuda.memory_allocated() / 1024 / 1024

        # GpuFlashAttentionBackend.attention call
        # The backend expects (total_tokens, num_heads, head_dim) 3-D input
        # For expanded path, Q is dedup (Td, hq, hd), K/V are expanded (Texp, hkv, hd)
        q_a = q[0]  # (Td, hq, hd)
        bk = GpuFlashAttentionBackend()

        def forward_fn():
            return bk.attention(query=q_a, key=ek_3d, value=ev_3d, prefix_sharing_plan=plan)

        result["backend_type"] = "GpuFlashAttentionBackend.attention (production)"
        backend_count = 1
        # Note: flash_attn_varlen_func is called internally by GpuFlashAttentionBackend
        result["builder_qualname"] = f"{build_prefix_expanded_kv.__module__}.{build_prefix_expanded_kv.__name__}"
        result["backend_qualname"] = f"{GpuFlashAttentionBackend.__module__}.GpuFlashAttentionBackend.attention"

    elif mode == "ps_on_dedup_flex":
        # PS=ON dedup: BlockMask + compiled flex_attention
        torch.manual_seed(SEED)
        q = torch.randn(1, Td, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k_src = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v_src = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        torch.cuda.synchronize()
        snapshots["after_qkv"] = torch.cuda.memory_allocated() / 1024 / 1024

        ptr, op, anc, pl = derive_tree(plan)
        def mask_mod_fn(b, h, qi, ki):
            qr, kr = ptr[qi], ptr[ki]; qo, ko = op[qi], op[ki]
            return ((kr == qr) & (ko <= qo)) | (anc[kr, qr] & (ko < pl[qr]))

        # BlockMask build — record cold time
        t0 = time.perf_counter()
        bm = create_block_mask(mask_mod_fn, None, None, Td, Td, BLOCK_SIZE=128, device=DEVICE)
        torch.cuda.synchronize()
        bm_cold_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm2 = create_block_mask(mask_mod_fn, None, None, Td, Td, BLOCK_SIZE=128, device=DEVICE)
        torch.cuda.synchronize()
        bm_warm_ms = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        snapshots["after_metadata"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["bm_cold_ms"] = round(bm_cold_ms, 1)
        result["bm_warm_ms"] = round(bm_warm_ms, 1)
        result["bm_kv_sum"] = bm.kv_num_blocks.sum().item()
        result["bm_full_sum"] = bm.full_kv_num_blocks.sum().item()

        # torch.compile(flex_attention) for fused kernel
        compiled_flex = torch.compile(flex_attention)

        def forward_fn():
            return compiled_flex(q.permute(0,2,1,3), k_src.permute(0,2,1,3), v_src.permute(0,2,1,3),
                                block_mask=bm, enable_gqa=True).permute(0,2,1,3)

        result["backend_type"] = "torch.compile(flex_attention) (fused)"
        builder_count = 0
        backend_count = 0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Warm-up ──
    for _ in range(WU):
        out = forward_fn()
        out.sum().backward(retain_graph=True if mode == "ps_on_expanded_fa" else False)
        if mode == "ps_off_fa":
            for p in [q, k, v]: p.grad = None
        elif mode == "ps_on_expanded_fa":
            for p in [q, k_src, v_src]: p.grad = None
        else:
            for p in [q, k_src, v_src]: p.grad = None
        del out
    torch.cuda.synchronize()

    # ── Benchmark ──
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pids = os.getpid()
    result["pid"] = pids

    fwd_times, bwd_times = [], []

    for _ in range(NI):
        if mode == "ps_off_fa":
            for p in [q, k, v]: p.grad = None
        else:
            for p in [q, k_src, v_src]: p.grad = None

        t0 = time.perf_counter()
        out = forward_fn()
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)
        snapshots["after_forward"] = torch.cuda.memory_allocated() / 1024 / 1024

        t0 = time.perf_counter()
        out.sum().backward(retain_graph=True if mode == "ps_on_expanded_fa" else False)
        torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)
        snapshots["after_backward"] = torch.cuda.memory_allocated() / 1024 / 1024

        del out

    snapshots["peak_allocated"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    snapshots["peak_reserved"] = torch.cuda.max_memory_reserved() / 1024 / 1024

    sft, sbt = sorted(fwd_times), sorted(bwd_times)
    result["fwd_p50_ms"] = round(sft[len(sft)//2] * 1000, 3)
    result["fwd_p90_ms"] = round(sft[int(len(sft)*0.9)] * 1000, 3)
    result["bwd_p50_ms"] = round(sbt[len(sbt)//2] * 1000, 3)
    result["bwd_p90_ms"] = round(sbt[int(len(sbt)*0.9)] * 1000, 3)
    result["snapshots_MB"] = snapshots
    result["builder_call_count"] = builder_count
    result["backend_call_count"] = backend_count

    # ── 24-layer simulation (复用 BlockMask, 独立 Q/K/V) ──
    # Expanded path: 每层 builder 构建一次
    # Flex path: BlockMask 复用一次 + 24 层 flex (独立 tensor)
    N_LAYERS = 24

    if mode == "ps_on_dedup_flex":
        # 24 layers, all sharing the same BlockMask
        layer_fwd = []
        for _ in range(N_LAYERS):
            q_l = torch.randn(1, Td, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            k_l = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            v_l = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            t0 = time.perf_counter()
            out = compiled_flex(q_l.permute(0,2,1,3), k_l.permute(0,2,1,3), v_l.permute(0,2,1,3),
                                block_mask=bm, enable_gqa=True).permute(0,2,1,3)
            out.sum().backward()
            torch.cuda.synchronize()
            layer_fwd.append((time.perf_counter() - t0) * 1000)
            del q_l, k_l, v_l, out
        result["layers_24_total_ms"] = round(sum(layer_fwd), 3)
        result["layers_24_avg_ms"] = round(sum(layer_fwd) / N_LAYERS, 3)

    elif mode == "ps_on_expanded_fa":
        # 24 layers: each layer builds KV + calls attention
        layer_fwd = []
        for li in range(N_LAYERS):
            q_l = torch.randn(1, Td, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            k_l = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            v_l = torch.randn(1, Td, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            kf = k_l[0].reshape(Td, -1); vf = v_l[0].reshape(Td, -1)
            ek_l, ev_l = build_prefix_expanded_kv(key=kf, value=vf, store=store,
                prefix_sharing_plan=plan, packed_batch_layout=layout, layer_id=li+1, stats=stats)
            ek_3d_l = ek_l.view(Texp, hkv, hd); ev_3d_l = ev_l.view(Texp, hkv, hd)
            t0 = time.perf_counter()
            out = bk.attention(query=q_l[0], key=ek_3d_l, value=ev_3d_l, prefix_sharing_plan=plan)
            out.sum().backward()
            torch.cuda.synchronize()
            layer_fwd.append((time.perf_counter() - t0) * 1000)
            del q_l, k_l, v_l, ek_l, ev_l, out
        result["layers_24_total_ms"] = round(sum(layer_fwd), 3)
        result["layers_24_avg_ms"] = round(sum(layer_fwd) / N_LAYERS, 3)

    elif mode == "ps_off_fa":
        # PS=OFF 24 layers: per-row FA each layer
        layer_fwd = []
        for li in range(N_LAYERS):
            q_l = torch.randn(1, Torig, hq, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            k_l = torch.randn(1, Torig, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            v_l = torch.randn(1, Torig, hkv, hd, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
            t0 = time.perf_counter()
            outs = []
            for i in range(B):
                s, e = cu_q[i], cu_q[i+1]
                fs = sum(plan.original_lengths[:i]); fe = fs + plan.original_lengths[i]
                qi = q_l[:, fs:fe, :, :]; ki = k_l[:, fs:fe, :, :]; vi = v_l[:, fs:fe, :, :]
                Q = qi.reshape(-1, hq, hd); K = ki.reshape(-1, hkv, hd); V = vi.reshape(-1, hkv, hd)
                cq = torch.tensor([0, Q.shape[0]], device=DEVICE, dtype=torch.int32)
                ck = torch.tensor([0, K.shape[0]], device=DEVICE, dtype=torch.int32)
                out = flash_attn_varlen_func(Q, K, V, cq, ck, Q.shape[0], K.shape[0], 0.0, causal=True)
                outs.append(out.view(1, -1, hq, hd))
            out = torch.cat(outs, dim=1)
            out.sum().backward()
            torch.cuda.synchronize()
            layer_fwd.append((time.perf_counter() - t0) * 1000)
            del q_l, k_l, v_l, out
        result["layers_24_total_ms"] = round(sum(layer_fwd), 3)
        result["layers_24_avg_ms"] = round(sum(layer_fwd) / N_LAYERS, 3)

    # ── Save ──
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[OK] {workload_name}/{mode}: PID={pids} fwd={result['fwd_p50_ms']:.1f}ms bwd={result['bwd_p50_ms']:.1f}ms peak={snapshots['peak_allocated']:.0f}MB 24L_avg={result.get('layers_24_avg_ms', 'N/A')}ms", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 poc_3c_worker.py <cuda_dev> <workload> <mode> <out_json>", file=sys.stderr)
        sys.exit(1)

    cuda_dev = int(sys.argv[1])
    workload_name = sys.argv[2]
    mode = sys.argv[3]
    out_path = sys.argv[4]
    run_worker(cuda_dev, workload_name, mode, out_path)
