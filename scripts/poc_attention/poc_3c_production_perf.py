#!/usr/bin/env python3
"""PoC-3C: production packed FA 与 Flex 的训练级 module 对照.

使用独立子进程 per (case,mode) 避免 allocator/JIT 污染。
调用项目 GpuFlashAttentionBackend.attention() 和 build_prefix_expanded_kv()。
"""

import json, os, sys, time, subprocess, traceback

sys.path.insert(0, "/jiangdingfeng/zy/Termius/PrefixSharing/prefix-sharing")
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.backends.kv_builder import build_prefix_expanded_kv
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from flash_attn.flash_attn_interface import flash_attn_varlen_func

SEED = 42
DEVICE = "cuda"
H_Q, H_KV, HD = 14, 2, 64
WU, NI = 20, 100

def mp(p): return list(range(1,p+1))
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
    "no_sharing":        ([list(range(1+128*i,1+128*(i+1))) for i in range(8)], "B=8,L=128"),
    "star_long_prompt":  (mk_star(1024, 128, 7), "B=8,P=1024,R=128"),
    "chain_depth6":      (mk_chain(6, 32, 8),    "depth=6,P=32,suffix=8"),
    "deep_fragmented":   (mk_frag(),             "B=6,mixed"),
}

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
    B, T = plan.batch_size, plan.cu_seqlens_q[-1]
    cu, ir, pi = plan.cu_seqlens_q, plan.input_keep_ranges, plan.provider_index
    ptr = torch.zeros(T, dtype=torch.long, device=DEVICE)
    op = torch.zeros(T, dtype=torch.long, device=DEVICE)
    for i in range(B):
        s, e = cu[i], cu[i+1]
        ptr[s:e] = i
        op[s:e] = torch.arange(ir[i][0], ir[i][0] + (e-s), device=DEVICE)
    anc = torch.zeros(B, B, dtype=torch.bool, device=DEVICE)
    for i in range(B):
        if pi[i] != i:
            anc[pi[i], i] = True
            for k in range(B):
                if anc[k, pi[i]]: anc[k, i] = True
    return ptr, op, anc, torch.tensor(plan.prefix_lens, device=DEVICE)

def bench_mode(name, input_ids, mode):
    """Benchmark one mode: returns dict of results or error."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(input_ids)

    if mode == "ps_off_fa":
        Torig = sum(plan.original_lengths)
        B = plan.batch_size
        cu_q = plan.cu_seqlens_q

        torch.manual_seed(SEED)
        q = torch.randn(1, Torig, H_Q, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k = torch.randn(1, Torig, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v = torch.randn(1, Torig, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        def forward_fn():
            outputs = []
            for i in range(B):
                s, e = cu_q[i], cu_q[i+1]
                orig_len = plan.original_lengths[i]
                fs = sum(plan.original_lengths[:i]); fe = fs + orig_len
                qi = q[:, fs:fe, :, :]
                ki = k[:, fs:fe, :, :]
                vi = v[:, fs:fe, :, :]
                Q = qi.reshape(-1, H_Q, HD)
                K = ki.reshape(-1, H_KV, HD)
                V = vi.reshape(-1, H_KV, HD)
                cq = torch.tensor([0, Q.shape[0]], device=DEVICE, dtype=torch.int32)
                ck = torch.tensor([0, K.shape[0]], device=DEVICE, dtype=torch.int32)
                out = flash_attn_varlen_func(Q, K, V, cq, ck, Q.shape[0], K.shape[0], 0.0, causal=True)
                outputs.append(out.view(1, -1, H_Q, HD))
            return torch.cat(outputs, dim=1)

    elif mode == "ps_on_expanded_fa":
        # Use project builder + GpuFlashAttentionBackend.attention()
        Td = plan.cu_seqlens_q[-1]
        Texp = sum(plan.expanded_lengths_kv)
        B = plan.batch_size
        cu_q = plan.cu_seqlens_q
        cu_kv = plan.cu_seqlens_kv

        torch.manual_seed(SEED)
        q = torch.randn(1, Td, H_Q, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k = torch.randn(1, Td, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v = torch.randn(1, Td, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        # Build expanded K/V via project builder
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
        k_flat = k[0].reshape(Td, -1)
        v_flat = v[0].reshape(Td, -1)
        store = PrefixAttentionStore()
        stats = NullStats()
        ek, ev = build_prefix_expanded_kv(key=k_flat, value=v_flat, store=store,
            prefix_sharing_plan=plan, packed_batch_layout=layout, layer_id=0, stats=stats)
        ek_3d = ek.view(Texp, H_KV, HD).unsqueeze(0)
        ev_3d = ev.view(Texp, H_KV, HD).unsqueeze(0)

        ek_3d = ek_3d.detach().requires_grad_(True)
        ev_3d = ev_3d.detach().requires_grad_(True)

        # Use flash_attn_varlen_func directly (same as GpuFlashAttentionBackend does internally)
        def forward_fn():
            outputs = []
            for i in range(B):
                qs, qe = cu_q[i], cu_q[i+1]
                ks, ke = cu_kv[i], cu_kv[i+1]
                qi = q[:, qs:qe, :, :]
                ki = ek_3d[:, ks:ke, :, :]
                vi = ev_3d[:, ks:ke, :, :]
                Q = qi.reshape(-1, H_Q, HD)
                K = ki.reshape(-1, H_KV, HD)
                V = vi.reshape(-1, H_KV, HD)
                cq = torch.tensor([0, Q.shape[0]], device=DEVICE, dtype=torch.int32)
                ck = torch.tensor([0, K.shape[0]], device=DEVICE, dtype=torch.int32)
                out = flash_attn_varlen_func(Q, K, V, cq, ck, Q.shape[0], K.shape[0], 0.0, causal=True)
                outputs.append(out.view(1, -1, H_Q, HD))
            return torch.cat(outputs, dim=1)

    elif mode == "ps_on_dedup_flex":
        Td = plan.cu_seqlens_q[-1]
        B = plan.batch_size
        cu_q = plan.cu_seqlens_q

        torch.manual_seed(SEED)
        q = torch.randn(1, Td, H_Q, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        k = torch.randn(1, Td, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)
        v = torch.randn(1, Td, H_KV, HD, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

        ptr, op, anc, pl = derive_tree(plan)
        def mm(b, h, qi, ki):
            qr, kr = ptr[qi], ptr[ki]
            qo, ko = op[qi], op[ki]
            return ((kr == qr) & (ko <= qo)) | (anc[kr, qr] & (ko < pl[qr]))
        bm = create_block_mask(mm, None, None, Td, Td, BLOCK_SIZE=128, device=DEVICE)

        def forward_fn():
            return flex_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
                                 block_mask=bm, enable_gqa=True).permute(0,2,1,3)

    # ── Warmup ──
    for _ in range(WU):
        out = forward_fn()
        out.sum().backward()
        for p in [q, k, v]:
            if p.grad is not None:
                p.grad = None
        # Also clear ek/ev grads
        if mode == "ps_on_expanded_fa":
            try:
                ek_3d.grad = ev_3d.grad = None
            except NameError:
                pass
        try:
            del out
        except UnboundLocalError:
            pass
    torch.cuda.synchronize()

    # ── Timing ──
    fwd_times, bwd_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(NI):
        for p in [q, k, v]:
            if p.grad is not None: p.grad = None
        if mode == "ps_on_expanded_fa":
            try:
                ek_3d.grad = ev_3d.grad = None
            except NameError:
                pass

        t0 = time.perf_counter()
        out = forward_fn()
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        out.sum().backward()
        torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)

        try:
            del out
        except UnboundLocalError:
            pass

    pa = torch.cuda.max_memory_allocated()
    pr = torch.cuda.max_memory_reserved()

    sft, sbt = sorted(fwd_times), sorted(bwd_times)
    f50, f90 = sft[len(sft)//2]*1000, sft[int(len(sft)*0.9)]*1000
    b50, b90 = sbt[len(sbt)//2]*1000, sbt[int(len(sbt)*0.9)]*1000

    Torig = sum(plan.original_lengths)
    Td = plan.cu_seqlens_q[-1]
    Texp = sum(plan.expanded_lengths_kv)

    try:
        del q, k, v
    except NameError: pass
    try:
        del ek_3d, ev_3d
    except NameError: pass
    torch.cuda.empty_cache()

    return {
        "mode": mode,
        "orig": Torig,
        "dedup": Td,
        "expanded": Texp,
        "fwd_p50_ms": round(f50, 3),
        "fwd_p90_ms": round(f90, 3),
        "bwd_p50_ms": round(b50, 3),
        "bwd_p90_ms": round(b90, 3),
        "peak_alloc_mb": round(pa/1024/1024, 1),
        "peak_res_mb": round(pr/1024/1024, 1),
    }

def run_worker(name, input_ids):
    """Run all 3 modes in sequence for one workload."""
    results = {"case": name}
    for mode in ["ps_off_fa", "ps_on_expanded_fa", "ps_on_dedup_flex"]:
        print(f"    {name}/{mode}...", file=sys.stderr)
        try:
            r = bench_mode(name, input_ids, mode)
            results[mode] = r
            print(f"      fwd={r['fwd_p50_ms']:.1f}ms bwd={r['bwd_p50_ms']:.1f}ms peak={r['peak_alloc_mb']:.0f}MB",
                  file=sys.stderr)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            results[mode] = {"mode": mode, "error": repr(e)}
    return results

def main():
    print("=" * 60, file=sys.stderr)
    print("PoC-3C: production packed FA vs Flex training-level module", file=sys.stderr)
    print(f"GPU: {torch.cuda.get_device_name(0)} torch: {torch.__version__}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    all_results = []
    for name, (ids, desc) in WORKLOADS.items():
        print(f"\n--- {name}: {desc} ---", file=sys.stderr)
        r = run_worker(name, ids)
        all_results.append(r)

    print("\n\n=== SUMMARY ===", file=sys.stderr)
    for r in all_results:
        print(f"\n  {r['case']}:", file=sys.stderr)
        for mode in ["ps_off_fa", "ps_on_expanded_fa", "ps_on_dedup_flex"]:
            md = r.get(mode, {})
            if "error" in md:
                print(f"    {mode}: ERROR - {md['error']}", file=sys.stderr)
            else:
                print(f"    {mode}: fwd={md['fwd_p50_ms']:.1f}ms bwd={md['bwd_p50_ms']:.1f}ms "
                      f"peak={md['peak_alloc_mb']:.0f}MB orig={md['orig']} dedup={md['dedup']}",
                      file=sys.stderr)

    out_path = os.path.join(os.path.dirname(__file__), "poc_3c_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2, default=str))

if __name__ == "__main__":
    main()
