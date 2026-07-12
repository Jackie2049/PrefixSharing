import torch, json, sys, time, traceback
sys.path.insert(0, "prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED=42; DEVICE="cuda"; H_Q,H_KV,HD=14,2,64; DTYPE=torch.bfloat16
WU,NI=20,100; BS=[64,128,256]

def mp(p): return list(range(1,p+1))
def mb(p): return [mp(p)+list(range(100,125)),mp(p)+list(range(200,232)),mp(p)+list(range(300,318))]
def ms(pl,rl,n):
    p=mp(pl); return [p+list(range(100,100+rl))]+[p+list(range(200+pl*i,200+pl*i+rl)) for i in range(n)]
def mu(n):
    p=mp(67); rl=[23+i*7 for i in range(n)]
    r=[p+list(range(100,100+29+n))]
    for i in range(n): r.append(p+list(range(200+100*i,200+100*i+rl[i])))
    return r
def mc(d,pl,sl):
    p=mp(pl); r=[p]
    for i in range(1,d): r.append(r[-1]+list(range(300+sl*i,300+sl*(i+1))))
    return r
def mf(): return [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
    [1,2,3,4,5,6,7,8,9,10,11,12,13],[1,2,3,4,5,6,7,8,100,101,102],
    [1,2,3,4,5,6,7,8,100,101,102,103,104],[1,2,3,4,5,200,201,202]]

def gen_wl():
    w={}
    for pl,rl,n in [(64,65,4),(512,128,8),(1024,128,8)]:
        w[f"s_p{pl}r{rl}x{n}"]=ms(pl,rl,n)
    for d,pl,sl in [(3,64,16),(6,32,8),(12,16,4)]:
        w[f"c_d{d}_p{pl}_s{sl}"]=mc(d,pl,sl)
    w["una"]=mu(4); w["fra"]=mf()
    w["no_share"]=[list(range(1+128*i,1+128*(i+1))) for i in range(8)]
    return w

def dt(plan):
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
    return {"T":T,"ptr":ptr,"op":op,"anc":anc,"pl":torch.tensor(plan.prefix_lens),"cu":cu,"B":B}

def lp(tr):
    T,B,cu=tr["T"],tr["B"],tr["cu"]; c=0
    for i in range(B):
        s,e=cu[i],cu[i+1]; thi=tr["pl"][i].item()
        c+=(e-s)*(e-s+1)//2
        for kk in range(T):
            ki=tr["ptr"][kk].item()
            if ki!=i and tr["anc"][ki,i] and tr["op"][kk].item()<thi: c+=(e-s)
    return c

def pct(d,p): s=sorted(d); return s[int(len(s)*p/100)]

def run_wl(name,ids):
    cfg=PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
    plan=PrefixSharingPlanner(cfg).plan(ids); tr=dt(plan); T=tr["T"]
    print(f"{name}: rows={plan.batch_size} orig={sum(plan.original_lengths)} dedup={T} exp={sum(plan.expanded_lengths_kv)} lp={lp(tr)}",file=sys.stderr)
    torch.manual_seed(SEED)
    q=torch.randn(1,T,H_Q,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)
    k=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)
    v=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE).requires_grad_(True)
    ptr,op,anc,pl=tr["ptr"].cuda(),tr["op"].cuda(),tr["anc"].cuda(),tr["pl"].cuda()

    def mkmm():
        def mm(b,h,qi,ki):
            qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
            return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
        return mm

    def fa(qt,kt,vt,bm_):
        # flex_attention expects (B, H, T, D)
        return flex_attention(qt.permute(0,2,1,3),kt.permute(0,2,1,3),vt.permute(0,2,1,3),block_mask=bm_,enable_gqa=True)

    br={}
    for b in BS:
        if b > T and b != 64: continue
        t0=time.perf_counter()
        bm=create_block_mask(mkmm(),None,None,T,T,BLOCK_SIZE=b,device=DEVICE)
        torch.cuda.synchronize(); cold_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter()
        bm2=create_block_mask(mkmm(),None,None,T,T,BLOCK_SIZE=b,device=DEVICE)
        torch.cuda.synchronize(); warm_ms=(time.perf_counter()-t0)*1000
        del bm2

        # fwd only
        for _ in range(WU): fa(q,k,v,bm)
        torch.cuda.synchronize()
        ft=[]
        for _ in range(NI):
            t0=time.perf_counter(); fa(q,k,v,bm); torch.cuda.synchronize()
            ft.append(time.perf_counter()-t0)

        # fwd+bwd
        for _ in range(WU):
            o=fa(q,k,v,bm); o.sum().backward(); q.grad=k.grad=v.grad=None
        torch.cuda.synchronize()
        bt=[]
        for _ in range(NI):
            q.grad=k.grad=v.grad=None
            t0=time.perf_counter(); o=fa(q,k,v,bm); o.sum().backward(); torch.cuda.synchronize()
            bt.append(time.perf_counter()-t0)

        # peak memory
        torch.cuda.reset_peak_memory_stats()
        o=fa(q,k,v,bm); o.sum().backward(); torch.cuda.synchronize()
        pa=torch.cuda.max_memory_allocated(); pr=torch.cuda.max_memory_reserved()
        q.grad=k.grad=v.grad=None

        # scheduled blocks
        try: nb=bm.num_blocks
        except: nb=(T+b-1)//b
        sb=nb if isinstance(nb,int) else ((T+b-1)//b)
        sp=sb*b*b; lpl=lp(tr)

        br[f"bs{b}"]={"cold_ms":round(cold_ms,1),"warm_ms":round(warm_ms,1),
            "fwd50":round(pct(ft,50)*1000,3),"fwd90":round(pct(ft,90)*1000,3),
            "bwd50":round(pct(bt,50)*1000,3),"bwd90":round(pct(bt,90)*1000,3),
            "peak_alloc_mb":round(pa/1024/1024,1),"peak_res_mb":round(pr/1024/1024,1),
            "blocks":sb,"sched_pairs":sp,"log_pairs":lpl,"ratio":round(sp/max(lpl,1),2),
            "dedup":T,"expanded":sum(plan.expanded_lengths_kv)}
        print(f"  bs={b}: cold={cold_ms:.0f}ms warm={warm_ms:.0f}ms fwd={pct(ft,50)*1000:.1f}ms bwd={pct(bt,50)*1000:.1f}ms peak={pa/1024/1024:.0f}MB",file=sys.stderr)
    del q,k,v,bm; torch.cuda.empty_cache()
    return {"wl":name,"rows":plan.batch_size,"orig":sum(plan.original_lengths),"dedup":T,"expanded":sum(plan.expanded_lengths_kv),"br":br}

R=[]
print("=== PoC-B+C ===",file=sys.stderr)
print(f"GPU:{torch.cuda.get_device_name(0)} {torch.__version__} cuda:{torch.version.cuda}",file=sys.stderr)
for n,ids in gen_wl().items():
    try: R.append(run_wl(n,ids))
    except:
        e=sys.exc_info()[1]; traceback.print_exc(file=sys.stderr)
        R.append({"wl":n,"error":str(e)})

# Dynamic shape
print("\n=== Dynamic ===",file=sys.stderr)
shapes=[ms(1024,128,8),mb(33)+mb(33),mc(8,128,20),ms(1024,128,8)]
cfg=PrefixSharingConfig(enable_prefix_sharing=True,min_prefix_len=3,min_group_size=2)
DYN=[]
for idx in range(50):
    try:
        ids=shapes[idx%4]; plan=PrefixSharingPlanner(cfg).plan(ids); tr=dt(plan); T=tr["T"]
        ptr,op,anc,pl=tr["ptr"].cuda(),tr["op"].cuda(),tr["anc"].cuda(),tr["pl"].cuda()
        q=torch.randn(1,T,H_Q,HD,dtype=DTYPE,device=DEVICE)
        k=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE)
        v=torch.randn(1,T,H_KV,HD,dtype=DTYPE,device=DEVICE)
        def mkm():
            def mm(b,h,qi,ki):
                qr,kr=ptr[qi],ptr[ki]; qo,ko=op[qi],op[ki]
                return ((kr==qr)&(ko<=qo))|(anc[kr,qr]&(ko<pl[qr]))
            return mm
        t0=time.perf_counter()
        bm=create_block_mask(mkm(),None,None,T,T,BLOCK_SIZE=128,device=DEVICE)
        torch.cuda.synchronize(); mask_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter()
        flex_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),block_mask=bm,enable_gqa=True)
        torch.cuda.synchronize(); fwd_ms=(time.perf_counter()-t0)*1000
        DYN.append({"i":idx,"shape":idx%4,"T":T,"mask_ms":round(mask_ms,1),"fwd_ms":round(fwd_ms,1)})
        del q,k,v,bm
        if idx%20==0: torch.cuda.empty_cache()
    except:
        DYN.append({"i":idx,"error":str(sys.exc_info()[1])})

print(json.dumps({"meta":{"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda},"results":R,"dynamic":DYN},indent=2))
