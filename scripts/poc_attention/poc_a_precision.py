import torch, json, sys, time, traceback
sys.path.insert(0, "prefix-sharing")
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

SEED = 42; DEVICE = "cuda"
H_Q, H_KV, HD = 14, 2, 64; DTYPE = torch.float32

def make_prefix(p): return list(range(1, p+1))
def make_star(plen, rlen, n):
    p = make_prefix(plen)
    return [p + list(range(100, 100+rlen))] + [p + list(range(200+plen*i, 200+plen*i+rlen)) for i in range(n)]
def make_unaligned(n):
    p = make_prefix(67); rlens = [23 + i*7 for i in range(n)]
    rows = [p + list(range(100, 100+29+n))]
    for i in range(n): rows.append(p + list(range(200+100*i, 200+100*i+rlens[i])))
    return rows
def make_branch(plen):
    p = make_prefix(plen)
    return [p + list(range(100,125)), p + list(range(200,232)), p + list(range(300,318))]

CASES = [
    ("no_sharing", [list(range(1,9)), list(range(9,17)), list(range(17,25)), list(range(25,33))],
     "4 distinct seqs L=8"),
    ("star_aligned", make_star(64, 65, 4), "1 prov P64+R65, 4 reusers"),
    ("star_unaligned", make_unaligned(4), "1 prov P67+Rn, 4 reusers unaligned"),
    ("branch", make_branch(32), "1 prov P32+A25, 2 reusers"),
    ("chain", [list(range(1,33)), list(range(1,33))+list(range(100,116)),
               list(range(1,33))+list(range(100,116))+list(range(200,208))],
     "row0->row1->row2"),
    ("deep_frag", [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11],
                   [1,2,3,4,5,6,7,8,9,10,11,12,13],
                   [1,2,3,4,5,6,7,8,100,101,102],
                   [1,2,3,4,5,6,7,8,100,101,102,103,104],
                   [1,2,3,4,5,200,201,202]], "mixed depths 6 rows"),
]

def derive_tree(plan):
    """Compute per-token metadata: ptr[flat_idx]->row, op->original_pos, anc->ancestor."""
    B, T = plan.batch_size, plan.cu_seqlens_q[-1]
    cu, ir, pl, pi = plan.cu_seqlens_q, plan.input_keep_ranges, plan.prefix_lens, plan.provider_index
    ptr = torch.zeros(T, dtype=torch.long); op = torch.zeros(T, dtype=torch.long)
    row_range = []
    for i in range(B):
        s, e = cu[i], cu[i+1]
        ptr[s:e] = i; op[s:e] = torch.arange(ir[i][0], ir[i][0]+(e-s))
        row_range.append((s, e))
    # Ancestor matrix: anc[ancestor_row, descendant_row] = True
    anc = torch.zeros(B, B, dtype=torch.bool)
    for i in range(B):
        if pi[i] != i:
            anc[pi[i], i] = True
            for k in range(B):
                if anc[k, pi[i]]: anc[k, i] = True
    return {"T": T, "B": B, "ptr": ptr, "op": op, "anc": anc,
            "pl": torch.tensor(pl), "cu": cu, "row_range": row_range}

def make_dense_prefix_tree_mask(tree, device):
    """Build [T, T] bool mask for prefix tree semantics.

    For each query token q at row r:
      - same-row tokens with original_pos(K) <= original_pos(Q): True (causal)
      - tokens from ancestor rows with original_pos(K) < prefix_len(r): True (full)
    """
    T, B = tree["T"], tree["B"]
    ptr, op = tree["ptr"], tree["op"]
    anc, pl = tree["anc"], tree["pl"]
    rr = tree["row_range"]

    dmask = torch.zeros(T, T, dtype=torch.bool, device=device)
    for i in range(B):
        s, e = rr[i]
        this_prefix_len = pl[i].item()
        for qq in range(s, e):
            q_orig = op[qq].item()
            # Same row: causal within row
            dmask[qq, s:qq+1] = (op[s:qq+1] <= q_orig)
            # Ancestor rows: full visibility
            for kk in range(T):
                kr = ptr[kk].item()
                if kr != i and anc[kr, i] and op[kk].item() < this_prefix_len:
                    dmask[qq, kk] = True
    return dmask

def run_case(name, input_ids):
    cfg = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(cfg).plan(input_ids)
    tree = derive_tree(plan)
    T = tree["T"]
    print(f"  {name}: rows={plan.batch_size} orig={sum(plan.original_lengths)} dedup={T} expanded={sum(plan.expanded_lengths_kv)}", file=sys.stderr)

    torch.manual_seed(SEED)
    q = torch.randn(1, T, H_Q, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, T, H_KV, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, T, H_KV, HD, dtype=DTYPE, device=DEVICE)

    ptr, op = tree["ptr"].cuda(), tree["op"].cuda()
    anc, pl = tree["anc"].cuda(), tree["pl"].cuda()

    # Dense prefix-tree mask (correct semantics)
    dmask = make_dense_prefix_tree_mask(tree, DEVICE)

    # Flex block mask
    def mkm():
        def mm(b, h, qi, ki):
            qr, kr = ptr[qi], ptr[ki]
            qo, ko = op[qi], op[ki]
            same = (kr == qr) & (ko <= qo)
            ancv = anc[kr, qr] & (ko < pl[qr])
            return same | ancv
        return mm
    bm = create_block_mask(mkm(), None, None, T, T, BLOCK_SIZE=128, device=DEVICE)

    def docmp(qq, kk, vv, dm):
        kr = kk.repeat_interleave(H_Q//H_KV, dim=2)
        vr = vv.repeat_interleave(H_Q//H_KV, dim=2)
        o = torch.nn.functional.scaled_dot_product_attention(
            qq.permute(0,2,1,3), kr.permute(0,2,1,3), vr.permute(0,2,1,3),
            attn_mask=dm[None,None,:,:].expand(qq.shape[0], H_Q, -1, -1))
        return o.permute(0,2,1,3)

    def ffa(qq, kk, vv, bm_):
        o = flex_attention(qq.permute(0,2,1,3), kk.permute(0,2,1,3), vv.permute(0,2,1,3), block_mask=bm_, enable_gqa=True)
        return o.permute(0,2,1,3)

    ref = docmp(q, k, v, dmask)
    fout = ffa(q, k, v, bm)
    diff = (ref - fout).float().abs()
    mx, mn = diff.max().item(), diff.mean().item()
    bad = torch.isnan(diff).any().item() or torch.isinf(diff).any().item()

    # Gradient comparison
    torch.manual_seed(SEED+1); noise = torch.randn_like(q)
    q1,k1,v1 = [x.detach().clone().requires_grad_(True) for x in (q,k,v)]
    torch.sum(docmp(q1,k1,v1,dmask)*noise).backward()
    g1 = (q1.grad.clone(), k1.grad.clone(), v1.grad.clone())
    q2,k2,v2 = [x.detach().clone().requires_grad_(True) for x in (q,k,v)]
    torch.sum(ffa(q2,k2,v2,bm)*noise).backward()
    g2 = (q2.grad.clone(), k2.grad.clone(), v2.grad.clone())
    gd = {}
    for ng, ga, gb in [("q",g1[0],g2[0]),("k",g1[1],g2[1]),("v",g1[2],g2[2])]:
        d = (ga-gb).float().abs()
        gd[ng] = {"max": round(d.max().item(), 8), "mean": round(d.mean().item(), 8)}
        bad = bad or bool(torch.isnan(ga).any() or torch.isinf(ga).any() or torch.isnan(gb).any() or torch.isinf(gb).any())

    del q,k,v,dmask,bm,ref,fout,q1,k1,v1,q2,k2,v2; torch.cuda.empty_cache()
    ok = mx < 2e-5 and not bad
    print(f"  {'PASS' if ok else 'FAIL'}: out_max={mx:.2e} out_mean={mn:.2e} gq_max={gd['q']['max']:.2e}",
          file=sys.stderr)
    return {"case": name, "orig": sum(plan.original_lengths), "dedup": T,
            "expanded": sum(plan.expanded_lengths_kv),
            "output_max_abs": round(mx,8), "output_mean_abs": round(mn,8),
            "gradient": gd, "finite": not bad, "passed": ok}

results, failed = [], []
for name, ids, desc in CASES:
    print(f"--- {name}: {desc} ---", file=sys.stderr)
    try:
        r = run_case(name, ids); results.append(r)
        if not r["passed"]: failed.append(name)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        results.append({"case": name, "error": str(e)}); failed.append(name)

print("\n=== SUMMARY ===", file=sys.stderr)
for r in results:
    c = r.get("case","?")
    if "error" in r: print(f"  {c}: ERROR - {r['error']}", file=sys.stderr)
    else:
        print(f"  {c}: {'PASS' if r.get('passed') else 'FAIL'} "
              f"out={r.get('output_max_abs','?'):.2e} "
              f"gq={r.get('gradient',{}).get('q',{}).get('max','?')}", file=sys.stderr)
print(f"\nFailed: {len(failed)}: {failed}", file=sys.stderr)
print(json.dumps({"results": results, "failed": failed}, indent=2))
