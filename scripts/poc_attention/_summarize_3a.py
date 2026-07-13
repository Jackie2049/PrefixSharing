import json
d = json.load(open("poc_3a_results.json"))
for r in d:
    f2o = r.get("flex_vs_oracle", {})
    gfo = r.get("grad_flex_vs_ora", {})
    g = r.get("expanded_error", "")
    p = r.get("provider_directed_grad_norm", -1)
    bc = r.get("builder_calls", "?")
    if f2o:
        qg = gfo.get("q", {}).get("max", 0)
        kg = gfo.get("k", {}).get("max", 0)
        vg = gfo.get("v", {}).get("max", 0)
        print(f"{r['case']:20s} {r['dtype']:15s} calls={str(bc):>3} f2o_max={f2o.get('max',0):.2e} q_g={qg:.2e} k_g={kg:.2e} v_g={vg:.2e} prov_d={p:.1e}")
    else:
        print(f"{r['case']:20s} {r['dtype']:15s} ERROR: {g[:80]}")
