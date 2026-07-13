import sys, json
d = json.load(open("poc_3a_real_path_results.json"))
print()
for r in d:
    if r.get("dtype") != "torch.float32": continue
    e2o = r.get("exp_vs_oracle", {})
    f2o = r.get("flex_vs_oracle", {})
    bc = r.get("builder_calls","?")
    pk = r.get("provider_k_grad_norm", -1)
    pv = r.get("provider_v_grad_norm", -1)
    print("%-20s fp32: bc=%s e2o_max=%s f2o_max=%s pk=%.1f pv=%.1f" % (
        r["case"], str(bc),
        str(e2o.get("max","?")),
        str(f2o.get("max","?")),
        pk, pv))
for r in d:
    if r.get("dtype") != "torch.bfloat16": continue
    e2f = r.get("exp_vs_flex", {})
    bc = r.get("builder_calls","?")
    pk = r.get("provider_k_grad_norm", -1)
    pv = r.get("provider_v_grad_norm", -1)
    print("%-20s bf16: bc=%s e2f_max=%s e2f_rel=%s" % (
        r["case"], str(bc),
        str(e2f.get("max","?")),
        str(e2f.get("rel_l2","?"))))
PY