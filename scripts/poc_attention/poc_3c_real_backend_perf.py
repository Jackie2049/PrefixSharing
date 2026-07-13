#!/usr/bin/env python3
"""补充3C 父进程：为每个 (workload, mode) 启动独立 Python worker。

Usage:
  cd /path/to/flex-attention
  python3 poc_3c_real_backend_perf.py

要求：
  - 每个 worker 独立进程，独立 HBM
  - worker 在 empty_cache + reset_peak 后创建输入
  - spy builder/backend.attention 调用次数
  - 报告 compile cold/warm
  - 24-layer 摊销口径
"""

import json, os, sys, subprocess, time, glob

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

WORKLOADS = ["no_sharing", "star_long_prompt", "chain_depth6", "deep_fragmented"]
MODES = ["ps_off_fa", "ps_on_expanded_fa", "ps_on_dedup_flex"]

def main():
    workdir = "/jiangdingfeng/zy/Termius/flex-attention"
    worker_py = os.path.join(workdir, "poc_3c_worker.py")
    out_dir = os.path.join(workdir, "poc_3c_results")
    os.makedirs(out_dir, exist_ok=True)

    cuda_dev = int(os.environ.get("CUDA_VISIBLE_DEVICES", "1"))

    # Launch each (workload, mode) as independent subprocess
    procs = []
    for wl in WORKLOADS:
        for mode in MODES:
            out_path = os.path.join(out_dir, f"{wl}_{mode}.json")
            if os.path.exists(out_path):
                print(f"[skip] {wl}/{mode} already exists", file=sys.stderr)
                continue
            cmd = [
                "python3", worker_py,
                str(cuda_dev), wl, mode, out_path
            ]
            log_path = os.path.join(out_dir, f"{wl}_{mode}.log")
            print(f"[launch] {wl}/{mode} -> {out_path}", file=sys.stderr)
            f = open(log_path, "w")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=workdir)
            procs.append((p, wl, mode, out_path, f))

    # Wait for all
    results = {}
    for p, wl, mode, out_path, f in procs:
        rc = p.wait()
        f.close()
        if rc == 0 and os.path.exists(out_path):
            with open(out_path) as jf:
                results[f"{wl}/{mode}"] = json.load(jf)
            print(f"[done] {wl}/{mode}: exit={rc}", file=sys.stderr)
        else:
            print(f"[FAIL] {wl}/{mode}: exit={rc}", file=sys.stderr)
            results[f"{wl}/{mode}"] = {"error": f"exit code {rc}"}

    # Summary
    print("\n\n=== SUMMARY ===", file=sys.stderr)
    for key, r in sorted(results.items()):
        if "error" in r:
            print(f"  {key}: ERROR - {r['error']}", file=sys.stderr)
        else:
            fwd = r.get("fwd_p50_ms", "?")
            bwd = r.get("bwd_p50_ms", "?")
            peak = r.get("snapshots_MB", {}).get("peak_allocated", "?")
            lay = r.get("layers_24_avg_ms", "?")
            bc = r.get("builder_call_count", "?")
            ac = r.get("backend_call_count", "?")
            bm_info = f" BM_cold={r.get('bm_cold_ms','?')}ms BM_warm={r.get('bm_warm_ms','?')}ms" if "bm_cold_ms" in r else ""
            print(f"  {key}: fwd={fwd}ms bwd={bwd}ms peak={peak}MB 24L_avg={lay}ms builder={bc} backend={ac}{bm_info}", file=sys.stderr)

    # Aggregate JSON
    agg_path = os.path.join(out_dir, "_all_results.json")
    with open(agg_path, "w") as af:
        json.dump(results, af, indent=2, default=str)
    print(f"\n[Aggregated] {agg_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
