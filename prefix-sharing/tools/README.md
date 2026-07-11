# PrefixSharing Tools

This directory keeps standalone verification and benchmark entry points.  These
scripts are not imported by the training runtime and may print progress to
stdout.

## Kept tools

- `verify_p0_correctness.py`: correctness guard for prefilter and `build_kv`
  semantics, including gradient preservation.
- `perf_baseline_benchmark.py`: focused performance baseline for detector,
  planner, KV expansion, attention, and memory overhead.
- `perf_comprehensive_benchmark.py`: broader performance matrix across sharing
  patterns, batch sizes, sequence lengths, model shapes, and backends.
- `test_e_plan_representation.py`: plan-representation analysis with prefilter
  and detector/planner split.
- `perf_test_e_plan_rep.py`: compact runner for the same plan-representation
  performance question.

## Cleanup policy

Keep scripts that are still useful for release-level correctness or performance
validation.  Before removing a tool, check whether it protects one of these
areas:

- precision consistency against a reference implementation;
- gradient preservation through reused prefix activations;
- FSDP or MCore integration behavior;
- performance regressions on detector, planner, KV expansion, attention, or
  memory usage.

Historical one-off experiments should be removed once their result has been
captured elsewhere and no current workflow references the script.
