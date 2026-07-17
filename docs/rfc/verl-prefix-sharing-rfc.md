# [RFC] Integrate PrefixSharing to verl: Generalizing PrefixGrouper for Arbitrary-Prefix Reuse in Agentic RL Training

## 1. Summary

**Prefix repetition across trajectories is prevalent** in Agentic RL scenarios, especially in GRPO, Step-wise, and Tree-structure rollout paradigms. Current training workflow always recomputes these prefix sub-sequences separately for different trajectories (during compute_old_log_prob & update_actor), resulting in redundant memory overhead and computational cost.

This issue proposes **PrefixSharing to extend verl's existing PrefixGrouper from fixed-length prompt reuse to arbitrary-length prefix sharing**, enabling better support for Step-wise and Tree-structure rollout trajectories. The key mechanism is to construct a prefix tree for input batches, then reuse KV activations during attention computation while preserving baseline logits, LogP, loss and gradient precisions. The current PrefixSharing has improved training throughput by xxx and reduced memory consumption by yyy.

Prototype: https://github.com/Jackie2049/PrefixSharing/tree/open-source

## 2. Concepts

* **Provider**: A trajectory sample whose prefix is shared and reused by reuser samples within the same batch.
* **Reuser**: A trajectory who reuses providers' prefix. Note that a reuser can also be other reusers' provider in complex scenarios.

## 3. Motivation

Prefix redundancy appears in 3 typical rollout paradigms, with different requirements on prefix reuse algorithm.

### 3.1 GRPO-style trajectories

```text
trajectory 0: [prompt P][response A] <--- reward 0
trajectory 1: [prompt P][response B] <--- reward 1
trajectory 2: [prompt P][response C] <--- reward 2
```

This is the simplest reuse workload, where all trajectories share fix-length prompts (_'prompt P'_) and their responses diverge (_'response A/B/C'_). verl's existing PrefixGrouper splits attention into two stages (prompt & response) to enable prompt-only prefix reuse. PrefixSharing also supports prompt reuse by prefix tree algorithm, but DOES NOT aim to replace PrefixGroupers's simpler and more efficient path.

### 3.2 Step-style trajectories

```text
trajectory 0: [prompt P][step 1][step 2A] <--- reward 0        
trajectory 1: [prompt P][step 1][step 2B] <--- reward 1
trajectory 2: [prompt P][step 1][step 2A][step 3A] <--- reward 2
trajectory 3: [prompt P][step 1][step 2A][step 3B] <--- reward 3
trajectory 4: [prompt P][step 1][step 2A][step 3A][step 4A] <--- reward 4
```

Step-wise rollout generates trajectories where some of them (_'trajectory 0'_) are sub-sequence and history steps of the others (_'trajectory 2/3'_). In this scenario, prefix lengths differ from one to another, and a trajectory that reuses prefix can also become provider of longer prefix for subsequent trajectories (**'trajectory 0' => 'trajectory 2' => 'trajectory 4'**). PrefixGrouper is not able to express such chained reusing relationship and fails to reuse prefix longer than prompt, while PrefixSharing handles this easily by a Trie.

### 3.3 Tree-style trajectories

```text
                         /-- [turn 2A] -- [leaf A]
[root][turn 1 shared] --+
                         \-- [turn 2B] -- [leaf B]
                                           \-- [turn 3B] -- [leaf C]

trajectory 0: [root][turn 1 shared][turn 2A][leaf A] <--- reward 0
trajectory 1: [root][turn 1 shared][turn 2B][leaf B] <--- reward 1
trajectory 2: [root][turn 1 shared][turn 2B][turn 3B][leaf C] <--- reward 1
```

Tree-structure rollout produces branches with common ancestors and it is similar with Step-wise rollout from the view of trajectories. Common ancestors of leaves (_'[turn 1 shared][turn 2B]'_) are naturally common prefix among trajectories (_'trajectory 1'_ & _'trajectory 2'_). Currently verl and PrefixGrouper does not support reusing arbitray prefix in Tree-structure trajectories. 

## 4. Design

### 4.1 Scope

- Support arbitrary prefix reuse within micro-batch.
- Performance improvement (less memory consumption or faster forward).
- Preserve baseline forward and backward semantics. Compuatation precision aligned.
- Inherit and extend PrefixGrouper's interface in verl. **Minimal modification to verl**.
- FSDP/transformers as training engine for the first version.
- Support full attention (e.g. Qwen-2/2.5/3) for the first version.

### 4.2 Overview

```mermaid
flowchart TD
    A["forward step entry: verl/transformer_impl.py"]
    A --> B{"Reuse mode"}
    B -->|"prompt_only"| C["PrefixGrouper"]
    B -->|"arbitrary_prefix"| D["PrefixSharing"]
    D --> E1["detect & plan reuse relationships: PrefixSharing/planner.py"]
    E1 --> E2["trim redundant inputs: PrefixSharing/batch_trim.py"]
    E2 --> E3["create context: PrefixSharing/context.py"]
    E3 --> G["dispatch attention: transformers/modeling_utils.py"]
    G --> F["KV reuse & compute attention: PrefixSharing/attention.py"]
    F --> J["restore prefix tokens: verl/verl_mcore.py"]
    C --> K["standard verl training outputs: loss, entropy, logp"]
    J --> K
    K --> L["subsequent training process"]

    classDef verl fill:#dbeafe,stroke:#2563eb,color:#111827,stroke-width:2px
    classDef prefixSharing fill:#dcfce7,stroke:#16a34a,color:#111827,stroke-width:2px
    classDef transformers fill:#fef9c3,stroke:#ca8a04,color:#111827,stroke-width:2px
    classDef existing fill:#f3f4f6,stroke:#6b7280,color:#111827,stroke-width:1px
    class A,B,K,L verl
    class D,E1,E2,E3,F,J prefixSharing
    class G transformers
    class C existing
```

**Walkthrough (verl impact).** PrefixSharing hooks verl at `forward_step` only; rollout, dataloader, and PPO loss stay unchanged.

1. **Entry (A → B).** Patched `FSDPEngineWithLMHead.forward_step` reads `prefix_grouper.mode` (`prompt_only` | `arbitrary_prefix`). Disabled or non-`arbitrary_prefix` paths call the original `forward_step` with zero overhead.
2. **Prompt-only (C → K).** Existing PrefixGrouper path; no PrefixSharing involvement.
3. **Prepare (D → E3).** Inside the patch, before `prepare_model_inputs()`: plan reuse (`planner.py`), trim reuser prefixes (`batch_trim.py`), open runtime context (`context.py`). verl batch keys and engine prepare hooks are reused on the trimmed micro-batch.
4. **Forward (E3 → G → F).** verl still runs `self.module(...)` unchanged. Transformers dispatches attention via `ALL_ATTENTION_FUNCTIONS.get_interface`; PrefixSharing intercepts externally and performs KV reuse. FSDP/Megatron engines keep autocast, FSDP wrap, and `use_cache=False`.
5. **Restore (F → J → K).** After `prepare_model_outputs()`, restore reuser prefix columns to baseline layout (`integrations/verl_mcore.py`). log_probs / entropy match the **original** micro-batch shape so `loss_function` and backward need no changes.
6. **Afterward (K → L).** Standard verl training continues (loss, optimizer, sync).

**verl touch surface:** config (`prefix_grouper.mode`), monkey-patched `forward_step`, external attention hooks — everything else unchanged.

### 4.3 Integration

PrefixSharing plugs into verl as a **mode extension of PrefixGrouper**, not a parallel training entry. The first upstream PR targets the FSDP / Transformers path. Prototype code today reaches the same surfaces via external monkey-patches; the upstream form replaces those patches with thin in-tree hooks and keeps PrefixSharing algorithm logic in an external package.

**FSDP first PR — verl files touched**

| verl files | Impact | Modification |
|---|---|---|
| `workers/config/actor.py` | config schema | Keep `use_prefix_grouper` as the enable switch. Add fields like `mode` to `prefix_grouper`;  |
| `workers/engine/fsdp/transformer_impl.py` | main training hook | In `FSDPEngineWithLMHead.forward_step`, branch on `mode`: `mode: prompt_only` keeps existing PrefixGrouper path; `mode: arbitrary_prefix` runs PrefixSharing(plan reusing → trim prefixes → create context → forward with prefix sharing → prefix logits/logp/entropy restore for outputs) |
| `models/transformers/monkey_patch.py` | attention dispatch | Extend existing PrefixGrouper wrappers so that attention is routed to PrefixSharing's arbitrary KV reuse when its context is active; inactive context remains a zero-overhead passthrough. |
| `trainer/ppo/prefix_grouper_utils.py` | mode boundary | Keep prompt-only helpers as-is; ensure `arbitrary_prefix` does not enter PrefixGrouper's fixed prompt/response regroup layout. |

**Explicitly unchanged in verl**

| Area | Files / components | Why |
|---|---|---|
| Rollout / inference | vLLM / SGLang workers | PrefixSharing is actor / ref training only. |
| Data pipeline | dataloader, sampler | Consumes the same micro-batch keys. |
| PPO objective | `verl/workers/utils/losses.py`, advantage / KL helpers | Restore returns baseline-shaped `log_probs` / entropy; loss code needs no rewrite. |
| Trainer control loop | `ray_trainer.py` step orchestration | Existing `use_prefix_grouper` batch-balance behavior can remain; no new trainer stage. |
| Optimizer / checkpoint / param sync | FSDP engine non-forward paths | Outside the forward_step boundary. |

**Megatron follow-up (not in the first PR)** — same integration idea, larger surface: `verl/workers/engine/megatron/transformer_impl.py` (`forward_step` + logprob restore), and trimming-aware `no_padding_2_padding` call sites under `verl/workers/utils/` and trainer loss helpers.

### 4.5 Quick Use

参数集成、快速使用

We propose a backward-compatible extension of the existing PrefixGrouper configuration:

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
```

The exact naming is open for discussion. The first upstream path would integrate with verl's FSDP model engine and Transformers attention dispatch. Megatron support is also ready and can follow separately after the common interface stabilizes.

### 3.5 Attention

The KV-injection backend follows three rules:

1. **One forward graph:** provider and reuser sequences remain in the same differentiable forward.
2. **KV reuse without `detach()`:** reusers attend to provider prefix KV plus their own suffix KV, and gradients flow through the provider computation.
3. **Prefix-last restore:** trimming a reuser prefix removes the output needed for its first suffix-token log-probability. The backend restores that boundary output before verl consumes token-level results.

These rules are the precision boundary of the design. Performance optimizations must not change them.


### 3.2 Non-goals for the first upstream contribution => Roadmap

- Cross-micro-batch activation sharing.
- Context parallelism and every fused attention kernel.
- DeltaNet or other recurrent-state reuse.
- Upstreaming FSDP, Megatron, MindSpeed, and NPU support at the same time.
- Guaranteeing speedup for every batch composition or prefix ratio.
1、支持Megatron模型并行。支持NPU等设备；
2、未来支持自定义prefix-reuse关系；
3、支持其他注意力；
4、性能优化（build_kv）；
5、支持BSHD；

## 4. Current Results

The implementation and evaluation are still being consolidated. This section will be updated with the reproducible minimal report before the RFC is opened for community review.

### 4.1 Correctness

Initial FSDP and Megatron validation covers:

| Stage | Compared values | Status |
|---|---|---|
| Internal forward | Attention output and prefix/suffix boundaries | Initial validation passed |
| Training output | Logits and token log-probabilities | Initial validation passed |
| Backward | Loss, parameter gradients, and gradient norms | Initial validation passed |

The final report will add exact model and environment revisions, dtype, tolerance, maximum absolute/relative differences, and prompt-only/arbitrary-prefix/no-sharing test cases.

### 4.2 Performance

Systematic FSDP measurements are in progress. The report will compare baseline and PrefixSharing under identical configurations and include:

- no-sharing, low-sharing, and high-sharing batches;
- actor forward/backward and end-to-end step time;
- throughput and peak memory;
- prefix detection/planning and KV construction overhead.

Current profiling indicates that no-sharing planning and physical expanded-KV construction are the main optimization targets. No final speedup claim is made in this draft.

## 5. Related Works

- [PrefixGrouper](https://github.com/CASIA-IVA-Lab/PrefixGrouper) provides differentiable prompt-level sharing and the existing verl-facing integration model. This proposal extends the sharing granularity to arbitrary prefixes.
- Prefix-tree shared attention systems use flat deduplicated token layouts and sparse/custom attention masks. We view this as a complementary execution backend to KV injection.
- [Tree Training](https://arxiv.org/abs/2511.00413) studies tree-structured RL training and broader model architectures.
- [AReaL dynamic tree attention](https://github.com/areal-project/AReaL/tree/feat/dta) focuses on scalable tree execution and load balancing.
- vLLM and SGLang prefix caches optimize inference/rollout execution; PrefixSharing targets differentiable actor and reference-policy training.

## 6. TODO & Roadmap

TODO:
1. Attach a minimal report into this RFC (including precision & performance results).
2. Discuss with community developers and maintainers.
3. Revise the design based on community feedback.
4. Convert PrefixSharing's verl monkey-patches into in-tree hooks and open a PR.

Roadmap:
1. Optimize prefix-reuse performance (KV concat → FlexAttention → MagiAttention).
2. Fully support Megatron-LM, including DP, TP, PP, and CP (partially already implemented).
3. Support prefix reuse across micro-batches.
4. Support more complex attention structures (Qwen3.5 HybridAttention; DeepSeek SWA / CSA / HCA).
