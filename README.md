# PrefixSharing

This is a Python module to reuse KV activations across sequence samples (also agent trajectories) during Forward/Backward pass in verl RL pipeline. Redundant KV computation and memory of common prefix sub-sequences is commonly seen in GRPO-style / Step-wise /  Tree-wise rollout, while PrefixSharing eliminates them entirely and preserves gradient semantics.

PrefixSharing currently supports verl 0.8.0 with FSDP (recommended) or Megatron-LM as engine backends. Compared with PrefixGrouper (which is already incorporated in verl 0.8.0), this feature extends prefix reuse to sub-sequences with arbitrary lengths, not just limited to prompts! This is realized via prefix tree algorithm and KV reuse within micro-batch. Most importantly, it inherits and extends PrefixGrouper-style configuration fields and user entries, limiting modifications in verl to a minimal scope.

## 1. Installation

### 1.1 Install PrefixSharing

To install this module:

```bash
cd prefix-sharing && pip install -e .
```

### 1.2 Prepare Environments

This module is developed and tested on the following environment. For a first-time out-of-the-box experience, it is highly recommended to use these dependency versions:

verl + FSDP pipeline (recommended):

| Dependency       | Version    |
|------------------|------------|
| verl             | cdd9014f   |
| torch            | 2.4        |
| Megatron-Bridge  | de93536e   |

verl + Megatron-LM pipeline:

| Dependency       | Version    |
|------------------|------------|
| verl             | cdd9014f   |
| Megatron-LM core | v0.16.1    |
| MindSpeed core   | r0.16.0    |
| Megatron-Bridge  | de93536e   |

Depite from installing the above environment using pip or other installation tools, users can also install from source code under `dependency/`, where above version snapshots are stored.

```bash
cd dependency/Megatron-Bridge_de93536e   && pip install --no-deps -v -e .
cd dependency/Megatron-LM-core_v0.16.1   && pip install --no-deps -v -e .
cd dependency/MindSpeed_core_r0.16.0     && pip install --no-deps -v -e .
cd dependency/verl_cdd9014f              && pip install --no-deps -v -e .
```

## 2. Quick Start

### 2.1 Configuring PrefixSharing

PrefixSharing inherits and extends PrefixGrouper-style configuration fields and user entries:

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix # turn on PrefixSharing for arbitrary prefix reuse
      min_prefix_len: 8      # minimum prefix length to enable reuse
      min_group_size: 2      # minimum samples to reuse common prefix
```

`prompt_only` remains the basic PrefixGrouper algorithm. `arbitrary_prefix` enters
PrefixSharing's prefix tree algorithm for arbitrary prefix reuse.

### 2.2 Integrating PrefixSharing

Integrating PrefixSharing into verl pipeline is straightforward: import the package inside verl and setup patches will be installed implicitly. By default, PrefixSharing detects the installed training stack and installs all compatible patch sets, so an environment that supports both FSDP and Megatron-LM receives both patches.

Default integration:

```python
import prefix_sharing
```

For debugging or narrowing the patch scope, set `PREFIX_SHARING_PATCHSET` before importing PrefixSharing:

```bash
PREFIX_SHARING_PATCHSET=verl080_fsdp python your_verl_entry.py
PREFIX_SHARING_PATCHSET=verl080_fsdp,verl080_mcore0161_ms0160 python your_verl_entry.py
```

Programmatic `prefix_sharing.setup.install(...)` remains available for controlled environments that do not rely on import-time auto activation. This activates the patches under `prefix-sharing/setup/`, which use Python's monkey patch to dynamically modify corresponding functions. `dependency/verl_cdd9014f/verl/workers/engine/megatron/transformer_impl.py:1039` provides an example.

### 2.3 Run Your First Demo

Prepare data: download [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) from HuggingFace and convert it to parquet format following the [verl data preparation guide](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

Prepare model weights: download [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) from HuggingFace as usual.

Now is time to try-out PrefixSharing. To enable this feature, first setup YaML configuration files in verl:

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
```

Then run the verl training script as following:

```bash
bash examples/run_prefix_sharing.sh
```

For local debugging, environment variable `ENABLE_PREFIX_SHARING` is available for a quick runtime switch:

```bash
ENABLE_PREFIX_SHARING=1 bash examples/run_prefix_sharing.sh
ENABLE_PREFIX_SHARING=0 bash examples/run_prefix_sharing.sh
```

## 5. Citation

```bibtex
@misc{prefixsharing2026,
  title={PrefixSharing: Sharing Prefix Activations for Efficient RL Training}
  author={PrefixSharing Team},
  year={2026},
  howpublished={\url{https://github.com/your-org/PrefixSharing}},
  note={GitHub repository},
}
```

## License

MIT
