# PrefixSharing

This is a Python module to reuse prefix KV activations across sequence samples
or trajectories during verl RL training. The open-source mainline currently
prioritizes the verl 0.8.0 FSDP path and exposes arbitrary-prefix sharing as an
extension of verl's PrefixGrouper-style user entry.

Redundant prefix computation is common in GRPO-style, step-wise, and tree-wise
rollout. PrefixSharing reduces that duplicated attention KV work while
preserving logprob / loss / gradient semantics.

## 1. Installation

### 1.1 Install PrefixSharing

To install this module:

```bash
cd prefix-sharing && pip install -e .
```

### 1.2 Prepare Environments

This module is developed and tested on the following environment. For a
first-time out-of-the-box experience, it is recommended to use the vendored
verl 0.8.0 dependency snapshot and Qwen2.5-0.5B.

| Dependency       | Version    |
|------------------|------------|
| verl             | cdd9014f   |
| Megatron-LM core | v0.16.1    |
| MindSpeed core   | r0.16.0    |
| Megatron-Bridge  | de93536e   |

The FSDP path only requires the verl / Transformers side at runtime. Megatron,
MindSpeed, and Megatron-Bridge are kept for advanced Megatron/MCore paths.

Besides installing the above environment using pip or other installation tools,
users can also install from source code under `dependency/`, where above version
snapshots are stored.

```bash
cd dependency/Megatron-Bridge_de93536e   && pip install --no-deps -v -e .
cd dependency/Megatron-LM-core_v0.16.1   && pip install --no-deps -v -e .
cd dependency/MindSpeed_core_r0.16.0     && pip install --no-deps -v -e .
cd dependency/verl_cdd9014f              && pip install --no-deps -v -e .
```

## 2. Quick Start

### 2.1 PrefixSharing and PrefixGrouper

From the verl user's perspective, PrefixSharing is positioned as an
`arbitrary_prefix` mode under the existing PrefixGrouper-style feature entry:

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
      min_prefix_len: 32
      min_group_size: 2
```

`prompt_only` remains the PrefixGrouper baseline. `arbitrary_prefix` enters
PrefixSharing's provider/reuser planner, KV injection, and restore runtime.
PrefixSharing does not vendor or reimplement PrefixGrouper's prompt-only
algorithm.

### 2.2 Integrating PrefixSharing

Integrating PrefixSharing into verl is done through the setup patch entry. For
explicit installation:

```python
import prefix_sharing

prefix_sharing.setup.install("verl080_fsdp")
```

For existing verl external-module workflows, importing the package can also
auto-install the patch set:

```python
import prefix_sharing
```

When no explicit patch set is provided, the compatibility matrix prefers the
`verl080_fsdp` patch set for verl 0.8.0 environments. Megatron/MCore patch sets
should be selected explicitly when needed.

### 2.3 Run Your First Demo

Prepare data: download [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) from HuggingFace and convert it to parquet format following the [verl data preparation guide](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

Prepare model weights: download [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) from HuggingFace as usual.

Enable PrefixSharing through verl config:

```yaml
actor_rollout_ref:
  actor:
    use_prefix_grouper: true
    prefix_grouper:
      mode: arbitrary_prefix
```

Then run the verl training script:

```bash
bash examples/run_prefix_sharing.sh
```

For local debugging, `ENABLE_PREFIX_SHARING` remains available as a fallback
runtime switch:

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
