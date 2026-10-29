# PrefixSharing: Eliminating Prefix Redundancies for Efficient RL Training

This is a Python module to reuse KV activations across sequence samples (or trajectories) during Forward/Backward pass of verl + Megatron-LM RL pipeline. 
Redundant KV computation and memory of common prefix sub-sequences is commonly seen in GRPO-style / Step-wise /  Tree-wise rollout, while PrefixSharing eliminates them entirely and preserves gradient semantics.

## 1. Installation

### 1.1 Install PrefixSharing

To install this module:

```bash
cd prefix-sharing && pip install -e .
```

### 1.2 Prepare Environments

This module is developed and tested on the following environment. For a first-time out-of-the-box experience, it is highly recommended to use these dependency versions:

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

### 2.1 Integrating PrefixSharing

Integrating PrefixSharing into verl + Megaton-LM pipeline is straightforward: the only operation required is to import the package inside verl:

```python
import prefix_sharing
```

This activates the patches under `prefix-sharing/setup/`, which use Python's monkey patch to dynamically modify corresponding functions in verl and Megatron-LM. `dependency/verl_cdd9014f/verl/workers/engine/megatron/transformer_impl.py:1039` provides an example.

### 2.2 Run Your First Demo!

Prepare data: download [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) from HuggingFace and convert it to parquet format following the [verl data preparation guide](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html).

Prepare model weights: download [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) from HuggingFace as usual.

Now is time to try-out PrefixSharing. To enable or disable this feature, just setup environment variable `ENABLE_PREFIX_SHARING` and run a verl training script as following:

```bash
# to enable PrefixSharing
ENABLE_PREFIX_SHARING=1 bash examples/run_prefix_sharing.sh

# to disable PrefixSharing
ENABLE_PREFIX_SHARING=0 bash examples/run_prefix_sharing.sh
```

## 5. Citation

```bibtex
@misc{prefixsharing2026,
  title={PrefixSharing: Sharing Prefix Activations for Efficient RL Training}
  author={PrefixSharing Team},
  year={2026},
  howpublished={\url{https://github.com/Jackie2049/PrefixSharing/tree/release-to-developers}},
  note={GitHub repository},
}
```

## License

MIT
