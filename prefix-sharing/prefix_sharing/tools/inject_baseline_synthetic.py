"""Inject synthetic data for GEMM precision baseline — reuse nested-prefix
batch builder, then shuffle sequence order and optionally stack.

Usage::

    PREFIX_SHARING_BASELINE_SYNTHETIC=/path/to/data.json
    PREFIX_SHARING_BASELINE_NUM_SEQ=4      # gen_batch_size (sequences to build)
    PREFIX_SHARING_BASELINE_STACK=3        # stack the shuffled batch N times
    PREFIX_SHARING_BASELINE_SEED=42        # shuffle seed

    # Single-copy:  stack=1  →  4 sequences (no stacking)
    # Multi-copy:   stack=3  → 12 sequences (4 shuffled × 3 stacked)
"""

import json
import os
import random

import torch


def patch_baseline_synthetic(
    trainer,
    json_path: str,
    max_prompt_length: int,
    max_response_length: int,
    num_seq: int = 1,
    stack: int = 1,
    seed: int = 42,
    num_workers: int = 8,
):
    """Monkey-patch generate_sequences to return baseline synthetic data.

    Builds ``num_seq`` sequences via the existing :func:`_build_synthetic_batch`,
    deterministically shuffles their order, then stacks the shuffled batch
    ``stack`` times.

    Args:
        trainer: RayPPOTrainer instance.
        json_path: Path to JSON with input_ids.
        max_prompt_length: Prompt length.
        max_response_length: Response length.
        num_seq: Number of distinct sequences (env: BASELINE_NUM_SEQ).
        stack: How many times to tile the shuffled batch (env: BASELINE_STACK).
        seed: Shuffle seed (env: BASELINE_SEED).
        num_workers: Agent loop workers for chunk() divisibility.
    """
    from prefix_sharing.tools.inject_synthetic_prefix import _build_synthetic_batch, _load_base_tokens
    from verl.protocol import DataProto

    base_tokens = _load_base_tokens(json_path)

    # ── 1. Build the batch using the existing nested-prefix builder ──
    batch = _build_synthetic_batch(
        base_tokens=base_tokens,
        batch_size=num_seq,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )

    # ── 2. Shuffle sequence order (deterministic) ──
    rng = random.Random(seed)
    n = batch["input_ids"].shape[0]
    idx = list(range(n))
    rng.shuffle(idx)
    for k in batch:
        if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == n:
            batch[k] = batch[k][idx]

    # ── 3. Stack (tile) the shuffled batch ──
    if stack > 1:
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == n:
                batch[k] = batch[k].repeat(stack, *([1] * (batch[k].dim() - 1)))

    total_bs = n * stack
    print(
        f"[BaselineSynthetic] num_seq={num_seq} stack={stack} total_bs={total_bs}"
        f" P={max_prompt_length} R={max_response_length} seed={seed}"
        f" shuffle_idx={idx}"
    )

    # ── 4. Multi-modal placeholder (verl >= 0.8.0) ──
    non_tensors = None
    try:
        import verl
        from packaging.version import parse as parse_version

        if parse_version(verl.__version__) > parse_version("0.7.99"):
            import numpy as np
            non_tensors = {"multi_modal_inputs": np.array([{}] * total_bs, dtype=object)}
    except Exception:
        pass

    fixed_data = DataProto.from_dict(batch, non_tensors=non_tensors)

    # Pad to be divisible by num_workers
    rem = len(fixed_data) % num_workers
    if rem:
        fixed_data.padding(num_workers - rem, "last")
        print(f"[BaselineSynthetic] Padded {len(fixed_data) - (num_workers - rem)} -> {len(fixed_data)}")

    def _patched(batch, **kwargs):
        print(
            f"[BaselineSynthetic] Returning synthetic baseline data "
            f"(num_seq={num_seq}, stack={stack}, total_bs={total_bs}, "
            f"P={max_prompt_length}, R={max_response_length}, seed={seed})."
        )
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    trainer.actor_rollout_wg.generate_sequences = _patched
    print("[BaselineSynthetic] Patched actor_rollout_wg.generate_sequences.")

    if hasattr(trainer, "async_rollout_manager") and trainer.async_rollout_manager is not None:
        trainer.async_rollout_manager.generate_sequences = _patched
        print("[BaselineSynthetic] Patched async_rollout_manager.generate_sequences.")
