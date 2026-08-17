"""Inject synthetic prefix-sharing data, replacing generate_sequences output.

Creates batch_size sequences with nested prefix structure from a JSON token
sequence:

  Given 2*batch_size equal segments of size S:
    seg1, seg2, seg3, ..., seg_{2N}

  Nested assignment:
    seq1: prompt=[seg1]                          | response=[seg2]
    seq2: prompt=[seg1, seg2, seg3]              | response=[seg4]
    seq3: prompt=[seg1, seg2, seg3, seg4, seg5]  | response=[seg6]
    ...
    seqN: prompt=[seg1..seg_{2N-1}]              | response=[seg_{2N}]

All prompts share prefix seg1 -> enables prefix-sharing optimization.

Usage:
    USE_SYNTHETIC_PREFIX=/path/to/data.json python train_script.py
"""

import json

import torch


def _load_base_tokens(json_path: str) -> list[int]:
    """Load JSON and return the longest unpadded token sequence.

    1. Find sample with most valid tokens via position_ids (max = N-1).
    2. Slice out only valid tokens using position_ids to skip padding.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"[SyntheticPrefix] JSON not found: {json_path}")

    outputs = raw.get("outputs", raw)

    def _ensure_list(v):
        return json.loads(v) if isinstance(v, str) else v

    for k in list(outputs.keys()):
        outputs[k] = _ensure_list(outputs[k])

    ids = outputs.get("input_ids")
    if not ids:
        raise RuntimeError("[SyntheticPrefix] No 'input_ids' in JSON")
    pos = outputs.get("position_ids")
    if pos is None:
        raise RuntimeError("[SyntheticPrefix] No 'position_ids' in JSON")

    # Pick sample with longest valid sequence
    def _valid_len(p):
        return max(p) + 1 if p else 0

    best = max(range(len(ids)), key=lambda i: _valid_len(pos[i]))
    best_pos = pos[best]

    try:
        first_one = next(i for i, p in enumerate(best_pos) if p == 1)
    except StopIteration:
        raise RuntimeError("[SyntheticPrefix] Sample has no position_id=1")

    start = first_one - 1
    end = max(i for i, p in enumerate(best_pos) if p > 0) + 1
    return ids[best][start:end]


def _build_synthetic_batch(
    base_tokens: list[int],
    batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    pad_id: int = 151643,
) -> dict:
    """Create synthetic batch with nested prefix structure.

    Returns a dict of tensors ready for ``DataProto.from_dict()``.
    """
    n_seg = 2 * batch_size
    S = min(max_response_length, max_prompt_length // (n_seg - 1))

    total_needed = n_seg * S
    if len(base_tokens) < total_needed:
        raise RuntimeError(
            f"[SyntheticPrefix] Need {total_needed} tokens, have {len(base_tokens)}. "
            f"Reduce batch_size or increase max lengths."
        )

    print(f"[SyntheticPrefix] bs={batch_size} seg_size={S} total={total_needed}")
    tokens = base_tokens[:total_needed]
    segs = [tokens[i * S:(i + 1) * S] for i in range(n_seg)]

    # Build per-sample prompt/response token lists
    prompt_lists: list[list[int]] = []
    response_lists: list[list[int]] = []
    for i in range(batch_size):
        prompt_lists.append(sum(segs[:2 * i + 1], []))
        response_lists.append(segs[2 * i + 1])

    P, R, bs = max_prompt_length, max_response_length, batch_size

    # -- left-padded prompt --
    prompts = torch.full((bs, P), pad_id, dtype=torch.long)
    for i in range(bs):
        pl = len(prompt_lists[i])
        prompts[i, -pl:] = torch.tensor(prompt_lists[i], dtype=torch.long)

    # -- right-padded response --
    responses = torch.full((bs, R), pad_id, dtype=torch.long)
    for i in range(bs):
        rl = len(response_lists[i])
        responses[i, :rl] = torch.tensor(response_lists[i], dtype=torch.long)

    # -- combined fields --
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = (input_ids != pad_id).long()

    # position_ids: left-pad=0, prompt tokens=0..pl-1,
    # response tokens=pl..pl+rl-1, right-pad=0
    position_ids = torch.zeros(bs, P + R, dtype=torch.long)
    for i in range(bs):
        pl = len(prompt_lists[i])
        rl = len(response_lists[i])
        if pl > 0:
            position_ids[i, P - pl:P] = torch.arange(pl)
        if rl > 0:
            position_ids[i, P:P + rl] = torch.arange(rl) + pl

    response_mask = torch.zeros(bs, R, dtype=torch.float32)
    for i in range(bs):
        response_mask[i, :len(response_lists[i])] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "token_level_rewards": response_mask.clone(),
        "rollout_log_probs": torch.zeros(bs, R, dtype=torch.float32),
        "rm_scores": torch.ones(bs, 1, dtype=torch.float32),
    }


def patch_synthetic_prefix(
    trainer,
    json_path: str,
    batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    num_workers: int = 8,
):
    """Monkey-patch generate_sequences to return synthetic prefix data.

    Args:
        trainer: RayPPOTrainer / RayMegatronTrainer instance (self in fit()).
        json_path: Path to JSON with at least one long input_ids entry.
        batch_size: Number of sequences per batch.
        max_prompt_length: Prompt length (left-padded).
        max_response_length: Response length (right-padded).
        num_workers: Agent loop workers for chunk() divisibility.
    """
    from verl.protocol import DataProto

    base_tokens = _load_base_tokens(json_path)
    batch = _build_synthetic_batch(
        base_tokens=base_tokens,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )
    # verl>=0.8.0 trainer.fit() hard-indexes batch.non_tensor_batch["multi_modal_inputs"]
    # (ray_trainer.py:1483, even for text-only). Synthetic injection lacks this field → KeyError.
    # v070 trainer does not touch this field, so no placeholder needed.
    # Only fill an empty dict placeholder when verl>=0.8.0
    # (downstream 'image_grid_thw' check skips empty dicts, correct behavior).
    # Note: use > 0.7.99 instead of >= 0.8.0, because packaging parses "0.8.0.dev"
    # as a prerelease strictly < "0.8.0", so >= 0.8.0 would miss dev builds → KeyError.
    non_tensors = None
    try:
        import verl
        from packaging.version import parse as parse_version

        if parse_version(verl.__version__) > parse_version("0.7.99"):
            import numpy as np

            n_samples = batch["input_ids"].shape[0]
            non_tensors = {
                "multi_modal_inputs": np.array([{}] * n_samples, dtype=object)
            }
            print(
                f"[SyntheticPrefix] verl={verl.__version__}, filled empty 'multi_modal_inputs' "
                f"placeholder for {n_samples} text-only samples."
            )
    except Exception as e:  # import or version detection failed → skip placeholder
        print(f"[SyntheticPrefix] skip 'multi_modal_inputs' placeholder: {e}")

    fixed_data = DataProto.from_dict(batch, non_tensors=non_tensors)

    # Pad to be divisible by num_workers
    n = len(fixed_data)
    rem = n % num_workers
    if rem:
        pad_size = num_workers - rem
        fixed_data.padding(pad_size, "last")
        print(f"[SyntheticPrefix] Padded {n} -> {n + pad_size} (divisible by {num_workers})")

    def _patched(batch, **kwargs):
        print(
            f"[SyntheticPrefix] Returning synthetic prefix data "
            f"(bs={batch_size}, P={max_prompt_length}, R={max_response_length})."
        )
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    trainer.actor_rollout_wg.generate_sequences = _patched
    print("[SyntheticPrefix] Patched actor_rollout_wg.generate_sequences.")

    if hasattr(trainer, "async_rollout_manager") and trainer.async_rollout_manager is not None:
        trainer.async_rollout_manager.generate_sequences = _patched
        print("[SyntheticPrefix] Patched async_rollout_manager.generate_sequences.")
