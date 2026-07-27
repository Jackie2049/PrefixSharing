"""Capture or inject fixed rollout data at a trainer rollout boundary.

The patch receives the concrete rollout object used by the trainer:

    from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout
    patch_fixed_rollout(self.async_rollout_manager, json_path="/path/to/your_data.json")

The JSON format expected:
{
    "outputs": {
        "input_ids":  [[...], [...], ...],
        "attention_mask": [[...], ...],
        "position_ids": [[...], ...],
        "responses": [[...], ...],
        "prompts": [[...], ...],
        "token_level_rewards": [[...], ...],
        "response_mask": [[...], ...],
        "rm_scores": [[...], ...],
        "rollout_log_probs": [[...], ...]
    }
}
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


def read_rollout_replay_paths() -> tuple[str | None, str | None]:
    """Read capture/replay env vars and reject an ambiguous trainer setup.

    The two modes wrap the same ``generate_sequences`` boundary.  Installing
    both wrappers makes the effective behavior depend on wrapper order, so a
    run must choose exactly one mode.
    """
    capture_path = os.environ.get("PREFIX_SHARING_CAPTURE_ROLLOUT", "").strip() or None
    fixed_path = os.environ.get("PREFIX_SHARING_FIXED_ROLLOUT", "").strip() or None
    if capture_path and fixed_path:
        raise ValueError(
            "PREFIX_SHARING_CAPTURE_ROLLOUT and PREFIX_SHARING_FIXED_ROLLOUT "
            "are mutually exclusive"
        )
    return capture_path, fixed_path


def _load_json_to_dataproto(json_path: str):
    """Load a JSON file and convert to DataProto."""
    from verl.protocol import DataProto

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"[FixedRollout] JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[FixedRollout] Invalid JSON in {json_path}: {e}")

    if "outputs" not in raw:
        raise RuntimeError(f"[FixedRollout] Missing key 'outputs' in {json_path}")

    outputs = raw["outputs"]

    # Values may be JSON strings like "[[1,2],[3,4]]" instead of actual lists
    def _ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)
        return val

    outputs = {k: _ensure_list(v) for k, v in outputs.items()}

    def _pad_long(seqs, pad_id=0):
        """Pad list of int lists to rectangular LongTensor."""
        max_len = max(len(s) for s in seqs)
        tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return tensor

    def _pad_float(seqs, pad_val=0.0):
        """Pad list of float lists to rectangular FloatTensor."""
        max_len = max(len(s) for s in seqs)
        tensor = torch.full((len(seqs), max_len), pad_val, dtype=torch.float32)
        for i, s in enumerate(seqs):
            tensor[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
        return tensor

    batch = {
        "input_ids": _pad_long(outputs["input_ids"]),
        "attention_mask": _pad_long(outputs["attention_mask"]),
        "position_ids": _pad_long(outputs["position_ids"]),
        "responses": _pad_long(outputs["responses"]),
        "prompts": _pad_long(outputs["prompts"]),
    }

    # Optional float fields
    for key in ("token_level_rewards", "response_mask", "rm_scores", "rollout_log_probs"):
        if key in outputs:
            batch[key] = _pad_float(outputs[key])

    # Build sequences = prompts + responses if not present
    if "sequences" not in outputs:
        batch["sequences"] = torch.cat([batch["prompts"], batch["responses"]], dim=1)

    # verl>=0.8.0 的 trainer.fit() 会硬索引 batch.non_tensor_batch["multi_modal_inputs"]
    # （ray_trainer.py:1483，纯文本场景也走这行）。纯文本/虚拟注入没有这个字段会 KeyError。
    # v070 trainer 不碰这个字段，无需添加。这里只在 verl>=0.8.0 时填一个空字典占位
    # （下游 'image_grid_thw' 检查会对空 dict continue 跳过，行为正确）。
    # 注意：用 > 0.7.99 而不是 >= 0.8.0，因为 packaging 解析下 "0.8.0.dev" 是
    # prerelease，严格 < "0.8.0"，直接用 >= 0.8.0 会让 dev 版本漏掉导致 KeyError。
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
                f"[FixedRollout] verl={verl.__version__}, filled empty 'multi_modal_inputs' "
                f"placeholder for {n_samples} text-only samples."
            )
    except Exception as e:  # import 失败或版本探测失败，退回到不填占位
        print(f"[FixedRollout] skip 'multi_modal_inputs' placeholder: {e}")

    data = DataProto.from_dict(batch, non_tensors=non_tensors)
    print(f"[FixedRollout] Loaded {data.batch['input_ids'].shape[0]} samples from {json_path}")
    return data


def _save_dataproto_to_json(data: Any, json_path: str) -> None:
    """Save a DataProto's batch tensors as a JSON file for replay."""
    import numpy as np

    batch = data.batch
    outputs = {}
    for key in ("input_ids", "attention_mask", "position_ids",
                "responses", "prompts", "response_mask",
                "token_level_rewards", "rm_scores", "rollout_log_probs"):
        if key in batch:
            tensor = batch[key]
            if tensor.dtype == torch.float32:
                outputs[key] = tensor.float().cpu().tolist()
            else:
                outputs[key] = tensor.long().cpu().tolist()

    record = {"outputs": outputs}
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    print(f"[FixedRollout] Captured rollout with {len(data)} samples → {json_path}")


def _is_validation_rollout(batch: Any) -> bool:
    return bool(getattr(batch, "meta_info", {}).get("validate", False))


def patch_capture_rollout(rollout_obj: Any, json_path: str) -> None:
    """Intercept ``generate_sequences`` on *rollout_obj*, capture first training rollout as JSON.

    A normal PS=OFF run both creates the baseline dump and captures rollout
    output. Replaying it in the PS=ON run avoids a third PS=OFF replay run.
    Validation rollouts are left untouched.
    """
    original_fn = rollout_obj.generate_sequences
    captured = [False]

    def _capturing_patch(batch, **kwargs):
        result = original_fn(batch, **kwargs)
        if not captured[0] and not _is_validation_rollout(batch):
            _save_dataproto_to_json(result, json_path)
            captured[0] = True
            rollout_obj.generate_sequences = original_fn
            print("[FixedRollout] Capture complete — generate_sequences restored.")
        return result

    rollout_obj.generate_sequences = _capturing_patch
    print(f"[FixedRollout] Patched generate_sequences for capture → {json_path}")


def patch_fixed_rollout(rollout_obj: Any, json_path: str):
    """Monkey-patch ``generate_sequences`` on *rollout_obj* to return fixed data.

    Args:
        rollout_obj: Object with a ``generate_sequences(batch) -> DataProto`` method
                     (e.g. ``AgentLoopManager`` or trainer.actor_rollout_wg).
        json_path: Absolute path to the JSON file.
    """
    fixed_data = _load_json_to_dataproto(json_path)

    batch = dict(fixed_data.batch)
    meta_info = dict(fixed_data.meta_info)
    n_orig = len(fixed_data)

    import torch as _torch
    import numpy as _np

    # ── 1. Select first N sequences ──
    _num_seq = int(os.environ.get("PREFIX_SHARING_BASELINE_NUM_SEQ", "0"))
    if _num_seq > 0 and n_orig > _num_seq:
        for _key in batch:
            if isinstance(batch[_key], _torch.Tensor) and batch[_key].shape[0] == n_orig:
                batch[_key] = batch[_key][:_num_seq]
        n_orig = _num_seq
        print(f"[FixedRollout] Selected {_num_seq} sequences")

    # ── 2. Randomize reward scores (debug only) ──
    _torch.manual_seed(42)
    for _key in ("token_level_rewards", "rm_scores"):
        _rm = batch.get(_key)
        if _rm is not None:
            _rm[...] = _torch.randint(0, 2, _rm.shape, dtype=_rm.dtype)

    # ── 3. Stack (tile dim-0) ──
    _stack = int(os.environ.get("PREFIX_SHARING_BASELINE_STACK", "1"))
    if _stack > 1:
        for _key in batch:
            if isinstance(batch[_key], _torch.Tensor) and batch[_key].shape[0] == n_orig:
                batch[_key] = batch[_key].repeat(_stack, *([1] * (batch[_key].dim() - 1)))
        n_orig *= _stack
        print(f"[FixedRollout] Stacked batch x{_stack}: {n_orig} sequences")

    # ── 4. Rebuild DataProto ──
    from verl.protocol import DataProto
    fixed_data = DataProto.from_dict(
        batch,
        non_tensors={"multi_modal_inputs": _np.array([{}] * n_orig, dtype=object)},
    )
    fixed_data.meta_info = meta_info

    original_generate_sequences = rollout_obj.generate_sequences

    def replay_training_rollout(batch: Any, **kwargs: Any) -> Any:
        if _is_validation_rollout(batch):
            return original_generate_sequences(batch, **kwargs)
        print("[FixedRollout] Returning fixed rollout data, skipping generation.")
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    rollout_obj.generate_sequences = replay_training_rollout
    print(f"[FixedRollout] Replay enabled <- {json_path}")
