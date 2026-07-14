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
from typing import Any, Optional

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
    """Save replay-relevant rollout tensors as a JSON fixture."""
    outputs = {}
    for key in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "prompts",
        "response_mask",
        "token_level_rewards",
        "rm_scores",
        "rollout_log_probs",
    ):
        if key not in data.batch:
            continue
        tensor = data.batch[key]
        outputs[key] = tensor.float().cpu().tolist() if tensor.dtype == torch.float32 else tensor.long().cpu().tolist()

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"outputs": outputs}, file)
    print(f"[FixedRollout] Captured rollout with {len(data)} samples -> {json_path}")


def _is_validation_rollout(batch: Any) -> bool:
    return bool(getattr(batch, "meta_info", {}).get("validate", False))


def patch_capture_rollout(rollout_obj: Any, json_path: str) -> None:
    """Capture the first training rollout while leaving validation untouched.

    A normal PS=OFF run both creates the baseline dump and captures rollout
    output. Replaying it in the PS=ON run avoids a third PS=OFF replay run.
    """
    original_generate_sequences = rollout_obj.generate_sequences
    captured = False

    def capture_training_rollout(batch: Any, **kwargs: Any) -> Any:
        nonlocal captured
        result = original_generate_sequences(batch, **kwargs)
        if not captured and not _is_validation_rollout(batch):
            _save_dataproto_to_json(result, json_path)
            captured = True
            rollout_obj.generate_sequences = original_generate_sequences
            print("[FixedRollout] Capture complete; restored generate_sequences.")
        return result

    rollout_obj.generate_sequences = capture_training_rollout
    print(f"[FixedRollout] Capture enabled -> {json_path}")


def patch_fixed_rollout(rollout_obj: Any, json_path: str, num_workers: Optional[int] = None) -> None:
    """Inject fixed rollout data for training calls and preserve validation calls.

    Args:
        rollout_obj: Object exposing ``generate_sequences``.
        json_path: Absolute path to the JSON file.
        num_workers: Optional legacy padding divisor for manually prepared JSON.
            Captured fixtures should leave this unset because their original
            batch size is already valid for the rollout path that produced it.
    """
    fixed_data = _load_json_to_dataproto(json_path)

    if num_workers is not None:
        n = len(fixed_data)
        remainder = n % num_workers
        if remainder != 0:
            pad_size = num_workers - remainder
            fixed_data.padding(pad_size, "last")
            print(f"[FixedRollout] Padded from {n} to {n + pad_size} samples (divisible by {num_workers}).")

    original_generate_sequences = rollout_obj.generate_sequences

    def replay_training_rollout(batch: Any, **kwargs: Any) -> Any:
        if _is_validation_rollout(batch):
            return original_generate_sequences(batch, **kwargs)
        print("[FixedRollout] Returning fixed rollout data, skipping generation.")
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    rollout_obj.generate_sequences = replay_training_rollout
    print(f"[FixedRollout] Replay enabled <- {json_path}")
