from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


class _FakeDataProto:
    def __init__(self, batch: dict[str, object], meta_info: dict[str, object] | None = None):
        self.batch = batch
        self.meta_info = dict(meta_info or {})

    def __len__(self):
        return next(iter(self.batch.values())).shape[0]

    def padding(self, pad_size: int, _mode: str):
        for key, value in self.batch.items():
            self.batch[key] = torch.cat([value, value[-1:].repeat((pad_size,) + (1,) * (value.ndim - 1))])


class _FakeRolloutManager:
    def __init__(self):
        self.calls: list[_FakeDataProto] = []

    def generate_sequences(self, batch, **_kwargs):
        self.calls.append(batch)
        return _FakeDataProto(
            {
                "input_ids": batch.batch["input_ids"].clone(),
                "attention_mask": batch.batch["attention_mask"].clone(),
                "position_ids": batch.batch["position_ids"].clone(),
                "responses": torch.tensor([[9, 10]], dtype=torch.long),
                "prompts": torch.tensor([[1, 2]], dtype=torch.long),
                "response_mask": torch.ones(1, 2, dtype=torch.float32),
            },
            meta_info={"timing": {"gen": 1.0}},
        )


def _request(*, validate: bool) -> _FakeDataProto:
    return _FakeDataProto(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        },
        meta_info={"validate": True} if validate else {},
    )


def test_capture_rollout_skips_validation_and_records_first_training_output(tmp_path):
    from prefix_sharing.tools.inject_fixed_rollout import patch_capture_rollout

    rollout_manager = _FakeRolloutManager()
    capture_path = tmp_path / "rollout.json"
    patch_capture_rollout(rollout_manager, str(capture_path))

    validation_output = rollout_manager.generate_sequences(_request(validate=True))
    assert validation_output.batch["responses"].tolist() == [[9, 10]]
    assert not capture_path.exists()

    training_output = rollout_manager.generate_sequences(_request(validate=False))
    assert training_output.batch["responses"].tolist() == [[9, 10]]
    assert capture_path.exists()

    # Capture is one-shot; later training calls use the original generator.
    rollout_manager.generate_sequences(_request(validate=False))
    assert len(rollout_manager.calls) == 3

    record = json.loads(capture_path.read_text())
    assert record["outputs"]["input_ids"] == [[1, 2]]


def test_fixed_rollout_skips_validation_and_replays_training_output(monkeypatch):
    from prefix_sharing.tools import inject_fixed_rollout

    fixed_data = _FakeDataProto(
        {"input_ids": torch.tensor([[7, 8]], dtype=torch.long)},
        meta_info={"timing": {"captured": 1.0}},
    )
    monkeypatch.setattr(inject_fixed_rollout, "_load_json_to_dataproto", lambda _path: fixed_data)

    rollout_manager = _FakeRolloutManager()
    inject_fixed_rollout.patch_fixed_rollout(rollout_manager, "unused.json", num_workers=1)

    validation_output = rollout_manager.generate_sequences(_request(validate=True))
    assert validation_output.batch["input_ids"].tolist() == [[1, 2]]

    training_output = rollout_manager.generate_sequences(_request(validate=False))
    assert training_output is fixed_data
    assert training_output.meta_info["timing"] == {}
    assert len(rollout_manager.calls) == 1


def test_rollout_patch_rejects_capture_and_fixed_modes_together(monkeypatch):
    from prefix_sharing.setup.patches.verl080_fsdp.rollout_patch import _apply_rollout_env

    monkeypatch.setenv("PREFIX_SHARING_CAPTURE_ROLLOUT", "/tmp/capture.json")
    monkeypatch.setenv("PREFIX_SHARING_FIXED_ROLLOUT", "/tmp/fixed.json")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _apply_rollout_env(_FakeRolloutManager())
