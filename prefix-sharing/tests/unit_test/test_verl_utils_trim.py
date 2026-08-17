"""Tests for nested-tensor trim helpers in verl_utils."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.verl_utils import trim_redundant_prefix_in_dense_tensor
from prefix_sharing.integrations.verl_utils import trim_redundant_prefix_in_nested_tensor


def _prefix_sharing_plan():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    return planner.plan([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])


def _nested(rows):
    return torch.nested.as_nested_tensor(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        layout=torch.jagged,
    )


def _unpacked_rows(nested_tensor):
    offsets = nested_tensor.offsets()
    values = nested_tensor.values()
    return [
        values[offsets[i] : offsets[i + 1]].tolist()
        for i in range(offsets.numel() - 1)
    ]


def test_trim_redundant_prefix_returns_kept_position_rows_from_nested_slice():
    if not hasattr(torch, "nested"):
        pytest.skip("torch.nested is unavailable")

    plan = _prefix_sharing_plan()
    batch = {
        "input_ids": _nested([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]),
        "position_ids": _nested([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]),
        "loss_mask": _nested([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]]),
    }

    trimmed_batch, kept_position_rows = trim_redundant_prefix_in_nested_tensor(batch, plan)

    assert [row.tolist() for row in kept_position_rows] == [[0, 1, 2, 3, 4], [3, 4, 5]]
    assert _unpacked_rows(trimmed_batch["position_ids"]) == [
        row.tolist() for row in kept_position_rows
    ]
    assert _unpacked_rows(trimmed_batch["input_ids"]) == [[1, 2, 3, 10, 11], [20, 21, 22]]
    assert _unpacked_rows(trimmed_batch["loss_mask"]) == [[1, 1, 1, 1, 0], [1, 1, 0]]


def test_trim_redundant_prefix_returns_kept_position_rows_from_2d_slice():
    if not hasattr(torch, "nested"):
        pytest.skip("torch.nested is unavailable")

    plan = _prefix_sharing_plan()
    batch = {
        "input_ids": _nested([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    }

    trimmed_batch, kept_position_rows = trim_redundant_prefix_in_nested_tensor(batch, plan)

    assert [row.tolist() for row in kept_position_rows] == [[0, 1, 2, 3, 4], [3, 4, 5]]
    assert _unpacked_rows(trimmed_batch["position_ids"]) == [
        row.tolist() for row in kept_position_rows
    ]


def test_trim_redundant_prefix_in_dense_tensor_crops_masks_and_returns_position_rows():
    if not hasattr(torch, "nested"):
        pytest.skip("torch.nested is unavailable")

    plan = _prefix_sharing_plan()
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 10, 11, 0],
                [1, 2, 3, 20, 21, 22],
            ],
            dtype=torch.long,
        ),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "loss_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    }
    valid_indices = [
        batch["attention_mask"][row].nonzero(as_tuple=False).flatten()
        for row in range(batch["input_ids"].shape[0])
    ]

    trimmed_batch, kept_position_rows = trim_redundant_prefix_in_dense_tensor(
        batch, plan, valid_indices
    )

    assert [row.tolist() for row in kept_position_rows] == [[0, 1, 2, 3, 4], [3, 4, 5]]
    assert trimmed_batch["position_ids"] is batch["position_ids"]
    assert trimmed_batch["position_ids"][1, 3:6].tolist() == [3, 4, 5]
    assert trimmed_batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]
    assert trimmed_batch["loss_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]
    assert trimmed_batch["input_ids"] is batch["input_ids"]
