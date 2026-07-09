import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout


@pytest.mark.parametrize(
    ("tp_size", "expected_padded_lengths", "expected_cu_seqlens", "expected_positions", "expected_mask"),
    [
        (
            2,
            [6, 2],
            [0, 6, 8],
            [0, 1, 2, 3, 4, 0, 3, 4],
            [True, True, True, True, True, False, True, True],
        ),
        (
            4,
            [8, 4],
            [0, 8, 12],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False],
        ),
        (
            8,
            [8, 8],
            [0, 8, 16],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False, False, False, False, False],
        ),
    ],
)
def test_packed_batch_layout_aligns_valid_rows_for_common_tensor_parallel_sizes(
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
    expected_positions,
    expected_mask,
):
    layout = PackedBatchLayout.from_kept_position_rows(
        [
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([3, 4]),
        ],
        align_size=tp_size,
    )

    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.max_seqlen == max(expected_padded_lengths)
    assert layout.total_valid_length == 7
    assert layout.total_padded_length == expected_cu_seqlens[-1]
    assert layout.packed_position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    assert layout.packed_index(1, 0) == expected_cu_seqlens[1]


def test_packed_batch_layout_rejects_padding_slot_as_valid_index():
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.tensor([0, 1, 2])],
        align_size=2,
    )

    with pytest.raises(IndexError):
        layout.packed_index(0, 3)


def test_packed_batch_layout_builds_context_parallel_local_view():
    layout = PackedBatchLayout.from_kept_position_rows(
        [
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([3, 4]),
        ],
        align_size=8,
        cp_rank=1,
        cp_size=2,
    )

    cp = layout.context_parallel
    assert cp is not None
    assert cp.local_padded_lengths == [4, 4]
    assert cp.local_cu_seqlens == [0, 4, 8]
    assert cp.local_total_padded_length == 8
    # row0 padded global [0..7], cp_rank=1 gets front [2,3] and back [4,5].
    # row1 padded global [8..15], cp_rank=1 gets front [10,11] and back [12,13].
    assert cp.global_indices_by_local.tolist() == [2, 3, 4, 5, 10, 11, 12, 13]
    assert cp.local_position_ids.tolist() == [2, 3, 4, 0, 0, 0, 0, 0]
    assert cp.local_valid_token_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert cp.global_to_local(4) == 2
    assert cp.global_to_local(1) is None
    assert cp.local_to_global(2) == 4
    assert cp.row_local_slice(1) == slice(4, 8)
    assert cp.row_local_valid_mask(0).tolist() == [True, True, True, False]


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_packed_batch_layout_context_parallel_common_sizes(cp_size):
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.arange(17), torch.arange(9)],
        align_size=2 * cp_size,
        cp_rank=cp_size - 1,
        cp_size=cp_size,
    )

    cp = layout.context_parallel
    assert cp is not None
    assert cp.local_total_padded_length == layout.total_padded_length // cp_size
    assert len(cp.global_indices_by_local) == cp.local_total_padded_length
