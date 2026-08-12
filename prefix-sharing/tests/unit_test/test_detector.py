from prefix_sharing.core.prefix_detector import PrefixDetector, TriePrefixDetector, common_prefix_len


def test_prefix_detector_is_abstract():
    """PrefixDetector cannot be instantiated directly."""
    try:
        PrefixDetector()
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "abstract" in str(e).lower()


def test_trie_detector_is_instance_of_abstract():
    """TriePrefixDetector is a concrete implementation of PrefixDetector."""
    detector = TriePrefixDetector()
    assert isinstance(detector, PrefixDetector)


def test_common_prefix_len():
    # Basic case: multiple sequences share a common prefix
    assert common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    # No common prefix
    assert common_prefix_len([[1], [2]]) == 0
    # Empty list
    assert common_prefix_len([]) == 0
    # Single sequence: entire sequence is the prefix
    assert common_prefix_len([[1, 2, 3]]) == 3
    # All sequences are identical
    assert common_prefix_len([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == 3
    # Prefix length equals the shortest sequence length
    assert common_prefix_len([[1, 2], [1, 2, 3], [1, 2, 4, 5]]) == 2
    # Longer prefix
    assert common_prefix_len([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 7, 8]]) == 4
    # Sequences of different lengths, only first element matches
    assert common_prefix_len([[1], [1, 2], [1, 2, 3]]) == 1
    # Nested lists are treated as single token-like elements, not flattened
    assert common_prefix_len([[[1, 2], 3], [[1, 2], 4], [[1, 2], 5, 6]]) == 1


def test_trie_detector_builds_per_sample_reuse_relations():
    detector = TriePrefixDetector(min_prefix_len=2, min_group_size=2)
    result = detector.detect(
        [
            [1, 2, 3, 4, 5, 10],
            [1, 2, 3, 20],
            [1, 2, 3, 4, 5, 30],
            [7, 8, 40],
            [7, 8, 9, 50],
        ]
    )

    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in result.reuse_specs] == [
        (1, 0, 3),
        (2, 0, 5),
        (4, 3, 2),
    ]
    assert result.provider_index == (0, 0, 0, 3, 3)
    assert result.prefix_lens == (0, 3, 5, 0, 2)
    assert result.is_provider == (True, False, False, True, False)


def test_trie_detector_allows_reuser_to_provide_longer_prefix_later():
    detector = TriePrefixDetector(min_prefix_len=2, min_group_size=2)
    result = detector.detect(
        [
            [1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5],
        ]
    )

    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in result.reuse_specs] == [
        (1, 0, 3),
        (2, 1, 4),
    ]
    assert result.provider_index == (0, 0, 1)
    assert result.prefix_lens == (0, 3, 4)


def test_trie_detector_respects_min_group_size_for_relation_threshold():
    detector = TriePrefixDetector(min_prefix_len=2, min_group_size=3)
    result = detector.detect(
        [
            [1, 2, 10],
            [1, 2, 20],
            [1, 2, 30],
        ]
    )

    assert [(s.reuse_idx_in_batch, s.provider_idx_in_batch, s.prefix_len) for s in result.reuse_specs] == [
        (2, 0, 2),
    ]
    assert result.provider_index == (0, 1, 0)
    assert result.prefix_lens == (0, 0, 2)
