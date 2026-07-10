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
    # 基本情况：多个序列有共同前缀
    assert common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    # 无共同前缀
    assert common_prefix_len([[1], [2]]) == 0
    # 空列表
    assert common_prefix_len([]) == 0
    # 只有一个序列：整个序列都是前缀
    assert common_prefix_len([[1, 2, 3]]) == 3
    # 所有序列完全相同
    assert common_prefix_len([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) == 3
    # 前缀长度等于最短序列长度
    assert common_prefix_len([[1, 2], [1, 2, 3], [1, 2, 4, 5]]) == 2
    # 较长前缀
    assert common_prefix_len([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 7, 8]]) == 4
    # 不同长度的序列，只有第一个元素相同
    assert common_prefix_len([[1], [1, 2], [1, 2, 3]]) == 1
    # 嵌套列表作为单个 token-like 元素处理，不做 flatten
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
