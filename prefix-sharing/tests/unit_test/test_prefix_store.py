import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.core.prefix_store import (
    PREFIX_STATE_TYPE_ATTENTION_KV,
    PrefixActivationSlotId,
    PrefixActivationStore,
    PrefixAttentionStore,
)


def test_prefix_attention_store_lifecycle_and_isolation():
    store = PrefixAttentionStore()
    slot_id = PrefixActivationSlotId(1, 2, 3, 4, PREFIX_STATE_TYPE_ATTENTION_KV, 5)
    key_tensor = torch.randn(7, 2, 4, requires_grad=True)
    value_tensor = torch.randn(7, 2, 4, requires_grad=True)

    store.store(slot_id, key_tensor=key_tensor, value_tensor=value_tensor, prefix_len=7)
    entry = store.load(slot_id)

    assert entry.key_tensor is key_tensor
    assert entry.value_tensor is value_tensor
    assert entry.prefix_len == 7
    assert entry.key_tensor.requires_grad
    assert entry.value_tensor.requires_grad

    with pytest.raises(KeyError):
        store.store(
            slot_id,
            key_tensor=torch.zeros_like(key_tensor),
            value_tensor=torch.zeros_like(value_tensor),
            prefix_len=7,
        )

    other_micro_batch = PrefixActivationSlotId(1, 99, 3, 4, PREFIX_STATE_TYPE_ATTENTION_KV, 5)
    assert not store.contains(other_micro_batch)
    store.close()
    assert store.closed
    with pytest.raises(RuntimeError):
        store.load(slot_id)


def test_prefix_activation_store_base_rejects_duplicate_entries():
    store = PrefixActivationStore()
    slot_id = PrefixActivationSlotId(1, 2, 3, 4, "custom_state", 0)
    entry = object()

    store.store_entry(slot_id, entry=entry)

    assert store.load_entry(slot_id) is entry
    with pytest.raises(KeyError):
        store.store_entry(slot_id, entry=object())


def test_prefix_store_public_api_is_attention_kv_only():
    import prefix_sharing.core as core
    import prefix_sharing.core.prefix_store as prefix_store

    assert hasattr(core, "PrefixAttentionStore")
    assert hasattr(core, "StoredAttentionKV")
    assert hasattr(prefix_store, "PrefixAttentionStore")
    assert hasattr(prefix_store, "StoredAttentionKV")

    for name in (
        "PREFIX_STATE_TYPE_DELTANET_STATE",
        "PrefixDeltanetStore",
        "StoredDeltanetState",
    ):
        assert not hasattr(core, name)
        assert not hasattr(prefix_store, name)
