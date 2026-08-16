"""Boundary tests for shared verl integration helpers."""

from __future__ import annotations

import inspect

from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config


def test_runtime_state_lives_in_shared_module():
    assert PrefixSharingRuntimeState.__module__ == "prefix_sharing.integrations.runtime_state"


def test_verl_config_reader_lives_in_shared_utils():
    assert read_ps_config_from_engine_config.__module__ == "prefix_sharing.integrations.verl_utils"


def test_verl_fsdp_does_not_import_shared_helpers_from_mcore():
    import prefix_sharing.integrations.verl_fsdp as verl_fsdp

    source = inspect.getsource(verl_fsdp)
    forbidden_imports = (
        "from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState",
        "from prefix_sharing.integrations.verl_mcore import collect_kept_position_rows",
        "from prefix_sharing.integrations.verl_mcore import extract_seq_from_nested_tensor",
        "from prefix_sharing.integrations.verl_mcore import is_nested_tensor",
        "from prefix_sharing.integrations.verl_mcore import trim_redundant_prefix_in_nested_tensor",
    )
    for forbidden_import in forbidden_imports:
        assert forbidden_import not in source
