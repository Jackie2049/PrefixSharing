"""Framework patch integrations."""

from prefix_sharing.integrations.context import (
    PackedPrefixLastRestoreIndex,
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_runtime_context,
)
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo, get_megatron_parallel_info
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.integrations.runtime_state import PrefixSharingRuntimeState
from prefix_sharing.integrations.verl_utils import read_ps_config_from_engine_config
from prefix_sharing.integrations.verl_mcore import (
    build_prefix_sharing_micro_batch_verl080,
    restore_reuser_prefix_columns_2d,
)
from prefix_sharing.integrations.verl_fsdp import (
    PrefixSharingFSDPAttentionRuntime,
    prepare_for_prefix_sharing_fsdp,
    forward_prefix_sharing_micro_batch_fsdp,
    restore_prefix_sharing_outputs_2d,
)
from prefix_sharing.integrations.megatron_runtime import (
    prefix_attention,
)

__all__ = [
    "PackedBatchLayout",
    "PackedPrefixLastRestoreIndex",
    "MegatronParallelInfo",
    "PrefixSharingRuntimeContext",
    "PrefixSharingRuntimeState",
    "current_prefix_sharing_context",
    "prefix_sharing_runtime_context",
    "build_prefix_sharing_micro_batch_verl080",
    "restore_reuser_prefix_columns_2d",
    "read_ps_config_from_engine_config",
    "prefix_attention",
    "get_megatron_parallel_info",
    "PrefixSharingFSDPAttentionRuntime",
    "prepare_for_prefix_sharing_fsdp",
    "forward_prefix_sharing_micro_batch_fsdp",
    "restore_prefix_sharing_outputs_2d",
]
