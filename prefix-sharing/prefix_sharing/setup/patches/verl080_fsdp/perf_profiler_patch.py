"""Performance profiling patches for verl 0.8.0 FSDP.

These patches replace the direct source edits in:
  - verl.workers.engine.base.BaseEngine.train_batch
  - verl.workers.engine.fsdp.transformer_impl.FSDPEngine.forward_backward_batch
  - verl.workers.engine_workers.TrainingWorker.train_mini_batch / infer_batch

They are installed by ``prefix_sharing.setup.patches.verl080_fsdp`` when the
perf-profiler patch set is loaded.  When ``PREFIX_SHARING_PERF_PROFILE`` is not
set, the patches are no-ops and behave exactly like the original methods.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

from prefix_sharing.tools.perf_profiler import ProfilerScope


# verl.single_controller.base.decorator.MAGIC_ATTR.
# The @register decorator stores Ray dispatch metadata on the wrapper function
# under this attribute.  When monkey-patching a registered method we must copy
# it to the new wrapper, otherwise verl's single-controller will not recognise
# the method as a remote callable.
_RAY_REGISTER_MAGIC_ATTR = "attrs_3141562937"


def _copy_ray_register_attrs(original: Callable, wrapped: Callable) -> None:
    """Copy verl's @register MAGIC_ATTR from ``original`` to ``wrapped``."""
    attrs = getattr(original, _RAY_REGISTER_MAGIC_ATTR, None)
    if attrs is not None:
        setattr(wrapped, _RAY_REGISTER_MAGIC_ATTR, attrs)


def _patch_train_batch(original: Callable) -> Callable:
    """Wrap BaseEngine.train_batch to time optimizer_step as PHASE_UPDATE.

    Instead of editing verl source, we temporarily replace ``self.optimizer_step``
    with a wrapper that starts/stops the ``update`` phase while a step-level
    ProfilerScope is active.
    """

    def wrapped(self: Any, data: Any, loss_function: Any) -> Any:
        scope = ProfilerScope.current()
        if scope is None:
            return original(self, data, loss_function)

        orig_optimizer_step = self.optimizer_step

        def _timed_optimizer_step() -> Any:
            scope.start_phase(scope.PHASE_UPDATE)
            try:
                return orig_optimizer_step()
            finally:
                if scope.is_phase_active(scope.PHASE_UPDATE):
                    scope.stop_phase(scope.PHASE_UPDATE)

        self.optimizer_step = _timed_optimizer_step
        try:
            return original(self, data, loss_function)
        finally:
            self.optimizer_step = orig_optimizer_step

    return wrapped


def _patch_forward_backward_batch(original: Callable) -> Callable:
    """Wrap FSDPEngine.forward_backward_batch with hierarchical micro-batch profiling.

    The original method is re-implemented here (importing its helper functions)
    so that ``begin_micro_batch`` / ``end_micro_batch`` and ``PHASE_BACKWARD``
    can be inserted around each micro-batch without touching verl source.
    """

    def wrapped(self: Any, data: Any, loss_function: Any, forward_only: bool = False) -> Any:
        scope = ProfilerScope.current()
        if scope is None:
            return original(self, data, loss_function, forward_only)

        # Local imports keep the patch module lightweight when profiling is off.
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
        from verl.utils.device import get_device_id

        import torch

        # The following body mirrors FSDPEngine.forward_backward_batch and only
        # adds profiling markers around each micro-batch and loss.backward().
        import verl.utils.tensordict_utils as tu

        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()
        scaler = getattr(self, "scaler", None)

        for micro_idx, micro_batch in enumerate(micro_batches):
            scope.begin_micro_batch(micro_idx, forward_only=forward_only)
            try:
                with ctx:
                    loss, meta_info = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only
                    )

                    if not forward_only:
                        scope.start_phase(scope.PHASE_BACKWARD)
                        try:
                            if scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        finally:
                            if scope.is_phase_active(scope.PHASE_BACKWARD):
                                scope.stop_phase(scope.PHASE_BACKWARD)
            finally:
                scope.end_micro_batch()

            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    return wrapped


def _patch_train_mini_batch(original: Callable) -> Callable:
    """Wrap TrainingWorker.train_mini_batch with a step-level ProfilerScope.

    The wrapped method preserves the Ray ``@register`` decorator's MAGIC_ATTR so
    that verl's single-controller still recognises it as a remote method.
    """

    def wrapped(self: Any, data: Any) -> Any:
        scope = ProfilerScope.create_if_enabled(
            getattr(self.engine, "_ps_step_counter", 0), kind="train"
        )
        if scope is None:
            return original(self, data)

        mb_idx = 0
        orig_train_batch = self.train_batch

        def _timed_train_batch(batch_data: Any) -> Any:
            nonlocal mb_idx
            scope.begin_minibatch(mb_idx)
            try:
                return orig_train_batch(batch_data)
            finally:
                scope.end_minibatch()
                mb_idx += 1

        self.train_batch = _timed_train_batch
        with scope:
            try:
                result = original(self, data)
            finally:
                self.train_batch = orig_train_batch
                self.engine._ps_step_counter = getattr(self.engine, "_ps_step_counter", 0) + 1
        return result

    _copy_ray_register_attrs(original, wrapped)
    return wrapped


def _patch_infer_batch(original: Callable) -> Callable:
    """Wrap TrainingWorker.infer_batch with a logp ProfilerScope."""

    def wrapped(self: Any, data: Any) -> Any:
        scope = ProfilerScope.create_if_enabled(
            getattr(self.engine, "_ps_step_counter", 0), kind="logp"
        )
        if scope is None:
            return original(self, data)

        with scope:
            return original(self, data)

    _copy_ray_register_attrs(original, wrapped)
    return wrapped


# PatchSpec factory helpers used by __init__.py

def patch_train_batch(original: Callable) -> Callable:
    return _patch_train_batch(original)


def patch_forward_backward_batch(original: Callable) -> Callable:
    return _patch_forward_backward_batch(original)


def patch_train_mini_batch(original: Callable) -> Callable:
    return _patch_train_mini_batch(original)


def patch_infer_batch(original: Callable) -> Callable:
    return _patch_infer_batch(original)
