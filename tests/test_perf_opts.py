from __future__ import annotations

from finetune_library.config import DistributedConfig
from finetune_library.optim import _can_use_fused_adamw


class _FakeParam:
    """Minimal stand-in for the attributes `_can_use_fused_adamw` inspects."""

    def __init__(self, device_type: str, class_name: str = "Parameter") -> None:
        self.device = type("Device", (), {"type": device_type})()
        self.__class__ = type(class_name, (), {})


def test_distributed_config_defaults_reproduce_current_behaviour() -> None:
    config = DistributedConfig()
    assert config.reshard_after_forward is None
    assert config.fsdp_prefetch_layers == 0


def test_distributed_config_accepts_explicit_reshard_and_prefetch_opt_in() -> None:
    config = DistributedConfig(reshard_after_forward=False, fsdp_prefetch_layers=2)
    assert config.reshard_after_forward is False
    assert config.fsdp_prefetch_layers == 2


def test_can_use_fused_adamw_allows_dtensor_params_on_cuda() -> None:
    # torch==2.11.0 (this project's pin) supports the fused kernel over
    # DTensor params (verified against a real FSDP2-sharded DTensor on GPU);
    # the guard should no longer reject them by class name.
    fake = _FakeParam("cuda", class_name="DTensor")
    assert _can_use_fused_adamw([fake], requested=True) is True  # type: ignore[list-item]


def test_can_use_fused_adamw_still_rejects_cpu_params() -> None:
    fake = _FakeParam("cpu")
    assert _can_use_fused_adamw([fake], requested=True) is False  # type: ignore[list-item]


def test_can_use_fused_adamw_respects_requested_false() -> None:
    fake = _FakeParam("cuda")
    assert _can_use_fused_adamw([fake], requested=False) is False  # type: ignore[list-item]
