from __future__ import annotations

import torch
from torch import nn

from finetune_library.runtime import _model_uses_liger_kernels


class _FakeNative(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeLigerByName(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeLigerByModule(nn.Module):
    pass


_FakeLigerByName.__name__ = "LigerGEGLUMLPForGemma4"
_FakeLigerByModule.__module__ = "liger_kernel.transformers.geglu"


def test_liger_guard_detects_gemma4_class_names() -> None:
    model = nn.Sequential(_FakeNative(), _FakeLigerByName())
    assert _model_uses_liger_kernels(model)


def test_liger_guard_detects_liger_kernel_module_path() -> None:
    model = nn.Sequential(_FakeNative(), _FakeLigerByModule())
    assert _model_uses_liger_kernels(model)


def test_liger_guard_ignores_native_modules() -> None:
    model = nn.Sequential(_FakeNative(), nn.Linear(4, 4))
    assert not _model_uses_liger_kernels(model)


def test_maybe_compile_skips_when_model_kernels_liger(monkeypatch) -> None:
    from finetune_library.config import ExperimentConfig
    from finetune_library.runtime import maybe_compile

    compiled = []

    def _fake_compile(module, **kwargs):
        compiled.append(module)
        return module

    monkeypatch.setattr("finetune_library.runtime.torch.compile", _fake_compile)
    config = ExperimentConfig.from_yaml(
        "configs/benchmarks/gemma4-26b-liger-compile-ddp-smoke.yaml"
    )
    model = nn.Linear(4, 4)
    result = maybe_compile(model, config, model_kernels="liger")
    assert result is model
    assert compiled == []
