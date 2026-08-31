"""Skip Triton autotune benchmarking for FLA kernels on memory-tight runs.

FLA's gated-delta-rule backward autotuner can OOM when EP+FSDP already
occupies most of the H100. Set FLA_SKIP_TRITON_AUTOTUNE=1 (default when
this file is sourced via PYTHONSTARTUP) to pick the first valid config
without benchmarking every candidate.
"""
from __future__ import annotations

import os

if os.environ.get("FLA_SKIP_TRITON_AUTOTUNE", "1") != "1":
    raise SystemExit(0)

from triton.runtime.autotuner import Autotuner

_orig_run = Autotuner.run


def _run_skip_autotune(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))
    if len(self.configs) > 1:
        all_args = {**self.nargs, **kwargs}
        selected_args = {key: all_args[key] for key in self.arg_names if key in all_args}
        key = [selected_args[item] for item in self.keys if item in selected_args]
        for _, arg in selected_args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)
        if key not in self.cache:
            pruned_configs = self.prune_configs(kwargs)
            if not pruned_configs:
                raise RuntimeError("no valid Triton autotuner configs after pruning")
            self.cache[key] = pruned_configs[0]
        config = self.cache[key]
    else:
        config = self.configs[0]

    self.best_config = config
    if config.pre_hook is not None:
        full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
        config.pre_hook(full_nargs)
    ret = self.fn.run(
        *args,
        **kwargs,
        **config.all_kwargs(),
    )
    self.nargs = None
    return ret


Autotuner.run = _run_skip_autotune
