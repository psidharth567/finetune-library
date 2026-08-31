from __future__ import annotations

import os


def maybe_patch_fla_triton_autotune() -> None:
    if os.environ.get("FLA_SKIP_TRITON_AUTOTUNE", "0") != "1":
        return

    from triton.runtime.autotuner import Autotuner

    if getattr(Autotuner.run, "__name__", "") == "_run_skip_autotune":
        return

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

    _run_skip_autotune.__name__ = "_run_skip_autotune"
    Autotuner.run = _run_skip_autotune  # type: ignore[method-assign]

    # FLA wraps Autotuner; patch the base class before any FLA kernels import.
    del _orig_run
