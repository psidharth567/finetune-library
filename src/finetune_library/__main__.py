from finetune_library.fla_runtime import maybe_patch_fla_triton_autotune

maybe_patch_fla_triton_autotune()

from finetune_library.cli import main

raise SystemExit(main())
