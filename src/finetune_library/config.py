from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(StrEnum):
    CPT = "cpt"
    SFT = "sft"
    DPO = "dpo"


class DataFormat(StrEnum):
    TEXT = "text"
    TOKENIZED = "tokenized"
    MESSAGES = "messages"
    PROMPT_COMPLETION = "prompt_completion"
    ALPACA = "alpaca"
    PREFERENCE = "preference"


class DistributedStrategy(StrEnum):
    AUTO = "auto"
    DDP = "ddp"
    HSDP = "hsdp"
    FSDP = "fsdp"


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    ADAMW_8BIT = "adamw_8bit"


class RuntimeBackend(StrEnum):
    AUTO = "auto"
    NATIVE = "native"
    UNSLOTH = "unsloth"


class ModelConfig(StrictModel):
    name: str = Field(description="Registry key or exact Hugging Face repository ID")
    revision: str | None = Field(
        default=None,
        description="Explicit revision override; registry models otherwise use their pinned commit",
    )
    # None = standard Hugging Face hub cache ($HF_HUB_CACHE, i.e. $HF_HOME/hub).
    # An explicit value is passed as the hub cache dir (directory holding models--*).
    cache_dir: str | None = None
    trust_remote_code: bool = False
    chat_template: str | None = Field(
        default=None,
        description=(
            "Jinja chat template overriding the tokenizer's own (e.g. plain ChatML "
            "to disable Qwen3 thinking blocks in training targets). None keeps the "
            "tokenizer template."
        ),
    )


class DataConfig(StrictModel):
    format: DataFormat
    path: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    messages_field: str = "messages"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    cache_dir: str = ".cache/prepared"
    max_samples: PositiveInt | None = None
    packing: bool = True
    drop_remainder: bool = True
    add_bos: bool = False
    add_eos: bool = True
    packing_isolation: Literal["none", "attention"] = "none"
    chunk_long_examples: bool = False
    chunk_overlap: int = Field(default=0, ge=0)
    chunk_strategy: Literal["truncate", "sliding_window"] = "truncate"
    require_full_seq_length: bool = False
    length_policy: Literal["global", "per_example", "per_dataset"] = "global"
    default_max_length: PositiveInt | None = None
    # SFT only: an example longer than training.max_seq_length is never
    # truncated (that would cut the prompt or the target). "drop" skips it and
    # reports the count at prepare time; "error" fails preparation instead.
    sft_overlength: Literal["drop", "error"] = "drop"

    @model_validator(mode="after")
    def validate_source(self) -> DataConfig:
        if (self.path is None) == (self.dataset_name is None):
            raise ValueError("exactly one of data.path or data.dataset_name must be set")
        return self

    @model_validator(mode="after")
    def validate_packing(self) -> DataConfig:
        if self.packing_isolation != "none" and not self.packing:
            raise ValueError("data.packing_isolation requires data.packing=true")
        if self.chunk_overlap and not self.chunk_long_examples:
            raise ValueError("data.chunk_overlap requires data.chunk_long_examples=true")
        if self.chunk_overlap and self.chunk_strategy != "sliding_window":
            raise ValueError("data.chunk_overlap requires data.chunk_strategy=sliding_window")
        return self


class PreferenceConfig(StrictModel):
    """Direct preference optimization (task=dpo) settings.

    The reference policy is the frozen base model: the LoRA adapter is
    disabled for the reference forward pass, so no second model copy is held.
    """

    beta: Annotated[float, Field(gt=0.0)] = 0.1
    # sigmoid: Rafailov et al. 2023; ipo: Azar et al. 2023 (length-normalized
    # log-probs, as in the paper and TRL); hinge: SLiC-style.
    loss_type: Literal["sigmoid", "ipo", "hinge"] = "sigmoid"
    # Conservative DPO (sigmoid only): probability the preference label is flipped.
    label_smoothing: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.0
    # Optional NLL on the chosen response (per-token mean per pair), RPO-style.
    sft_weight: Annotated[float, Field(ge=0.0)] = 0.0
    # Row fields. `data.prompt_field` names the prompt; chosen/rejected are
    # either response strings or message lists (with or without the prompt turns).
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    # Log-prob chunk (tokens) for the LM head; bounds peak logits memory.
    # 1024 OOMed Qwen3.5-35B-A3B EP4 (248k vocab); 256 fits (74 GiB) and cost
    # Qwen3-8B 0.7% tok/s vs 1024 (8xH100, 50 measured steps).
    logprob_chunk_tokens: PositiveInt = 256

    @model_validator(mode="after")
    def validate_loss(self) -> PreferenceConfig:
        if self.label_smoothing and self.loss_type != "sigmoid":
            raise ValueError("preference.label_smoothing only applies to loss_type=sigmoid")
        return self


class LoraSettings(StrictModel):
    rank: PositiveInt = 32
    alpha: PositiveInt = 64
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    expert_rank: PositiveInt = 4

    @model_validator(mode="after")
    def validate_ranks(self) -> LoraSettings:
        if self.expert_rank > self.rank:
            raise ValueError("lora.expert_rank cannot exceed lora.rank")
        return self


class OptimizerConfig(StrictModel):
    name: OptimizerName = OptimizerName.ADAMW
    learning_rate: Annotated[float, Field(gt=0.0)] | None = None
    weight_decay: Annotated[float, Field(ge=0.0)] = 0.01
    beta1: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.9
    beta2: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.999
    epsilon: Annotated[float, Field(gt=0.0)] = 1.0e-8
    fused: bool = True


class SchedulerConfig(StrictModel):
    name: Literal["cosine", "linear", "constant", "wsd"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    stable_steps: int | None = Field(default=None, ge=0)
    decay_steps: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_wsd(self) -> SchedulerConfig:
        if self.name == "wsd" and (self.stable_steps is None or self.decay_steps is None):
            raise ValueError("wsd requires scheduler.stable_steps and scheduler.decay_steps")
        return self


class PrecisionConfig(StrictModel):
    model_dtype: Literal["bfloat16"] = "bfloat16"
    lora_dtype: Literal["bfloat16"] = "bfloat16"
    # Full fine-tuning only (lora: null): dtype of the trainable weights the
    # optimizer updates (compute stays bf16 via autocast / the FSDP mixed
    # precision policy). float32 keeps small updates (LR ~1e-6..1e-5) that
    # pure-bf16 weights round away; the optimizer states follow this dtype.
    master_dtype: Literal["float32", "bfloat16"] = "float32"
    allow_tf32: bool = True


class DistributedConfig(StrictModel):
    strategy: DistributedStrategy = DistributedStrategy.AUTO
    shard_size: PositiveInt | None = None
    replicate_size: PositiveInt | None = None
    expert_parallel_size: PositiveInt = 1
    reduce_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    # None reproduces current behaviour byte-for-byte: per-layer units reshard
    # eagerly after forward (reshard_after_forward=True) while the root module
    # keeps its all-gathered params through backward (reshard_after_forward=False).
    # Setting this explicitly overrides the per-layer unit value only; the root
    # module is always kept at reshard_after_forward=False regardless.
    reshard_after_forward: bool | None = None
    # Opt-in FSDP2 forward/backward prefetch depth for decoder-layer shard
    # units. 0 (default) reproduces current behaviour: no explicit prefetch
    # hints are installed, so FSDP2 falls back to its implicit prefetch.
    fsdp_prefetch_layers: Annotated[int, Field(ge=0)] = 0
    # Opt-in DDP static_graph. False (default) reproduces current behaviour.
    # Only meaningful for strategy=ddp; ignored otherwise.
    ddp_static_graph: bool = False

    @model_validator(mode="after")
    def validate_mesh(self) -> DistributedConfig:
        if self.strategy == DistributedStrategy.HSDP and (
            self.shard_size is None or self.replicate_size is None
        ):
            raise ValueError("hsdp requires shard_size and replicate_size")
        if self.strategy == DistributedStrategy.FSDP and self.replicate_size not in (None, 1):
            raise ValueError("fsdp cannot use replicate_size > 1")
        if self.expert_parallel_size > 1 and self.strategy == DistributedStrategy.DDP:
            raise ValueError("expert_parallel_size > 1 requires fsdp or hsdp")
        return self


class RuntimeConfig(StrictModel):
    backend: RuntimeBackend = RuntimeBackend.AUTO
    attention: Literal[
        "auto",
        "sdpa",
        "flash_attention_2",
        "flash_attention_3",
        "eager",
        "flex_attention",
    ] = "auto"
    model_kernels: Literal["auto", "native", "liger"] = "auto"
    loss: Literal["auto", "cross_entropy", "fused_linear_cross_entropy"] = "auto"
    experts: Literal["auto", "eager", "grouped_mm"] = "auto"
    moe_a2a_backend: Literal["auto", "native", "deepep"] = "auto"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_scope: Literal["full", "loss_only", "blocks"] = "full"
    gradient_checkpointing: bool = False
    use_cache: bool = False
    verbose_lora_audit: bool = False


class CheckpointConfig(StrictModel):
    save_every_steps: int = Field(default=0, ge=0)
    keep_last: PositiveInt = 2
    resume_from: str | None = None
    save_final: bool = True


class WandbConfig(StrictModel):
    enabled: bool = False
    api_key: str | None = None
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    log_every_steps: PositiveInt | None = None
    watch_model: bool = False


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: bool = True
    console: bool = True
    metrics_jsonl: bool = True
    events_jsonl: bool = True
    wandb: WandbConfig = WandbConfig()


class TrainingConfig(StrictModel):
    output_dir: str
    max_seq_length: PositiveInt = 2048
    per_device_batch_size: PositiveInt = 1
    gradient_accumulation_steps: PositiveInt = 1
    max_steps: int = Field(default=0, ge=0)
    num_epochs: PositiveInt = 1
    seed: int = 42
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    log_every_steps: PositiveInt = 10
    dataloader_workers: int = Field(default=4, ge=0)
    prefetch_factor: PositiveInt = 4
    benchmark_warmup_steps: int = Field(default=10, ge=0)
    benchmark_steps: PositiveInt = 50


class ExperimentConfig(StrictModel):
    version: Literal[1] = 1
    task: Task
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    # null = full fine-tuning: every parameter trainable, no adapter.
    lora: LoraSettings | None = LoraSettings()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    precision: PrecisionConfig = PrecisionConfig()
    distributed: DistributedConfig = DistributedConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()
    # task=dpo only; None for CPT/SFT (and omitted from resolved configs then).
    preference: PreferenceConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def default_preference(cls, values: object) -> object:
        if (
            isinstance(values, dict)
            and values.get("task") in (Task.DPO, Task.DPO.value)
            and values.get("preference") is None
        ):
            return {**values, "preference": {}}
        return values

    @model_validator(mode="before")
    @classmethod
    def resolve_sft_packing_isolation(cls, values: object) -> object:
        """SFT packing always isolates examples; make that explicit in the config.

        Packed SFT examples must not attend to each other, so for task=sft with
        data.packing=true an omitted data.packing_isolation resolves to
        "attention" and an explicit "none" is rejected.
        """
        if not isinstance(values, dict) or values.get("task") not in (Task.SFT, Task.SFT.value):
            return values
        data = values.get("data")
        if not isinstance(data, dict) or not data.get("packing", True):
            return values
        isolation = data.get("packing_isolation")
        if isolation is None:
            return {**values, "data": {**data, "packing_isolation": "attention"}}
        if isolation != "attention":
            raise ValueError(
                "task=sft with data.packing=true packs whole examples and always isolates "
                "them (data.packing_isolation=attention); packing_isolation=none would let "
                "packed examples attend to each other"
            )
        return values

    def packing_isolation(self) -> str:
        """Effective isolation mode (SFT packing is always isolated).

        Use this rather than data.packing_isolation: model_copy() bypasses the
        validator above.
        """
        if self.task == Task.SFT and self.data.packing:
            return "attention"
        if self.task == Task.DPO:
            # DPO batches are always padding-free with per-sequence position_ids.
            return "attention"
        return self.data.packing_isolation

    @model_validator(mode="after")
    def validate_task_data(self) -> ExperimentConfig:
        cpt_formats = {DataFormat.TEXT, DataFormat.TOKENIZED}
        sft_formats = {
            DataFormat.MESSAGES,
            DataFormat.PROMPT_COMPLETION,
            DataFormat.ALPACA,
            DataFormat.TOKENIZED,
        }
        if self.preference is not None and self.task != Task.DPO:
            raise ValueError("the preference block only applies to task=dpo")
        if self.task == Task.DPO:
            self._validate_dpo()
            return self
        allowed = cpt_formats if self.task == Task.CPT else sft_formats
        if self.data.format not in allowed:
            raise ValueError(
                f"data.format={self.data.format.value!r} is invalid for task={self.task.value!r}"
            )
        if self.data.default_max_length is not None and self.data.length_policy == "global":
            raise ValueError("data.default_max_length requires length_policy != global")
        if self.data.packing_isolation == "attention" and self.runtime.attention == "flex_attention":
            # Isolation is driven by position_ids restarting at 0 in a padding-free
            # batch: flash attention turns that into varlen cu_seqlens, sdpa/eager
            # into a block-diagonal causal mask (transformers masking_utils). flex
            # is untested with packed position_ids.
            raise ValueError(
                "data.packing_isolation=attention is not supported with "
                "runtime.attention=flex_attention; use sdpa, eager, or flash_attention_2/3"
            )
        if self.task == Task.SFT and self.data.chunk_long_examples:
            raise ValueError(
                "data.chunk_long_examples splits examples and is CPT-only; SFT examples are "
                "never split or truncated (see data.sft_overlength)"
            )
        return self

    @property
    def full_finetune(self) -> bool:
        return self.lora is None

    @model_validator(mode="after")
    def validate_full_finetune(self) -> ExperimentConfig:
        if not self.full_finetune:
            return self
        if self.optimizer.learning_rate is None:
            raise ValueError(
                "full fine-tuning (lora: null) requires optimizer.learning_rate; the registry "
                "defaults are LoRA rates, 10-100x too high for full fine-tuning"
            )
        if self.runtime.loss == "fused_linear_cross_entropy":
            raise ValueError(
                "runtime.loss=fused_linear_cross_entropy is the fused LoRA head kernel; "
                "full fine-tuning uses runtime.loss=auto/cross_entropy"
            )
        if self.distributed.strategy == DistributedStrategy.DDP:
            raise ValueError(
                "full fine-tuning keeps weights, gradients and optimizer states for every "
                "parameter; use distributed.strategy=fsdp (one node) or hsdp (multi-node)"
            )
        if self.distributed.expert_parallel_size > 1:
            raise ValueError("full fine-tuning is not supported with expert parallelism yet")
        return self

    def _validate_dpo(self) -> None:
        if self.data.format != DataFormat.PREFERENCE:
            raise ValueError("task=dpo requires data.format=preference")
        if self.runtime.loss == "fused_linear_cross_entropy":
            raise ValueError(
                "task=dpo computes per-sequence log-probs itself; the fused LoRA "
                "cross-entropy kernel assumes uniform token weights. Use runtime.loss=auto"
            )
        if self.runtime.torch_compile:
            raise ValueError("task=dpo is not validated with runtime.torch_compile yet")
        if self.runtime.attention == "flex_attention":
            raise ValueError(
                "task=dpo uses padding-free batches (packed position_ids), which are "
                "not supported with runtime.attention=flex_attention"
            )
        if self.data.chunk_long_examples:
            raise ValueError("data.chunk_long_examples does not apply to task=dpo")

    def effective_learning_rate(self) -> float:
        if self.optimizer.learning_rate is not None:
            return self.optimizer.learning_rate
        from finetune_library.registry import resolve_model

        spec = resolve_model(self.model.name)
        if self.task == Task.CPT:
            return spec.recommended_lr_cpt
        if self.task == Task.DPO:
            # DPO moves a policy away from its reference; SFT-scale LoRA rates
            # overshoot. No per-model DPO presets are measured yet.
            return spec.recommended_lr_sft / 10
        return spec.recommended_lr_sft

    def resolved_optimizer_config(self) -> OptimizerConfig:
        if self.optimizer.learning_rate is not None:
            return self.optimizer
        return self.optimizer.model_copy(update={"learning_rate": self.effective_learning_rate()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping")
        return cls.model_validate(raw)

    def write_resolved(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def to_json_dict(self) -> dict[str, object]:
        """JSON dump; CPT/SFT configs omit the DPO-only preference block."""
        dumped = self.model_dump(mode="json")
        if self.preference is None:
            dumped.pop("preference")
        return dumped

    def with_model_revision(self, revision: str) -> ExperimentConfig:
        return self.model_copy(
            update={"model": self.model.model_copy(update={"revision": revision})}
        )
