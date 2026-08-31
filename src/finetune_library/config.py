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


class DataFormat(StrEnum):
    TEXT = "text"
    TOKENIZED = "tokenized"
    MESSAGES = "messages"
    PROMPT_COMPLETION = "prompt_completion"
    ALPACA = "alpaca"


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
    cache_dir: str = ".cache/huggingface"
    trust_remote_code: bool = False


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
    name: Literal["cosine", "constant", "wsd"] = "cosine"
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
    allow_tf32: bool = True


class DistributedConfig(StrictModel):
    strategy: DistributedStrategy = DistributedStrategy.AUTO
    shard_size: PositiveInt | None = None
    replicate_size: PositiveInt | None = None
    expert_parallel_size: PositiveInt = 1
    reduce_dtype: Literal["bfloat16", "float32"] = "bfloat16"

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
    attention: Literal["auto", "sdpa", "flash_attention_2", "eager", "flex_attention"] = "auto"
    model_kernels: Literal["auto", "native", "liger"] = "auto"
    loss: Literal["auto", "cross_entropy", "fused_linear_cross_entropy"] = "auto"
    experts: Literal["auto", "eager", "grouped_mm"] = "auto"
    moe_a2a_backend: Literal["auto", "native", "deepep"] = "auto"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_scope: Literal["full", "loss_only", "blocks"] = "full"
    gradient_checkpointing: bool = True
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
    lora: LoraSettings = LoraSettings()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    precision: PrecisionConfig = PrecisionConfig()
    distributed: DistributedConfig = DistributedConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def validate_task_data(self) -> ExperimentConfig:
        cpt_formats = {DataFormat.TEXT, DataFormat.TOKENIZED}
        sft_formats = {
            DataFormat.MESSAGES,
            DataFormat.PROMPT_COMPLETION,
            DataFormat.ALPACA,
            DataFormat.TOKENIZED,
        }
        allowed = cpt_formats if self.task == Task.CPT else sft_formats
        if self.data.format not in allowed:
            raise ValueError(
                f"data.format={self.data.format.value!r} is invalid for task={self.task.value!r}"
            )
        if self.data.default_max_length is not None and self.data.length_policy == "global":
            raise ValueError("data.default_max_length requires length_policy != global")
        return self

    def effective_learning_rate(self) -> float:
        if self.optimizer.learning_rate is not None:
            return self.optimizer.learning_rate
        from finetune_library.registry import resolve_model

        spec = resolve_model(self.model.name)
        return spec.recommended_lr_cpt if self.task == Task.CPT else spec.recommended_lr_sft

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
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def with_model_revision(self, revision: str) -> ExperimentConfig:
        return self.model_copy(
            update={"model": self.model.model_copy(update={"revision": revision})}
        )
