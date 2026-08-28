from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from finetune_library.checkpoint import (
    load_adapter,
    load_training_state,
    save_adapter,
    save_checkpoint,
)
from finetune_library.config import DataFormat, ExperimentConfig
from finetune_library.data import (
    CausalCollator,
    cache_key,
    count_loss_tokens,
    load_prepared,
    prepare_to_disk,
)
from finetune_library.distributed import (
    DistributedContext,
    initialize_distributed,
    maybe_no_sync,
    wrap_model,
)
from finetune_library.logging_utils import EventLogger, setup_training_logger
from finetune_library.lora import consolidate_tied_lora_gradients, inject_lora
from finetune_library.loss import build_training_model
from finetune_library.optim import build_optimizer, build_scheduler
from finetune_library.registry import resolve_model
from finetune_library.runtime import load_runtime, maybe_compile
from finetune_library.tracking import TrackingSession


@dataclass(slots=True)
class TrainPosition:
    step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0


@dataclass(slots=True)
class PendingStepMetric:
    step: int
    global_tokens: torch.Tensor
    global_loss_sum: torch.Tensor
    grad_norm: torch.Tensor | float | None
    data_seconds: float
    warmup: bool
    learning_rate: float
    started_at: float
    cpu_seconds: float
    start_event: Any = None
    end_event: Any = None

    def resolve(self, device: torch.device) -> dict[str, Any]:
        if self.end_event is not None:
            self.end_event.synchronize()
            step_seconds = float(self.start_event.elapsed_time(self.end_event) / 1000.0)
        else:
            step_seconds = self.cpu_seconds
        tokens = float(self.global_tokens.item())
        if tokens <= 0:
            raise ValueError("an optimizer step contains no trainable labels")
        loss = float((self.global_loss_sum / self.global_tokens).item())
        grad_norm = (
            float(self.grad_norm.item())
            if isinstance(self.grad_norm, torch.Tensor)
            else self.grad_norm
        )
        if not math.isfinite(loss) or (
            grad_norm is not None and not math.isfinite(grad_norm)
        ):
            raise FloatingPointError(f"non-finite training state at step {self.step}")
        return {
            "step": self.step,
            "loss": loss,
            "learning_rate": self.learning_rate,
            "grad_norm": grad_norm,
            "tokens": int(tokens),
            "tokens_per_second": tokens / step_seconds,
            "step_seconds": step_seconds,
            "data_seconds": self.data_seconds,
            "peak_memory_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "peak_allocated_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "warmup": self.warmup,
            "elapsed_seconds": time.perf_counter() - self.started_at,
        }


def _seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)


def _prepare_dataset(
    config: ExperimentConfig,
    tokenizer: Any,
    context: DistributedContext,
):
    source = Path(config.data.path) if config.data.path else None
    is_prepared = (
        config.data.format == DataFormat.TOKENIZED
        and source is not None
        and source.is_dir()
        and (source / "dataset_info.json").exists()
    )
    if is_prepared:
        assert source is not None
        prepared = source
    else:
        prepared = Path(config.data.cache_dir) / cache_key(config, tokenizer)
        if context.is_main:
            prepare_to_disk(config, tokenizer, prepared)
        context.barrier()
    return load_prepared(prepared)


def _make_loader(
    config: ExperimentConfig,
    tokenizer: Any,
    dataset: Any,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler[Any] | None]:
    sampler: DistributedSampler[Any] | None = None
    shuffle = True
    if context.data_parallel_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.data_parallel_size,
            rank=context.data_parallel_rank,
            shuffle=True,
            seed=config.training.seed,
            drop_last=True,
        )
        shuffle = False
    workers = config.training.dataloader_workers
    loader = DataLoader(
        dataset,
        batch_size=config.training.per_device_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=CausalCollator(tokenizer, config.training.max_seq_length),
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=config.training.prefetch_factor if workers > 0 else None,
        drop_last=True,
    )
    return loader, sampler


def _total_steps(config: ExperimentConfig, loader: DataLoader) -> int:
    available = len(loader) // config.training.gradient_accumulation_steps
    if available < 1:
        raise ValueError(
            "dataset has fewer full batches than gradient_accumulation_steps; "
            "reduce the accumulation or add data"
        )
    if config.training.max_steps:
        return config.training.max_steps
    return available * config.training.num_epochs


def _next_group(
    iterator: Any,
    accumulation_steps: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(accumulation_steps):
        try:
            batches.append(next(iterator))
        except StopIteration:
            return []
    return batches


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device=device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def _write_metric(
    output_dir: Path,
    metric: dict[str, Any],
    *,
    write_jsonl: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_jsonl:
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")


def run_training(config: ExperimentConfig, *, benchmark: bool = False) -> dict[str, Any]:
    spec = resolve_model(config.model.name)
    context = initialize_distributed(config.distributed, spec)
    output_dir = Path(config.training.output_dir)
    logger = setup_training_logger(output_dir, config.logging, rank=context.rank)
    events = EventLogger(output_dir, config.logging, rank=context.rank)
    tracking = TrackingSession(config, rank=context.rank)
    try:
        _seed_everything(config.training.seed, context.rank)
        if config.precision.allow_tf32 and context.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        runtime = load_runtime(config, spec, context)
        peft_model, audit = inject_lora(
            runtime.model,
            spec,
            config.lora,
            expert_implementation=config.runtime.experts,
        )
        if config.checkpoint.resume_from:
            load_adapter(peft_model, config.checkpoint.resume_from)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "lora_audit.json").write_text(
                json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                audit.format(verbose=config.runtime.verbose_lora_audit),
                flush=True,
            )

        training_model = build_training_model(peft_model, config.runtime.loss)
        model = wrap_model(training_model, context, spec, config.distributed)
        model = maybe_compile(model, config)
        tracking.watch(model)
        optimizer = build_optimizer(model, config.optimizer)

        dataset = _prepare_dataset(config, runtime.tokenizer, context)
        loader, sampler = _make_loader(config, runtime.tokenizer, dataset, context)
        total_steps = _total_steps(config, loader)
        if benchmark:
            total_steps = config.training.benchmark_warmup_steps + config.training.benchmark_steps
        scheduler = build_scheduler(optimizer, config.scheduler, total_steps)

        position = TrainPosition()
        if config.checkpoint.resume_from:
            restored = load_training_state(
                checkpoint=config.checkpoint.resume_from,
                optimizer=optimizer,
                scheduler=scheduler,
                context=context,
            )
            position = TrainPosition(**restored)

        metadata = {
            "model": spec.repo_id,
            "revision": runtime.revision,
            "backend": runtime.backend.value,
            "attention": runtime.attention,
            "model_kernels": runtime.model_kernels,
            "loss_implementation": training_model.loss_implementation,
            "strategy": context.strategy.value,
            "world_size": context.world_size,
            "shard_size": context.shard_size,
            "replicate_size": context.replicate_size,
            "optimizer": config.optimizer.name.value,
            "lora_dtype": str(audit.trainable_dtype),
            "lora_trainable_parameters": audit.trainable_parameters,
        }
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            config.with_model_revision(runtime.revision).write_resolved(
                output_dir / "resolved_config.json"
            )
            logger.info("run_start %s", json.dumps(metadata, sort_keys=True))
            events.emit("run_start", metadata=metadata)
            print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        context.barrier()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if context.device.type == "cuda":
            # Transformers loads a full checkpoint before FSDP2 replaces
            # parameters with local DTensor shards. Release that transient
            # allocator cache before measuring and before long-sequence work.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(context.device)

        measured_tokens = 0
        measured_time = 0.0
        measured_rates: list[float] = []
        last_metric: dict[str, Any] = {}
        pending_metrics: list[PendingStepMetric] = []
        train_started = time.perf_counter()
        epoch = position.epoch
        while position.step < total_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            for _ in range(position.batch_in_epoch):
                try:
                    next(iterator)
                except StopIteration:
                    break

            while position.step < total_steps:
                data_started = time.perf_counter()
                batches = _next_group(iterator, config.training.gradient_accumulation_steps)
                if not batches:
                    break
                data_seconds = time.perf_counter() - data_started
                position.batch_in_epoch += len(batches)

                local_tokens = sum(int(count_loss_tokens(batch).item()) for batch in batches)
                tokens = torch.tensor(
                    float(local_tokens), device=context.device, dtype=torch.float32
                )
                context.reduce_data_parallel(tokens)
                if position.step == 0 and float(tokens.item()) <= 0:
                    raise ValueError("an optimizer step contains no trainable labels")
                # DDP/HSDP averages across replicas, so each replica divides by
                # the mean token count to produce the exact global token mean.
                denominator = tokens / context.data_parallel_size

                step_started = time.perf_counter()
                start_event = None
                end_event = None
                if context.device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                local_loss_sum = torch.zeros(
                    (), device=context.device, dtype=torch.float32
                )
                for micro_index, cpu_batch in enumerate(batches):
                    batch = _move_batch(cpu_batch, context.device)
                    sync = micro_index == len(batches) - 1
                    with maybe_no_sync(model, synchronize=sync):
                        with torch.autocast(
                            device_type=context.device.type,
                            dtype=torch.bfloat16,
                            enabled=context.device.type == "cuda",
                        ):
                            output = model(
                                **batch,
                                use_cache=False,
                                num_items_in_batch=denominator,
                            )
                        if output.loss is None:
                            raise RuntimeError("model did not return a training loss")
                        output.loss.backward()
                        local_loss_sum.add_(output.loss.detach().float() * denominator)

                context.sync_replicated_gradients()
                consolidate_tied_lora_gradients(model)
                grad_norm: torch.Tensor | float | None = None
                if config.training.max_grad_norm > 0:
                    grad_norm = context.clip_grad_norm(
                        model.parameters(),
                        config.training.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if end_event is not None:
                    end_event.record()
                cpu_seconds = time.perf_counter() - step_started
                position.step += 1

                context.reduce_data_parallel(local_loss_sum)
                warmup = benchmark and (position.step <= config.training.benchmark_warmup_steps)
                pending_metrics.append(
                    PendingStepMetric(
                        step=position.step,
                        global_tokens=tokens,
                        global_loss_sum=local_loss_sum,
                        grad_norm=grad_norm,
                        data_seconds=data_seconds,
                        warmup=warmup,
                        learning_rate=scheduler.get_last_lr()[0],
                        started_at=train_started,
                        cpu_seconds=cpu_seconds,
                        start_event=start_event,
                        end_event=end_event,
                    )
                )
                save_every = config.checkpoint.save_every_steps
                should_log = (
                    position.step % config.training.log_every_steps == 0
                    or position.step == 1
                    or position.step == total_steps
                )
                should_flush = (
                    (not benchmark and should_log)
                    # Periodically resolve CUDA events during long stability
                    # benchmarks. Keeping one event pair per step until the
                    # final report would grow host/driver state without bound.
                    or (benchmark and should_log)
                    or (not benchmark and save_every and position.step % save_every == 0)
                )
                if should_flush:
                    for pending in pending_metrics:
                        resolved = pending.resolve(context.device)
                        if not pending.warmup:
                            measured_tokens += resolved["tokens"]
                            measured_time += resolved["step_seconds"]
                            measured_rates.append(resolved["tokens_per_second"])
                        last_metric = resolved
                        if context.is_main and (not benchmark or pending.step == total_steps):
                            _write_metric(
                                output_dir,
                                resolved,
                                write_jsonl=config.logging.metrics_jsonl,
                            )
                            if config.logging.console:
                                print(json.dumps(resolved, sort_keys=True), flush=True)
                            logger.info(
                                "step=%s loss=%.6f lr=%.2e grad_norm=%s tokens/s=%.1f",
                                resolved["step"],
                                resolved["loss"],
                                resolved["learning_rate"],
                                resolved["grad_norm"],
                                resolved["tokens_per_second"],
                            )
                            events.emit("step", **resolved)
                            tracking.log_step(resolved["step"], resolved)
                    pending_metrics.clear()
                if not benchmark and save_every and position.step % save_every == 0:
                    checkpoint_started = time.perf_counter()
                    save_checkpoint(
                        model=model,
                        tokenizer=runtime.tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        context=context,
                        step=position.step,
                        epoch=epoch,
                        batch_in_epoch=position.batch_in_epoch,
                        metadata=metadata,
                    )
                    last_metric["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
                    if context.is_main:
                        events.emit("checkpoint_saved", step=position.step)
                        logger.info("checkpoint_saved step=%s", position.step)

            epoch += 1
            position.epoch = epoch
            position.batch_in_epoch = 0
            if not config.training.max_steps and epoch >= config.training.num_epochs:
                break

        summary = {
            **metadata,
            "steps": position.step,
            "measured_tokens": measured_tokens,
            "measured_seconds": measured_time,
            "median_equivalent_tokens_per_second": (
                statistics.median(measured_rates) if measured_rates else math.nan
            ),
            "aggregate_tokens_per_second": (
                measured_tokens / measured_time if measured_time else math.nan
            ),
            "peak_memory_gib": (
                torch.cuda.max_memory_reserved(context.device) / 2**30
                if context.device.type == "cuda"
                else 0.0
            ),
            "peak_allocated_memory_gib": (
                torch.cuda.max_memory_allocated(context.device) / 2**30
                if context.device.type == "cuda"
                else 0.0
            ),
            "last_loss": last_metric.get("loss"),
        }
        if not benchmark and config.checkpoint.save_final:
            save_adapter(
                model,
                runtime.tokenizer,
                output_dir / "final",
                config,
                context,
                {**metadata, "step": position.step},
            )
        if context.is_main:
            (output_dir / ("benchmark.json" if benchmark else "summary.json")).write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            events.emit("run_end", summary=summary)
            logger.info("run_end %s", json.dumps(summary, sort_keys=True))
            tracking.finish(summary)
        context.barrier()
        return summary
    finally:
        tracking.finish()
        context.close()
