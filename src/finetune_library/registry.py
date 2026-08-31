from __future__ import annotations

from dataclasses import dataclass

from finetune_library.config import DistributedStrategy


@dataclass(frozen=True, slots=True)
class DistributedCandidate:
    strategy: DistributedStrategy
    shard_size: int
    replicate_size: int


DDP_8 = DistributedCandidate(DistributedStrategy.DDP, shard_size=1, replicate_size=8)
HSDP_2 = DistributedCandidate(DistributedStrategy.HSDP, shard_size=2, replicate_size=4)
HSDP_4 = DistributedCandidate(DistributedStrategy.HSDP, shard_size=4, replicate_size=2)
FSDP_8 = DistributedCandidate(DistributedStrategy.FSDP, shard_size=8, replicate_size=1)


HSDP_EP_4 = DistributedCandidate(DistributedStrategy.HSDP, shard_size=2, replicate_size=2)
FSDP_EP_8 = DistributedCandidate(DistributedStrategy.FSDP, shard_size=1, replicate_size=8)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo_id: str
    revision: str
    architecture: str
    layer_class: str
    preferred_strategy: DistributedStrategy
    shard_size: int = 1
    replicate_size: int = 8
    num_experts: int | None = None
    recommended_lr_cpt: float = 2.0e-4
    recommended_lr_sft: float = 1.0e-4
    # PyTorch SDPA is the measured winner on H100 at sequence length 2048.
    # Keep the binary FlashAttention wheel available for explicit benchmarks.
    attention_candidates: tuple[str, ...] = ("sdpa", "flash_attention_2")
    distributed_candidates: tuple[DistributedCandidate, ...] = (
        DDP_8,
        HSDP_2,
        HSDP_4,
        FSDP_8,
    )
    chat_template: str = "tokenizer"
    text_only: bool = False
    moe: bool = False
    # Promotion is benchmark-gated. No model is opted in merely because the
    # optional package can import its architecture.
    unsloth_compatible: bool = False


_SPECS = (
    ModelSpec(
        key="qwen3-8b",
        repo_id="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        architecture="dense",
        layer_class="Qwen3DecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        unsloth_compatible=True,
    ),
    ModelSpec(
        key="qwen3-14b",
        repo_id="Qwen/Qwen3-14B",
        revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        architecture="dense",
        layer_class="Qwen3DecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        unsloth_compatible=True,
    ),
    ModelSpec(
        key="qwen3-32b",
        repo_id="Qwen/Qwen3-32B",
        revision="9216db5781bf21249d130ec9da846c4624c16137",
        architecture="dense",
        layer_class="Qwen3DecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        distributed_candidates=(DDP_8, HSDP_2, HSDP_4, FSDP_8),
        unsloth_compatible=True,
    ),
    ModelSpec(
        key="deepseek-r1-distill-llama-8b",
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        revision="6a6f4aa4197940add57724a7707d069478df56b1",
        architecture="dense",
        layer_class="LlamaDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        unsloth_compatible=True,
    ),
    ModelSpec(
        key="olmo3-32b-think-dpo",
        repo_id="allenai/Olmo-3-32B-Think-DPO",
        revision="97604023f2602a7dfe6d31b1fc229985e0002022",
        architecture="dense",
        layer_class="Olmo3DecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        distributed_candidates=(DDP_8, HSDP_2, HSDP_4, FSDP_8),
    ),
    ModelSpec(
        key="gemma4-26b-a4b-it",
        repo_id="google/gemma-4-26B-A4B-it",
        revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        architecture="moe",
        layer_class="Gemma4TextDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        attention_candidates=("sdpa", "eager"),
        distributed_candidates=(DDP_8, HSDP_2, HSDP_4, FSDP_8, HSDP_EP_4, FSDP_EP_8),
        text_only=True,
        moe=True,
        num_experts=128,
        unsloth_compatible=False,
    ),
    ModelSpec(
        key="qwen3.5-35b-a3b",
        repo_id="Qwen/Qwen3.5-35B-A3B",
        revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        architecture="moe",
        layer_class="Qwen3_5MoeDecoderLayer",
        preferred_strategy=DistributedStrategy.FSDP,
        attention_candidates=("sdpa", "flash_attention_2", "eager"),
        distributed_candidates=(FSDP_8, HSDP_2, HSDP_4, HSDP_EP_4, FSDP_EP_8),
        text_only=True,
        moe=True,
        num_experts=256,
        unsloth_compatible=False,
    ),
    ModelSpec(
        key="gemma4-31b-it",
        repo_id="google/gemma-4-31B-it",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
        architecture="dense",
        layer_class="Gemma4TextDecoderLayer",
        preferred_strategy=DistributedStrategy.DDP,
        attention_candidates=("sdpa", "eager"),
        distributed_candidates=(DDP_8, HSDP_2, HSDP_4, FSDP_8),
        text_only=True,
        unsloth_compatible=False,
    ),
)

MODEL_REGISTRY = {spec.key: spec for spec in _SPECS}
MODEL_REGISTRY.update({spec.repo_id: spec for spec in _SPECS})


def resolve_model(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as error:
        supported = ", ".join(spec.key for spec in _SPECS)
        raise ValueError(f"unsupported model {name!r}; choose one of: {supported}") from error


def unique_models() -> tuple[ModelSpec, ...]:
    return _SPECS
