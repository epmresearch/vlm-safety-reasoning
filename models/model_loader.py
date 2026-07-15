"""
Centralized model loading for training and inference.

Uses Unsloth's FastVisionModel for optimized LoRA fine-tuning
with frozen vision encoder.
"""
from typing import Any, Dict, Optional, Tuple

from core.constants import MODEL_TIERS, MODEL_BATCH_CONFIGS, DEFAULT_MODEL_TIER
from core.logging import get_logger

logger = get_logger(__name__)


def get_model_info(tier: str = DEFAULT_MODEL_TIER) -> Dict[str, Any]:
    """Returns model info dict for the given tier.

    Args:
        tier: One of "2b", "4b", "8b".

    Returns:
        Dict with hf_path, short_name, size_label.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(
            f"Unknown model tier '{tier}'. Choose from: {list(MODEL_TIERS.keys())}"
        )
    return MODEL_TIERS[tier]


def get_batch_config(tier: str = DEFAULT_MODEL_TIER) -> Dict[str, int]:
    """Returns per-model batch size config for SFTTrainer.

    Args:
        tier: One of "2b", "4b", "8b".

    Returns:
        Dict with per_device_train_batch_size and gradient_accumulation_steps.
    """
    if tier not in MODEL_BATCH_CONFIGS:
        raise ValueError(
            f"Unknown model tier '{tier}'. Choose from: {list(MODEL_BATCH_CONFIGS.keys())}"
        )
    return MODEL_BATCH_CONFIGS[tier]


def load_model_for_training(
    model_name: Optional[str] = None,
    tier: str = DEFAULT_MODEL_TIER,
    sft_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """Loads a model with LoRA adapters for SFT training.

    Args:
        model_name: HuggingFace model path. If None, uses tier lookup.
        tier: Model tier for batch config lookup.
        sft_cfg: SFT config dict (from configs/sft.yaml). If None, uses defaults.

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastVisionModel

    if model_name is None:
        model_name = get_model_info(tier)["hf_path"]

    # Defaults if sft_cfg not provided
    load_in_4bit = True
    max_seq_length = 4096
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0
    use_gradient_checkpointing = "unsloth"
    finetune_vision_layers = False
    finetune_language_layers = True
    finetune_attention_modules = True
    finetune_mlp_modules = True

    if sft_cfg:
        load_in_4bit = sft_cfg.get("load_in_4bit", load_in_4bit)
        max_seq_length = sft_cfg.get("max_seq_length", max_seq_length)
        use_gradient_checkpointing = sft_cfg.get(
            "use_gradient_checkpointing", use_gradient_checkpointing
        )
        finetune_vision_layers = sft_cfg.get(
            "finetune_vision_layers", finetune_vision_layers
        )
        finetune_language_layers = sft_cfg.get(
            "finetune_language_layers", finetune_language_layers
        )
        finetune_attention_modules = sft_cfg.get(
            "finetune_attention_modules", finetune_attention_modules
        )
        finetune_mlp_modules = sft_cfg.get(
            "finetune_mlp_modules", finetune_mlp_modules
        )
        lora_cfg = sft_cfg.get("lora", {})
        lora_r = lora_cfg.get("r", lora_r)
        lora_alpha = lora_cfg.get("alpha", lora_alpha)
        lora_dropout = lora_cfg.get("dropout", lora_dropout)

    logger.info(f"Loading model: {model_name} (4-bit={load_in_4bit})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_seq_length=max_seq_length,
    )

    logger.info(
        f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"vision_frozen={not finetune_vision_layers}"
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=42,
        use_rslora=False,
        target_modules=sft_cfg.get("lora", {}).get("target_modules", "all-linear")
        if sft_cfg
        else "all-linear",
    )

    return model, tokenizer


def load_model_for_inference(
    model_name: Optional[str] = None,
    tier: str = DEFAULT_MODEL_TIER,
    adapter_path: Optional[str] = None,
) -> Tuple:
    """Loads a model for inference (with optional LoRA adapter).

    Args:
        model_name: HuggingFace model path. If None, uses tier lookup.
        tier: Model tier.
        adapter_path: Path to saved LoRA adapter. If None, loads base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastVisionModel

    if model_name is None:
        model_name = get_model_info(tier)["hf_path"]

    logger.info(f"Loading model for inference: {model_name}")

    if adapter_path:
        logger.info(f"Loading with adapter from: {adapter_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            adapter_path,
            load_in_4bit=True,
            max_seq_length=4096,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            max_seq_length=4096,
        )

    # Set to inference mode
    FastVisionModel.for_inference(model)

    return model, tokenizer
