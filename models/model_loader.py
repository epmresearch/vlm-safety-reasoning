"""
Centralized model loading for training and inference.

Uses Unsloth's FastVisionModel for optimized LoRA fine-tuning
with frozen vision encoder.
"""
from typing import Any, Dict, Optional, Tuple

from core.config import load_config, load_task_config
from core.logging import get_logger

logger = get_logger(__name__)


def get_model_info(tier: Optional[str] = None) -> Dict[str, Any]:
    """Returns model info dict for the given tier.

    Args:
        tier: One of "2b", "4b", "8b". If None, reads active_tier from config.

    Returns:
        Dict with hf_path, short_name, size_label.
    """
    config = load_config()
    if tier is None:
        tier = config.get("active_tier", "2b")
        
    models = config.get("models", {})
    if tier not in models:
        raise ValueError(
            f"Unknown model tier '{tier}'. Choose from: {list(models.keys())}"
        )
    return models[tier]


def get_batch_config(tier: Optional[str] = None) -> Dict[str, int]:
    """Returns per-model batch size config for SFTTrainer.

    Args:
        tier: One of "2b", "4b", "8b". If None, reads active_tier from config.

    Returns:
        Dict with per_device_train_batch_size and gradient_accumulation_steps.
    """
    model_info = get_model_info(tier)
    return {
        "per_device_train_batch_size": model_info.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": model_info.get("gradient_accumulation_steps", 1),
    }


def load_model_for_training(
    model_name: Optional[str] = None,
    tier: Optional[str] = None,
    sft_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """Loads a model with LoRA adapters for SFT training.

    Args:
        model_name: HuggingFace model path. If None, uses tier lookup.
        tier: Model tier for batch config lookup. If None, uses default config tier.
        sft_cfg: SFT config dict (from configs/sft.yaml). If None, uses defaults.

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastVisionModel

    if model_name is None:
        model_name = get_model_info(tier)["hf_path"]

    # Defaults if sft_cfg not provided
    load_in_4bit = True
    max_seq_length = 2048
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
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

    # Cap image resolution fed to the vision encoder (training memory safety)
    apply_pixel_bounds(
        tokenizer,
        min_pixels=sft_cfg.get("image_min_pixels") if sft_cfg else None,
        max_pixels=sft_cfg.get("image_max_pixels") if sft_cfg else None,
    )

    # Unsloth requires this explicit flip into training mode.
    FastVisionModel.for_training(model)

    return model, tokenizer, get_model_info(tier)


def load_model_for_inference(
    model_name: Optional[str] = None,
    tier: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
) -> Tuple:
    """Loads a model for inference (with optional LoRA adapter).

    Args:
        model_name: HuggingFace model path. If None, uses tier lookup.
        tier: Model tier. If None, uses default config tier.
        adapter_path: Path to saved LoRA adapter. If None, loads base model.
        max_seq_length: Maximum sequence length for the model.

    Returns:
        Tuple of (model, tokenizer).
    """
    from unsloth import FastVisionModel
    from core.config import load_config

    # Auto-load defaults from SFT config to ensure inference matches training constraints
    cfg = load_config(training_kind="sft")
    task_cfg = load_task_config("unified")

    if model_name is None:
        model_name = get_model_info(tier)["hf_path"]

    if max_seq_length is None:
        max_seq_length = task_cfg.get("inference_max_seq_length", 2816)

    if image_min_pixels is None:
        image_min_pixels = cfg.get("image_min_pixels")
        
    if image_max_pixels is None:
        image_max_pixels = cfg.get("image_max_pixels")

    logger.info(f"Loading model for inference: {model_name} with max_seq_length={max_seq_length}")

    if adapter_path:
        logger.info(f"Loading with adapter from: {adapter_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            adapter_path,
            load_in_4bit=True,
            max_seq_length=max_seq_length,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            max_seq_length=max_seq_length,
        )

    # Cap image resolution for inference memory safety
    apply_pixel_bounds(
        tokenizer,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )

    # Set to inference mode
    FastVisionModel.for_inference(model)

    return model, tokenizer, get_model_info(tier)


def apply_pixel_bounds(
    tokenizer,
    min_pixels: int = None,
    max_pixels: int = None,
) -> None:
    image_processor = getattr(tokenizer, "image_processor", None)
    if image_processor is None:
        logger.warning("No image_processor on tokenizer — cannot apply pixel bounds.")
        return

    # Fallback to existing values if None are provided
    current_min = min_pixels if min_pixels is not None else getattr(image_processor, "min_pixels", 200704)
    current_max = max_pixels if max_pixels is not None else getattr(image_processor, "max_pixels", 1204224)

    # Overwrite the size dictionary directly with the area bounds
    image_processor.size = {"min_pixels": current_min, "max_pixels": current_max}

    logger.info(f"Applied image pixel bounds via size dict: min={current_min}, max={current_max}")


def log_gpu_memory(tag: str = "") -> None:
    """One-shot GPU memory printout, useful right after model load and right
    before trainer.train() to sanity-check headroom."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(
            f"[GPU MEM{' ' + tag if tag else ''}] allocated={allocated:.2f}GB "
            f"reserved={reserved:.2f}GB total={total:.2f}GB free~={total - reserved:.2f}GB"
        )
    except Exception as e:
        logger.warning(f"Could not read GPU memory: {e}")