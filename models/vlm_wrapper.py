"""
UnifiedVLMWrapper: one interface for loading and generating from any VLM
in the registry, hiding processor differences between model families.
"""
import time
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from models.registry import get_model_entry
from core.logging import get_logger

logger = get_logger(__name__)


class UnifiedVLMWrapper:
    def __init__(self, model_id: str, load_in_4bit: bool = True, device: Optional[str] = None):
        self.model_id = model_id
        self.entry = get_model_entry(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_latency_ms: Optional[float] = None

        hf_path = self.entry["hf_path"]
        lora_path = self.entry.get("lora_path")

        logger.info(f"Loading base model '{hf_path}' for model_id='{model_id}' (4bit={load_in_4bit})")

        load_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self.model = AutoModelForVision2Seq.from_pretrained(hf_path, trust_remote_code=True, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)

        if lora_path:
            logger.info(f"Attaching LoRA adapter from: {lora_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()

    def _build_chat_input(self, image: Image.Image, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        return inputs

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        inputs = self._build_chat_input(image, system_prompt, user_prompt)

        start = time.time()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        self.last_latency_ms = (time.time() - start) * 1000

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    def generate_batch(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> List[str]:
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            for img in batch:
                results.append(self.generate(img, system_prompt, user_prompt, max_new_tokens))
        return results