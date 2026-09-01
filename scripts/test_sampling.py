import torch
from unsloth import FastVisionModel
from data.loader import load_processed_dataset
from data.prompt_templates import SYSTEM_PROMPT, get_prompt_for_task
from qwen_vl_utils import process_vision_info

# Task this smoke test exercises. Any registered task works; the default
# matches the pipeline most often being debugged.
TASK = "violations_only"


# Make sure this points to your real SFT adapter from yesterday!
ADAPTER = "/home/nabeel.shan/vlm-finetuning-project1/checkpoints/qwen3vl-8b/vo-sft-8b-v3/final"
BASE = "unsloth/Qwen3-VL-8B-Instruct"

print("Loading model and adapter...")
model, tokenizer = FastVisionModel.from_pretrained(BASE, load_in_4bit=True, max_seq_length=3200)
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER)
FastVisionModel.for_inference(model)

# Print the model's OWN generation_config before we touch anything
print("\n=== Model's default generation_config ===")
print(model.generation_config)

ds = load_processed_dataset()
sample = ds["train"][0]
img = sample["image"]

msgs = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "image", "image": img},
                                  {"type": "text", "text": get_prompt_for_task(TASK)}]},
]
text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
img_in, vid_in = process_vision_info(msgs)
inputs = tokenizer(text=text, images=img_in, videos=vid_in, return_tensors="pt", padding=True).to(model.device)

print("\n=== Generating 8 samples with temperature=0.9, top_p=0.95, do_sample=True ===")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=8,
        use_cache=True,
    )

input_len = inputs["input_ids"].shape[1]
texts = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)

print("\n=== RESULTS ===")
for i, t in enumerate(texts):
    print(f"\n--- sample {i} ---")
    print(t[:150]) # Truncate output to first 150 characters to keep it readable

print("\nIf all 8 samples are IDENTICAL, sampling is completely broken upstream.")
print("If the 8 samples are DIFFERENT, sampling works, and the GRPO fix is guaranteed to work!")
