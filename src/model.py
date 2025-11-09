
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import MODEL_CONFIG

def load_model():
    path = MODEL_CONFIG["model_path"]
    dtype = getattr(torch, MODEL_CONFIG.get("dtype", "float16"))
    device_map = MODEL_CONFIG.get("device_map", "auto")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device_map
    )
    return tokenizer, model


def generate_text(prompt: str):
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MODEL_CONFIG["max_new_tokens"],
        temperature=MODEL_CONFIG["temperature"],
        top_p=MODEL_CONFIG["top_p"],
        do_sample=MODEL_CONFIG["do_sample"]
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
