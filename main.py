# app.py  —— Arabic ➝ English Translator (Gradio)
print("📢 app.py started")

import gradio as gr
import torch, os, psutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ────────────────────────── Utils ──────────────────────────
def get_memory_usage_mb() -> float:
    """Return current RAM usage of this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ────────────────────────── Model load (one‑time) ──────────
print("🔄 Loading model + adapter …")
adapter_path = "adapters/ar_en/"
config       = PeftConfig.from_pretrained(adapter_path)

device       = "cuda" if torch.cuda.is_available() else "cpu"

base_model   = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype = torch.float16 if device == "cuda" else None,
).to(device)

model        = PeftModel.from_pretrained(base_model, adapter_path).to(device)
tokenizer    = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

print(f"✅ Model ready (RAM: {get_memory_usage_mb():.2f} MB)")

# ────────────────────────── Translation fn ─────────────────
def translate(text: str) -> str:
    text = text.strip()
    if not text:
        return "⚠️ Please enter some Arabic text."
    
    inputs  = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ────────────────────────── Gradio UI ──────────────────────
demo = gr.Interface(
    fn          = translate,
    inputs      = gr.Textbox(label="📝 Enter Arabic text"),
    outputs     = gr.Textbox(label="🔤 Translated Text (English)"),
    title       = "🌍 Arabic ➝ English Translator",
    description = f"Model loaded · RAM usage {get_memory_usage_mb():.1f} MB",
    examples    = ["بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ", "السلام عليكم ورحمة الله"]
)

if __name__ == "__main__":
    demo.launch()
