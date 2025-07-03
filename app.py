import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)  # in MB
st.set_page_config(page_title="Arabic to English Translator", layout="centered")
st.title("🌍 Arabic ➝ English Translator")

@st.cache_resource
def load_model():
    st.write("Loading base model")
    adapter_path = "adapters/ar_en/"
    config = PeftConfig.from_pretrained(adapter_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    st.write(f"RAM after base model: {get_memory_usage():.2f}MB")
    st.write("Loading Adapter")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    st.write(f"RAM after full model + tokenizer: {get_memory_usage():.2f}MB")
    return model, tokenizer

model, tokenizer = load_model()

src_text = st.text_area("📝 Enter Arabic text to translate:")

if st.button("Translate"):
    if not src_text.strip():
        st.warning("Please enter some Arabic text.")
    else:
        inputs = tokenizer(src_text, return_tensors="pt", truncation=True)
        output = model.generate(**inputs, max_new_tokens=128)
        translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        st.success("🔤 Translated Text (English):")
        st.write(translated_text)
