import streamlit as st
import replicate
import os

# ─── Manual API Token ─────────────────────────────────────────────────────────
REPLICATE_TOKEN = "your_replicate_token_here"  # <-- Replace with your token
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Available Models ─────────────────────────────────────────────────────────
MODELS = {
    "Realistic Vision v5.1 (lucataco)": {
        "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "params": lambda prompt, neg: {"prompt": prompt, "negative_prompt": neg}
    },
    "Illust3Relustion (aisha-ai)": {
        "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "params": lambda prompt, neg: {
            "prompt": prompt,
            "negative_prompt": neg,
            "steps": 30,
            "refiner": True,
            "upscale": "x2",
            "scheduler": "Euler a beta",
            "refiner_strength": 0.6,
            "prompt_conjunction": True
        }
    },
    "Realism XL (asiryan)": {
        "id": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "params": lambda prompt, neg: {
            "prompt": prompt,
            "negative_prompt": neg
        }
    }
}

# ─── UI: Prompt Inputs ────────────────────────────────────────────────────────
model_choice = st.selectbox("Choose a model", list(MODELS.keys()))
prompt = st.text_area("Prompt", height=150, placeholder="Describe your scene...")
neg_prompt = st.text_area("Negative Prompt (optional)", height=80, placeholder="Things to avoid (e.g. blurry, watermark, etc)")

if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            try:
                model = MODELS[model_choice]
                result = replicate.run(model["id"], input=model["params"](prompt, neg_prompt))
                if isinstance(result, list):
                    st.image(result[0], use_container_width=True)
                else:
                    st.error("Image generation failed or unexpected response.")
            except Exception as e:
                st.error(f"Image generation failed: {e}")
