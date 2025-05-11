import streamlit as st
import replicate
import os
import requests

# ─── API Token from Streamlit Secrets ─────────────────────────────────────────
REPLICATE_TOKEN = st.secrets.get("replicate_api_token", "")

if not REPLICATE_TOKEN:
    st.error("Replicate API token not found in secrets. Please add it.")
    st.stop()

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
selected_model = st.selectbox("Choose a model", list(MODELS.keys()))
prompt = st.text_area("Prompt", height=150)
neg_prompt = st.text_area("Negative Prompt (optional)", height=80)

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating image..."):
        try:
            model_info = MODELS[selected_model]
            output = replicate.run(
                model_info["id"],
                input=model_info["params"](prompt, neg_prompt)
            )
            if isinstance(output, list):
                st.image(output[0], use_container_width=True)
            else:
                st.error("Unexpected output type from model.")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
