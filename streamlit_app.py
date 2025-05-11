import streamlit as st
import replicate

# ─── Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Load Replicate Token from Secrets ───────────────────────────────────
REPLICATE_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Supported Models ────────────────────────────────────────────────────
MODELS = {
    "Realistic Vision v5.1": {
        "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "params": lambda prompt, neg: {"prompt": prompt, "negative_prompt": neg}
    },
    "Illust3Relustion": {
        "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "params": lambda prompt, neg: {
            "prompt": prompt,
            "refiner": True,
            "refiner_strength": 0.6,
            "scheduler": "Euler a beta",
            "upscale": "x2",
            "steps": 20,
            "prompt_conjunction": True
        }
    },
    "Realism XL": {
        "id": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "params": lambda prompt, neg: {"prompt": prompt}
    }
}

# ─── Presets (Editable) ───────────────────────────────────────────────────
PRESETS = {
    "Custom": {"prompt": "", "neg_prompt": ""},
    "Blowjob POV": {
        "prompt": "realistic 8k POV blowjob from above, long brown hair, nude woman, cum on face, drooling, mouth open, looking up, intense lighting",
        "neg_prompt": "bad anatomy, blurry, watermark, ugly face"
    },
    "Facial Close-Up": {
        "prompt": "close-up of woman's face with cum dripping from her lips and chin, 8K detailed photo",
        "neg_prompt": "cartoon, 3d, blurry, bad eyes"
    }
}

# ─── UI Input ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Settings")
    model_name = st.selectbox("Choose Model", list(MODELS.keys()))
    preset_name = st.selectbox("Choose a Scene Preset", list(PRESETS.keys()))
    
prompt = st.text_area("Prompt", PRESETS[preset_name]["prompt"], height=150)
neg_prompt = st.text_area("Negative Prompt (optional)", PRESETS[preset_name]["neg_prompt"], height=100)

if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                model_info = MODELS[model_name]
                inputs = model_info["params"](prompt.strip(), neg_prompt.strip())
                output = client.run(model_info["id"], input=inputs)
                for img in output:
                    st.image(img, caption=model_name, use_container_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
