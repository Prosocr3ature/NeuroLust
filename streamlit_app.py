import streamlit as st
import replicate

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: NSFW Image Generator")

# ─── API Token (from Streamlit secrets) ───────────────────────────────────────
REPLICATE_TOKEN = st.secrets["replicate_api_token"]
client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Model Selection ──────────────────────────────────────────────────────────
model_options = {
    "Realistic Vision v5.1": {
        "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "params": lambda prompt: {"prompt": prompt}
    },
    "Illust3relustion": {
        "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "params": lambda prompt: {
            "prompt": prompt,
            "steps": 20,
            "refiner": True,
            "upscale": "x2",
            "scheduler": "Euler a beta",
            "refiner_strength": 0.6,
            "prompt_conjunction": True
        }
    }
}
selected_model = st.selectbox("Choose a model:", list(model_options.keys()))

# ─── Prompt Input ─────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a scene preset:", ["Custom", "Blowjob", "Missionary", "Facial", "Ahegao"])
examples = {
    "Blowjob": "A nude woman with copper hair giving a deep blowjob from the side, ornate braids, choker, male pubic hair, 8K photo",
    "Missionary": "Woman in missionary position, spread legs, visible pussy and penis, emerald striped socks, 8K realism",
    "Facial": "Cum on woman's tongue and face, mouth open, realistic 8K detail, brown eyes and jewelry",
    "Ahegao": "Woman making exaggerated ahegao face, tongue out, eyes crossed, 8K raw closeup",
    "Custom": ""
}
default_prompt = examples[preset]
prompt = st.text_area("Enter your prompt", default_prompt, height=160)

# ─── Generate Button ──────────────────────────────────────────────────────────
if st.button("🚀 Generate NSFW Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            try:
                model_info = model_options[selected_model]
                output = client.run(model_info["id"], input=model_info["params"](prompt))

                for img in output:
                    if isinstance(img, str) and img.startswith("http"):
                        st.image(img, use_container_width=True)
                    elif hasattr(img, "read"):
                        st.image(img.read(), use_container_width=True)
                    else:
                        st.error("Unsupported image format.")
            except Exception as e:
                st.error(f"Image generation failed: {e}") 
