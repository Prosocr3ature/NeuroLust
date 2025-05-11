import streamlit as st
import replicate
import io
from PIL import Image

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token ───────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets. Please add it as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Available Models ────────────────────────────────────────────────────────
MODELS = {
    "Realistic Vision v5.1": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
    "Realism XL (Uncensored)": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
}

# ─── UI ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(MODELS.keys()))
    prompt = st.text_area("Prompt", height=150, placeholder="Enter your prompt...")
    negative_prompt = st.text_area("Negative Prompt (optional)", height=80)
    steps = st.slider("Sampling Steps", 20, 50, 35)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    st.info(f"Using model: {model_choice}")
    model_ref = MODELS[model_choice]

    input_payload = {
        "prompt": prompt.strip(),
        "steps": steps,
    }
    if negative_prompt.strip():
        input_payload["negative_prompt"] = negative_prompt.strip()

    with st.spinner("Generating image..."):
        try:
            outputs = replicate_client.run(model_ref, input=input_payload)
            for i, item in enumerate(outputs):
                if hasattr(item, "read"):  # Handle FileOutput (stream)
                    image_bytes = io.BytesIO(item.read())
                    image = Image.open(image_bytes)
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
                else:  # Handle URL string
                    st.image(item, caption=f"Image {i+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Image generation failed: {e}")
