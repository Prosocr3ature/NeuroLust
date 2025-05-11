import streamlit as st
import replicate

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Validate Replicate Token ─────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets. Please add it as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Model Selection ──────────────────────────────────────────────────────────
MODELS = {
    "Realistic Vision v5.1": {
        "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "supports_negative_prompt": True,
        "supports_steps": True
    },
    "Realism XL (Uncensored)": {
        "id": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "supports_negative_prompt": True,
        "supports_steps": True
    }
}

# ─── Sidebar UI ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛠️ Model Settings")
    model_choice = st.selectbox("Choose image model:", list(MODELS.keys()))
    prompt = st.text_area("Prompt", height=150)
    negative_prompt = st.text_area("Negative Prompt (optional)", height=80)
    steps = st.slider("Steps", 20, 50, 35) if MODELS[model_choice]["supports_steps"] else None

# ─── Image Generation ─────────────────────────────────────────────────────────
if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    model_data = MODELS[model_choice]
    model_id = model_data["id"]

    input_payload = {"prompt": prompt}
    if steps:
        input_payload["steps"] = steps
    if model_data["supports_negative_prompt"] and negative_prompt.strip():
        input_payload["negative_prompt"] = negative_prompt.strip()

    with st.spinner("Generating image..."):
        try:
            outputs = replicate_client.run(model_id, input=input_payload)
            for i, image_url in enumerate(outputs):
                st.image(image_url, caption=f"Result {i+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Image generation failed: {e}")
