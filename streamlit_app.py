import streamlit as st
import replicate

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Load Replicate API Key ───────────────────────────────────────────────────
try:
    REPLICATE_TOKEN = st.secrets["replicate_api_token"]
except KeyError:
    st.error("Replicate API token not found in Streamlit secrets.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Preset Tags ──────────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {"example": ""},
    "Blowjob POV": {
        "example": (
            "Princess Jasmine from Alladin, huge boobs, pierced nipples, gigantic ass, hour-glass body, "
            "dark long wavy hair, wearing nothing but blue fishnet stockings, giving a blowjob, semen all over her "
            "body dripping while she is drooling, 8K realistic detailed POV photo from above"
        )
    },
    "Facial Cumshot": {
        "example": (
            "A woman with messy cum on her face, tongue out, open mouth, brown hair, nude, realistic lighting, "
            "high-resolution close-up photo"
        )
    },
    "Doggystyle Scene": {
        "example": (
            "Nude blonde girl bent over in doggystyle, striped thigh-high socks, back arched, visible pussy and anus, "
            "realistic 8K lighting, solo focus, big ass"
        )
    },
}

# ─── Model Selection ───────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Realistic Vision (photo NSFW)": {
        "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "supports_extra": False
    },
    "Illust3Relustion (anime NSFW)": {
        "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "supports_extra": True
    }
}

selected_model_label = st.selectbox("Select Image Model", list(MODEL_OPTIONS.keys()))
model_info = MODEL_OPTIONS[selected_model_label]
model_ref = model_info["id"]

# ─── Prompt UI ─────────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys()))
if preset != "Custom":
    prompt = st.text_area("Prompt", PRESETS[preset]["example"], height=150)
else:
    prompt = st.text_area("Enter your custom prompt here", height=150)

# ─── Generate Button ──────────────────────────────────────────────────────────
if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.warning("Please enter or select a prompt.")
        st.stop()

    st.subheader("🖼️ Generated Image")
    with st.spinner("Generating image..."):

        # Prepare input based on model capabilities
        input_data = {"prompt": prompt}
        if model_info["supports_extra"]:
            input_data.update({
                "steps": 20,
                "refiner": True,
                "upscale": "x2",
                "scheduler": "Euler a beta",
                "refiner_strength": 0.6,
                "prompt_conjunction": True
            })

        try:
            output = replicate_client.run(model_ref, input=input_data)

            for i, img in enumerate(output):
                if isinstance(img, str) and img.startswith("http"):
                    st.image(img, caption=f"Image {i + 1}", use_container_width=True)
                elif hasattr(img, "url"):
                    st.image(img.url, caption=f"Image {i + 1}", use_container_width=True)
                elif hasattr(img, "read"):
                    st.image(img.read(), caption=f"Image {i + 1}", use_container_width=True)
                else:
                    st.error("Unknown image output format.")
        except Exception as e:
            st.error(f"Image generation failed: {e}")
