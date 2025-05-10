import streamlit as st
import replicate

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Load API Token ──────────────────────────────────────────────────────────
REPLICATE_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Preset Prompts ──────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {"example": ""},
    "Blowjob / Deepthroat from side": {
        "example": (
            "A nude woman with eyes closed while giving a blowjob to a man from the side, "
            "long copper hair in ornate braids, wearing earrings and a choker, "
            "realistic outdoors raw photo 8k, male pubic hair, testicles, deepthroat, "
            "penis, tongue out, fellatio, close-up"
        )
    },
    "Facial": {
        "example": (
            "A woman with long hair, looking at the viewer with brown eyes and wearing jewelry; "
            "she shows an open mouth and tongue as a nude man’s penis provides cum—"
            "cum in mouth and cum on tongue—realistic raw photo"
        )
    },
    "Doggystyle": {
        "example": (
            "Woman on all fours in doggystyle position with a man, realistic raw photo 8k; "
            "sandy blonde hair, big ass, pussy, anus, big veiny penis, male pubic hair, hetero, solo focus"
        )
    },
}

# ─── Image Model Selection ────────────────────────────────────────────────────
st.sidebar.header("Image Model")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ("Realistic Vision V5.1", "Illust3Relustion")
)

# ─── Prompt Input ─────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a preset scene:", list(PRESETS.keys()))
if preset != "Custom":
    prompt = st.text_area("Prompt", PRESETS[preset]["example"], height=150)
else:
    prompt = st.text_area("Enter your custom prompt here", height=150)

# ─── Image Generator ──────────────────────────────────────────────────────────
def generate_image_realistic_vision(prompt: str) -> str:
    model = "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb"
    output_url = client.run(model, input={"prompt": prompt})
    return output_url  # This is a URL string

def generate_image_illust3relustion(prompt: str) -> str:
    model = "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8"
    inputs = {
        "prompt": prompt,
        "steps": 20,
        "refiner": True,
        "upscale": "x2",
        "scheduler": "Euler a beta",
        "refiner_strength": 0.6,
        "prompt_conjunction": True
    }
    output_url = client.run(model, input=inputs)
    return output_url  # This is a URL string

# ─── Trigger Generation ───────────────────────────────────────────────────────
if st.button("🚀 Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            try:
                if model_choice == "Realistic Vision V5.1":
                    image_url = generate_image_realistic_vision(prompt)
                else:
                    image_url = generate_image_illust3relustion(prompt)

                st.subheader("🖼️ Generated Image")
                st.image(image_url, use_container_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
