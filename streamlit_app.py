import streamlit as st
import replicate
from io import BytesIO

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust: Uncensored AI Image Generator", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token from Secrets ──────────────────────────────────────────────────
try:
    REPLICATE_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
except KeyError:
    st.error("Replicate API token not found in Streamlit secrets.")
    st.stop()

replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Model Options ───────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Realistic Vision v5.1 (lucataco)": {
        "ref": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "input_keys": ["prompt"]
    },
    "Illust3relustion (aisha-ai)": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "input_keys": ["prompt", "steps", "refiner", "upscale", "scheduler", "refiner_strength", "prompt_conjunction"],
        "defaults": {
            "steps": 20,
            "refiner": True,
            "upscale": "x2",
            "scheduler": "Euler a beta",
            "refiner_strength": 0.6,
            "prompt_conjunction": True
        }
    }
}

# ─── Preset Prompts ──────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {"example": ""},
    "Blowjob POV": {
        "example": (
            "Princess Jasmine from Alladin, huge boobs, pierced nipples, gigantic ass, "
            "hour-glass body, dark long wavy hair, wearing nothing but blue fishnet stockings, "
            "giving a blowjob, semen dripping all over her face and body, drooling, POV from above, "
            "8K realistic detailed photo"
        )
    },
    "Facial Close-up": {
        "example": (
            "Anime girl with open mouth, cum on tongue, facial expression of bliss, "
            "8K ultra detailed realistic close-up"
        )
    },
    "Doggy Style Outdoor": {
        "example": (
            "Woman in doggy style on grass, nude, large breasts, perfect ass, penis visible, "
            "sunlight, 8K realism"
        )
    }
}

# ─── Image Generation Function ───────────────────────────────────────────────
def generate_image(prompt: str, model_key: str) -> BytesIO:
    model_info = MODEL_OPTIONS[model_key]
    model_ref = model_info["ref"]
    inputs = {key: model_info.get("defaults", {}).get(key, None) for key in model_info["input_keys"]}
    inputs["prompt"] = prompt

    # Filter out keys with None values
    inputs = {k: v for k, v in inputs.items() if v is not None}

    outputs = replicate_client.run(model_ref, input=inputs)

    # Handle single or multiple outputs
    if hasattr(outputs, "read"):  # FileOutput type
        return BytesIO(outputs.read())
    else:  # iterable of file-like outputs
        first = next(iter(outputs), None)
        return BytesIO(first.read()) if first else None

# ─── UI Layout ───────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys()))
prompt = st.text_area("Enter your custom prompt here:" if preset == "Custom" else "Prompt",
                      PRESETS[preset]["example"] if preset != "Custom" else "", height=150)

model_choice = st.selectbox("Choose a model:", list(MODEL_OPTIONS.keys()))

if st.button("🖼️ Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Generating your image..."):
            try:
                image_bytes = generate_image(prompt, model_choice)
                if image_bytes:
                    st.subheader("Generated Image")
                    st.image(image_bytes, use_container_width=True)
                else:
                    st.error("Image generation failed. No image returned.")
            except Exception as e:
                st.error(f"Image generation failed: {e}")
