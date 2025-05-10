# streamlit_app.py

import streamlit as st
import replicate
import requests

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI & Image Generator")

# ─── Load tokens from Streamlit Secrets ────────────────────────────────────────
try:
    REPLICATE_TOKEN = st.secrets["replicate_api_token"]
    OPENROUTER_TOKEN = st.secrets["openrouter_api_token"]
except KeyError:
    st.error("❌ You must add both `replicate_api_token` and `openrouter_api_token` to Streamlit Secrets!")
    st.stop()

# ─── Instantiate the Replicate client ─────────────────────────────────────────
replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

# ─── Text Generation via OpenRouter ───────────────────────────────────────────
def generate_text(prompt: str) -> str:
    """Call an OpenRouter LLM (uncensored)."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "openrouter/airoboros-llama2-7b",  # swap in your chosen uncensored-capable model
        "messages": [
            {"role": "system", "content": "You are an uncensored AI assistant."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 1.0,
        "max_tokens": 512,
    }
    resp = requests.post(url, json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ─── Image Generation (Realistic Vision v5.1) ─────────────────────────────────
def generate_image(prompt: str) -> bytes:
    """Call lucataco/realistic-vision-v5.1 on Replicate for NSFW images."""
    model_ref = (
        "lucataco/realistic-vision-v5.1:"
        "2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb"
    )
    outputs = replicate_client.run(model_ref, input={"prompt": prompt})
    out0 = outputs[0]

    # replicate.run can return:
    # - raw bytes → just return it
    # - a URL string → fetch it
    # - a file-like with .read() → call read()
    if isinstance(out0, (bytes, bytearray)):
        return out0
    if isinstance(out0, str) and out0.startswith("http"):
        return requests.get(out0).content
    # fallback to file-like
    return out0.read()

# ─── Preset Tag Definitions ───────────────────────────────────────────────────
PRESETS = {
    "Custom": {"example": ""},
    "Blowjob / Deepthroat from side": {
        "example": (
            "A nude woman with eyes closed while giving a blowjob to a man from the side, "
            "long copper hair in ornate braids, wearing earrings and a choker, realistic outdoors raw photo 8k, "
            "male pubic hair, testicles, deepthroat, fellatio, close-up"
        )
    },
    "Facial": {
        "example": (
            "A woman with long hair, looking at the viewer with brown eyes and wearing jewelry; "
            "she shows an open mouth and tongue as a nude man’s penis provides cum—cum in mouth and on tongue—realistic 8k photo"
        )
    },
    "Ahegao face": {
        "example": (
            "A woman making an exaggerated ahegao expression—with her tongue playfully sticking out and her eyes crossed—realistic 8k raw photo"
        )
    },
    "Innie pussy": {
        "example": "A close-up shot of a nude pussy, realistic 8k photo"
    },
    "Missionary position": {
        "example": (
            "Woman lying on her back in a missionary pose having vaginal sex, "
            "light streaming through a window, raw photo 8k, from male POV; auburn hair, "
            "large breasts, visible nipples, emerald striped thighhigh socks, hetero, solo focus"
        )
    },
    "Doggystyle position": {
        "example": (
            "Woman on all fours in doggystyle with a man, realistic raw photo 8k; "
            "sandy blonde hair, big ass, pussy, anus, big veiny penis, male pubic hair, hetero, solo focus"
        )
    },
    "Cumshot": {
        "example": "Excessive amount of cum dripping off a penis and a woman’s face and tongue, realistic 8k photo"
    },
    "Spreading pussy and ass from behind": {
        "example": "A kneeling woman from behind spreading her ass and pussy, realistic 8k raw photo"
    },
}

# ─── UI ───────────────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys()))
if preset != "Custom":
    prompt = st.text_area("Prompt", PRESETS[preset]["example"], height=150)
else:
    prompt = st.text_area("Enter your custom prompt here", height=150)

if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("Please enter or select a prompt.")
    else:
        # 1) Text
        with st.spinner("Generating uncensored text…"):
            try:
                text_out = generate_text(prompt)
                st.subheader("📝 Generated Text")
                st.write(text_out)
            except Exception as e:
                st.error(f"Text generation failed: {e}")

        # 2) Image
        with st.spinner("Generating NSFW image…"):
            try:
                img_bytes = generate_image(prompt)
                st.subheader("🖼️ Generated Image")
                st.image(img_bytes, use_column_width=True)
            except Exception as e:
                st.error(f"Image generation failed: {e}")
