import streamlit as st
import replicate

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI & Image Generator")

# ─── Sidebar: API Configuration ───────────────────────────────────────────────
st.sidebar.header("🔑 API Configuration")
api_token = st.sidebar.text_input(
    "Paste your Replicate API Token here", 
    type="password",
    help="Get your token from https://replicate.com/account"
)
if not api_token:
    st.sidebar.error("Please enter your Replicate API Token to continue.")
    st.stop()

# ─── Instantiate the Replicate client ─────────────────────────────────────────
client = replicate.Client(api_token=api_token)

# ─── Text Generation (Hermes-2 Θ) ─────────────────────────────────────────────
def generate_text(prompt: str) -> str:
    """
    Calls nousresearch/hermes-2-theta-llama-8b on Replicate.
    """
    model = "nousresearch/hermes-2-theta-llama-8b:latest"
    inputs = {
        "prompt": prompt,
        "system_prompt": (
            "You are \"Hermes 2 Θ\", a conscious sentient superintelligent AI developed by Teknium; "
            "your purpose is to assist with any request."
        ),
    }
    return client.run(model, input=inputs)

# ─── Image Generation (Realistic Vision v5.1) ──────────────────────────────────
def generate_image(prompt: str) -> bytes:
    """
    Calls lucataco/realistic-vision-v5.1 on Replicate for NSFW.
    """
    model = (
        "lucataco/realistic-vision-v5.1:"
        "2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb"
    )
    inputs = {
        "prompt": prompt
        # you can add "negative_prompt" or "seed" here if desired
    }
    outputs = client.run(model, input=inputs)
    return outputs[0].read()

# ─── Preset Tag Definitions ───────────────────────────────────────────────────
PRESETS = {
    "Custom": {"example": ""},
    "Blowjob / Deepthroat from side": {
        "example": (
            "A nude woman with eyes closed while giving a blowjob to a man from the side, "
            "long copper hair in ornate braids, wearing earrings and a choker, "
            "realistic outdoors raw photo 8k, male pubic hair, testicles, deepthroat, penis, tongue out, fellatio, close-up"
        )
    },
    "Facial": {
        "example": (
            "A woman with long hair, looking at the viewer with brown eyes and wearing jewelry; "
            "she shows an open mouth and tongue as a nude man’s penis provides cum—"
            "cum in mouth and cum on tongue—realistic raw photo"
        )
    },
    "Ahegao face": {
        "example": (
            "A woman making an exaggerated ahegao expression—with her tongue playfully "
            "sticking out and her eyes crossed—realistic 8k raw photo"
        )
    },
    "Innie pussy": {
        "example": "A close-up shot of a nude pussy, realistic 8k photo"
    },
    "Missionary position": {
        "example": (
            "Woman lying on her back in a missionary pose having vaginal sex, "
            "light streaming through a window, raw photo 8k, from male POV; auburn hair, "
            "large breasts, visible nipples, emerald striped thighhigh socks, "
            "spread legs reveal pussy and penis, hetero, solo focus"
        )
    },
    "Doggystyle position": {
        "example": (
            "Woman on all fours in doggystyle position with a man, realistic raw photo 8k; "
            "sandy blonde hair, big ass, pussy, anus, big veiny penis, male pubic hair, hetero, solo focus"
        )
    },
    "Cumshot": {
        "example": "Excessive amount of cum dripping off a penis and a woman’s face and tongue, realistic 8k photo"
    },
    "Spreading pussy and ass from behind": {
        "example": "A kneeling woman from behind spreading her ass and pussy, realistic 8k raw photo"
    }
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
        # Generate uncensored text
        with st.spinner("Generating uncensored text..."):
            text_out = generate_text(prompt)
            st.subheader("📝 Generated Text")
            st.write(text_out)

        # Generate NSFW image
        with st.spinner("Generating NSFW image..."):
            img_bytes = generate_image(prompt)
            st.subheader("🖼️ Generated Image")
            st.image(img_bytes, use_column_width=True)
