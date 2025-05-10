import streamlit as st
import replicate

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI & Image Generator")

# ─── Load API Key ────────────────────────────────────────────────────────────
replicate_token = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=replicate_token)

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

# ─── Image Generation (Flux.1dev NSFW) ─────────────────────────────────────────
def generate_image(prompt: str, scheduler: str) -> str:
    """
    Calls aisha-ai-official/flux.1dev-uncensored-msfluxnsfw-v3 on Replicate for NSFW.
    """
    model = (
        "aisha-ai-official/"
        "flux.1dev-uncensored-msfluxnsfw-v3:"
        "b477d8fc3a62e591c6224e10020538c4a9c340fb1f494891aff60019ffd5bc48"
    )
    inputs = {
        "prompt": prompt,
        "scheduler": scheduler
    }
    # replicate.run returns a list of binary file-like objects
    outputs = client.run(model, input=inputs)
    # Write the first image to a local URL and return it
    # Streamlit can accept raw bytes from a file-like, so we can write to disk or use st.image directly
    img_data = outputs[0].read()
    return img_data

# ─── Preset Tag Definitions ───────────────────────────────────────────────────
PRESETS = {
    "Custom": {"tags": [], "example": ""},
    "Blowjob / Deepthroat from side": {
        "tags": [
            "nude woman giving a blowjob to a man view from side",
            "male pubic hair", "testicles", "solo focus",
            "deepthroat", "penis", "tongue out", "fellatio", "close-up"
        ],
        "example": (
            "A nude woman with eyes closed while giving a blowjob to a man from the side, "
            "she has long, lustrous copper hair in ornate braids, wearing earrings and a choker, "
            "realistic outdoors raw photo 8k, male pubic hair, testicles, solo focus, "
            "deepthroat, penis, tongue out, fellatio, close-up"
        )
    },
    "Facial": {
        "tags": ["open mouth", "tongue", "facial", "penis", "cum", "solo focus", "cum on tongue"],
        "example": (
            "A woman with long hair, looking at the viewer with brown eyes and wearing jewelry; "
            "she shows an open mouth and tongue as a nude man’s penis provides cum—"
            "cum in mouth and cum on tongue—realistic raw photo"
        )
    },
    "Ahegao face": {
        "tags": ["ahegao face", "tongue sticking out", "eyes crossed"],
        "example": (
            "A woman making an exaggerated ahegao expression—with her tongue playfully "
            "sticking out and her eyes crossed—realistic 8k raw photo"
        )
    },
    "Innie pussy": {
        "tags": ["nude", "pussy"],
        "example": "A close-up shot of a nude pussy, realistic 8k photo"
    },
    "Missionary position": {
        "tags": [
            "missionary", "vaginal sex", "male POV", "spread legs",
            "lying", "pussy", "penis", "hetero", "solo focus"
        ],
        "example": (
            "Woman lying on her back in a missionary pose having vaginal sex, "
            "light streaming through a window, raw photo 8k, from male POV; auburn hair, "
            "large breasts, visible nipples, emerald striped thighhigh socks, "
            "spread legs reveal pussy and penis, hetero, solo focus"
        )
    },
    "Doggystyle position": {
        "tags": [
            "doggystyle", "vaginal sex", "male pov", "pussy", "anus",
            "penis", "male pubic hair", "hetero", "solo focus"
        ],
        "example": (
            "Woman on all fours in doggystyle position with a man, realistic raw photo 8k; "
            "sandy blonde hair, big ass, pussy, anus, big veiny penis, male pubic hair, hetero, solo focus"
        )
    },
    "Cumshot": {
        "tags": ["cum"],
        "example": "Excessive amount of cum dripping off a penis and a woman’s face and tongue, realistic 8k photo"
    },
    "Spreading pussy and ass from behind": {
        "tags": [
            "woman", "barefoot", "pussy", "from behind", "feet", "kneeling",
            "anus", "toes", "back", "soles", "spread ass", "hands on own ass"
        ],
        "example": "A kneeling woman from behind spreading her ass and pussy, realistic 8k raw photo"
    }
}

# ─── UI ───────────────────────────────────────────────────────────────────────
preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys()))
if preset != "Custom":
    st.markdown(f"**Tags:** {', '.join(PRESETS[preset]['tags'])}")
    prompt = st.text_area("Prompt", PRESETS[preset]["example"], height=150)
else:
    prompt = st.text_area("Enter your custom prompt here", height=150)

scheduler = st.text_input("Scheduler", value="Euler flux beta")

if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("Please enter or select a prompt.")
    else:
        # Generate text
        with st.spinner("Generating uncensored text..."):
            text_out = generate_text(prompt)
            st.subheader("📝 Generated Text")
            st.write(text_out)

        # Generate image
        with st.spinner("Generating NSFW image..."):
            img_bytes = generate_image(prompt, scheduler)
            st.subheader("🖼️ Generated Image")
            st.image(img_bytes, use_column_width=True)
