import streamlit as st
import replicate
import io
from PIL import Image
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token ───────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Available Models ────────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45,
        "scale": 8.0,
        "width": 768,
        "height": 1152,
        "scheduler": "DPMSolverMultistep",
        "preview": "https://replicate.delivery/pbxt/JqTfP3xup0D7quhKApwciUzEKCm36DyW7zHAcJ05ev8FuqaIA/out-0.png"
    },
    "Aisha Illust3 Relustion": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40,
        "scale": 9.0,
        "width": 768,
        "height": 1152,
        "scheduler": "PNDM",
        "preview": None
    }
}

# ─── Jasmine Base Prompt ─────────────────────────────────────────────────────
JASMINE_BASE = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge "
    "round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—adorned in sheer "
    "blue fishnet stockings, no underwear, pussy showing. Cinematic studio lighting, "
    "sharp focus, intricate textures, explicit nudity."
)

# ─── UI ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(IMAGE_MODELS.keys()))

    user_action = st.text_area(
        "What is Jasmine doing? (pose, action)",
        height=150,
        placeholder="e.g. spreading legs, doggystyle, licking, riding, POV blowjob"
    )

    full_prompt = f"{JASMINE_BASE} {user_action.strip()}" if user_action.strip() else JASMINE_BASE

    negative_prompt = st.text_area(
        "Negative Prompt",
        value="ugly face, poorly drawn hands, lowres, blurry, extra limbs, cartoon, watermark, censored, jpeg artifacts",
        height=80
    )

    seed_random = st.checkbox("Use random seed", value=True)
    seed = random.randint(1, 999999) if seed_random else st.number_input("Seed", value=1337)

# ─── Generate Button ─────────────────────────────────────────────────────────
if st.button("Generate"):
    if not user_action.strip():
        st.warning("Please describe what Jasmine is doing.")
        st.stop()

    st.info(f"Generating with {model_choice}…")
    config = IMAGE_MODELS[model_choice]

    payload = {
        "prompt": full_prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": config["width"],
        "height": config["height"],
        "guidance_scale": config["scale"],
        "seed": seed,
        "num_inference_steps": config["steps"],
        "scheduler": config["scheduler"]
    }

    try:
        with st.spinner("Generating Jasmine..."):
            output = replicate.run(config["ref"], input=payload)

        def show_image(item):
            if hasattr(item, "read"):
                st.image(item.read(), use_column_width=True)
            elif hasattr(item, "url"):
                st.image(item.url, use_column_width=True)
            elif isinstance(item, str) and item.startswith("http"):
                st.image(item, use_column_width=True)
            else:
                st.error("Unrecognized output format.")
                st.write(item)

        if isinstance(output, list):
            for img in output:
                show_image(img)
        else:
            show_image(output)

        st.success("Generation complete!")

    except Exception as e:
        st.error(f"Image generation failed: {e}")
