import streamlit as st
import replicate
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token ───────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Model Definitions & Schedulers ──────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0, "width": 768, "height": 1152,
        "schedulers": ["DPMSolverMultistep", "PNDM", "DDIM", "HeunDiscrete", "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER"],
        "scheduler": "DPMSolverMultistep"
    },
    "Aisha Illust3 Relustion": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40, "scale": 9.0, "width": 768, "height": 1152,
        "schedulers": ["PNDM", "DDIM", "DPMSolverMultistep", "HeunDiscrete", "K_EULER_ANCESTRAL", "K_EULER"],
        "scheduler": "PNDM"
    }
}

JASMINE_BASE = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge "
    "round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—adorned in sheer "
    "blue fishnet stockings, no underwear, pussy showing. Cinematic studio lighting, "
    "sharp focus, intricate textures, explicit nudity."
)

NEGATIVE_PROMPT = (
    "ugly face, poorly drawn hands, poorly drawn feet, blurry, lowres, extra limbs, "
    "cartoon, censored, watermark, jpeg artifacts, nsfw warning, error, cropped, out of frame, "
    "bad proportions, unnatural colors, bad anatomy, unrealistic, duplicate"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    user_action = st.text_area(
        "Describe Jasmine's pose/action (explicit allowed):",
        height=120,
        placeholder="e.g. spreading legs, doggystyle, licking, riding, POV blowjob"
    )

    # Dynamic sliders based on model's default
    steps = st.slider("Sampling Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    width = st.slider("Image Width (px)", 512, 1024, config["width"], step=64)
    height = st.slider("Image Height (px)", 512, 1536, config["height"], step=64)

    schedulers = config["schedulers"]
    scheduler = st.selectbox("Diffusion Scheduler", schedulers, index=schedulers.index(config["scheduler"]))

    # UNIQUE LABEL for extra negative prompt
    extra_negative = st.text_area(
        "Add extra negative prompt terms (optional):",
        value="",
        height=50
    )
    full_negative_prompt = NEGATIVE_PROMPT + (", " + extra_negative.strip() if extra_negative.strip() else "")

    seed_random = st.checkbox("Use random seed", value=True)
    seed = random.randint(1, 999999) if seed_random else st.number_input("Seed value", value=1337)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
if user_action.strip():
    action_phrase = f" She is {user_action.strip()}. This is explicit and must be shown with no censorship."
else:
    action_phrase = " She is posing erotically."

full_prompt = JASMINE_BASE + action_phrase

# ─── Generate Button ─────────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Generating with {model_choice}…")

    payload = {
        "prompt": full_prompt.strip(),
        "negative_prompt": full_negative_prompt.strip(),
        "width": int(width // 8) * 8,
        "height": int(height // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler
    }

    try:
        with st.spinner("Generating Jasmine..."):
            output = replicate.run(config["ref"], input=payload)

        def show_image(item):
            if hasattr(item, "read"):
                st.image(item.read(), use_container_width=True)
            elif hasattr(item, "url"):
                st.image(item.url, use_container_width=True)
            elif isinstance(item, str) and item.startswith("http"):
                st.image(item, use_container_width=True)
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
