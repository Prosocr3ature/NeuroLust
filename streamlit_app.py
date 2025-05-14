import streamlit as st
import replicate
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Authentication ────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Please add your Replicate API token to Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()
replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Constants & Models ────────────────────────────────────────────────────────
VALID_SCHEDULERS = [
    "DDIM", "PNDM", "DPMSolverMultistep", "HeunDiscrete",
    "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER",
]

# Improved default Jasmine prompt
DEFAULT_PROMPT = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge "
    "round breasts, a tiny waist, thick thighs, and a sculpted, prominent ass—adorned in sheer "
    "blue fishnet stockings, no underwear, and elegant nipple piercings. Cinematic studio lighting, "
    "sharp focus, intricate textures, explicit nudity."
)

IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0, "width": 768, "height": 1152,
        "scheduler": "DPMSolverMultistep",
        "preview": "https://replicate.delivery/pbxt/JqTfP3xup0D7quhKApwciUzEKCm36DyW7zHAcJ05ev8FuqaIA/out-0.png"
    },
    "Aisha Illust3 Relustion": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40, "scale": 8.5, "width": 768, "height": 1152,
        "preview": None
    },
    "Flux Uncensored": {
        "ref": "aisha-ai-official/flux.1dev-uncensored-jibmix:47f609a66d7fc3293a600467fad383fffa9ef5193c9c871d4f7f9b514a0afe3f",
        "extra_input": {
            "steps": 30,
            "width": 768,
            "height": 768,
            "cfg_scale": 3
        },
        "scale": 3.0,
        "preview": None
    }
}

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Generation Settings")

    # Model selector
    model_choice = st.selectbox("Model", list(IMAGE_MODELS.keys()))
    cfg = IMAGE_MODELS[model_choice]

    # Optional preview image
    if cfg.get("preview"):
        st.image(cfg["preview"], caption=model_choice, use_container_width=True)

    # Prompt (user writes any explicit scene/actions)
    prompt = st.text_area(
        "Prompt (include any actions, scenery, level of explicitness)",
        value=DEFAULT_PROMPT,
        height=200
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value="deformed, mutated, disfigured, bad anatomy, lowres, blurry, watermark",
        height=80
    )

    # Steps & Scale
    steps = st.slider("Inference Steps", 10, 100, value=cfg.get("steps", 40), step=5)
    if steps < 30:
        st.warning("Low step count may reduce quality (≥30 recommended).")
    scale = st.slider("Guidance Scale", 1.0, 20.0, value=cfg.get("scale", 7.5), step=0.5)
    if scale < 5.0:
        st.info("Low guidance scale yields more creative but less consistent output.")

    # Resolution
    width = st.slider("Width (px)", 256, 1536, value=cfg.get("width", 768), step=8)
    height = st.slider("Height (px)", 256, 1536, value=cfg.get("height", 1152), step=8)

    # Scheduler
    scheduler = st.selectbox("Scheduler", VALID_SCHEDULERS, index=VALID_SCHEDULERS.index(cfg.get("scheduler", "PNDM")))

    # Seed
    use_random = st.checkbox("Use random seed", value=True)
    seed = random.randint(1, 999_999) if use_random else st.number_input("Seed", value=1337)

# ─── Generation Handler ────────────────────────────────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.error("Prompt cannot be empty.")
        st.stop()
    st.info(f"Generating with {model_choice}…")

    # Build payload
    payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": width,
        "height": height,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler,
    }
    # Merge extra_input for Flux Uncensored
    payload.update(cfg.get("extra_input", {}))

    try:
        with st.spinner("Calling Replicate API…"):
            output = replicate_client.run(cfg["ref"], input=payload)

        # Display results
        def show_image(item):
            if hasattr(item, "read"):
                st.image(item.read(), use_container_width=True)
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
    except replicate.exceptions.ReplicateException as err:
        st.error(f"Model error: {err}")
        st.info("Try changing scheduler or resolution.")
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        st.info("Check your inputs and try again.")
