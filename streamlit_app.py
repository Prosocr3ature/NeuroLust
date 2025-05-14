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

# ─── Constants & Model Registry ────────────────────────────────────────────────
VALID_SCHEDULERS = [
    "DDIM", "PNDM", "DPMSolverMultistep", "HeunDiscrete",
    "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER",
]

# A rock-solid default Jasmine prompt
DEFAULT_PROMPT = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge "
    "round breasts, a tiny waist, thick thighs, and a sculpted, prominent ass—adorned in sheer "
    "blue fishnet stockings, no underwear, and elegant nipple piercings. Cinematic studio lighting, "
    "sharp focus, intricate textures, explicit nudity."
)

IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref":  "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0,
        "width": 768, "height": 1152,
        "scheduler": "DPMSolverMultistep",
        "preview": "https://replicate.delivery/pbxt/JqTfP3xup0D7quhKApwciUzEKCm36DyW7zHAcJ05ev8FuqaIA/out-0.png"
    },
    "Aisha Illust3 Relustion": {
        "ref":  "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40, "scale": 9.0,  # default CFG
        "width": 768, "height": 1152,
        "preview": None
    },
    "Flux Uncensored": {
        "ref":  "aisha-ai-official/flux.1dev-uncensored-jibmix:47f609a66d7fc3293a600467fad383fffa9ef5193c9c871d4f7f9b514a0afe3f",
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

    # Preview (if available)
    if cfg.get("preview"):
        st.image(cfg["preview"], caption=model_choice, use_container_width=True)

    # Free-form prompt field
    prompt = st.text_area(
        "Prompt (fully describe scene, actions, level of explicitness)",
        value=DEFAULT_PROMPT,
        height=180
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value="deformed, mutated, disfigured, bad anatomy, lowres, blurry, watermark",
        height=80
    )

    # Common controls
    steps = st.slider("Steps", 10, 100, value=cfg.get("steps", 40), step=5)
    if steps < 30:
        st.warning("Steps <30 may reduce detail.")
    scale = st.slider("Guidance / CFG Scale", 1.0, 20.0, value=cfg.get("scale", 7.5), step=0.5)
    if scale < 5.0:
        st.info("Low scale → more creative, less consistent.")

    width = st.slider("Width (px)", 256, 1536, value=cfg.get("width", 768), step=8)
    height = st.slider("Height (px)", 256, 1536, value=cfg.get("height", 1152), step=8)

    scheduler = st.selectbox(
        "Scheduler", VALID_SCHEDULERS,
        index=VALID_SCHEDULERS.index(cfg.get("scheduler", "PNDM"))
    )

    # Aisha-specific toggles
    if model_choice == "Aisha Illust3 Relustion":
        refiner_strength = st.slider("Refiner Strength", 0.1, 1.0, 0.7, step=0.05)
        upscale = st.checkbox("Upscale ×2", value=True)
        # always use prompt_conjunction
        prompt_conjunction = True
    else:
        refiner_strength = None
        upscale = None
        prompt_conjunction = None

    # Seed
    use_random = st.checkbox("Use random seed", value=True)
    seed = random.randint(1, 999_999) if use_random else st.number_input("Seed", value=1337)

# ─── Generation Handler ────────────────────────────────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.error("Prompt cannot be empty."); st.stop()

    st.info(f"Generating with {model_choice}...")

    # Build payload
    if model_choice == "Flux Uncensored":
        # Flux already carries its own steps/width/height/cfg_scale
        payload = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            **cfg["extra_input"]
        }

    elif model_choice == "Aisha Illust3 Relustion":
        # Aisha uses raw "steps" + "cfg_scale"
        payload = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "steps": steps,
            "cfg_scale": scale,
            "width": width,
            "height": height,
            "scheduler": scheduler,
            "refiner": True,
            "refiner_strength": refiner_strength,
            "prompt_conjunction": prompt_conjunction
        }
        if upscale:
            payload["upscale"] = "x2"

    else:  # Realism XL
        payload = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "num_inference_steps": steps,
            "guidance_scale": scale,
            "width": width,
            "height": height,
            "scheduler": scheduler,
            "seed": seed
        }

    try:
        with st.spinner("Calling Replicate API…"):
            output = replicate_client.run(cfg["ref"], input=payload)

        # Helper to display any result type
        def show_image(item):
            if hasattr(item, "read"):
                st.image(item.read(), use_container_width=True)
            elif hasattr(item, "url"):
                st.image(item.url, use_container_width=True)
            elif isinstance(item, str) and item.startswith("http"):
                st.image(item, use_container_width=True)
            else:
                st.error("Unrecognized output format."); st.write(item)

        if isinstance(output, list):
            for img in output:
                show_image(img)
        else:
            show_image(output)

        st.success("Generation complete!")
    except replicate.exceptions.ReplicateException as err:
        st.error(f"Model error: {err}")
        st.info("Try changing scheduler, steps or resolution.")
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        st.info("Check your inputs and try again.")
