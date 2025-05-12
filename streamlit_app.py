import streamlit as st
import replicate
import random

st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# Ensure API token is set
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error(
        "Replicate API token not found in Streamlit secrets. "
        "Please add it as REPLICATE_API_TOKEN."
    )
    st.stop()

replicate_client = replicate.Client(
    api_token=st.secrets["REPLICATE_API_TOKEN"]
)

VALID_SCHEDULERS = [
    "DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM",
    "K_EULER_ANCESTRAL", "K_EULER", "PNDM"
]

# ==================== MODELS ====================
MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0, "width": 768, "height": 1024,
        "scheduler": "DPMSolverMultistep",
        "preview": "https://replicate.delivery/pbxt/JqTfP3xup0D7quhKApwciUzEKCm36DyW7zHAcJ05ev8FuqaIA/out-0.png"
    },
    "ReLiberate v3 (Uncensored)": {
        "ref": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
        "steps": 40, "scale": 8.5, "width": 768, "height": 1024,
        "preview": "https://replicate.delivery/pbxt/GulWHD7mSgex0E0I1cT7ZkM5UPDJRyxyHbJpZMuNVBjIZumOA/out-0.png"
    },
    "Realistic Vision v5.1": {
        "ref": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "steps": 40, "scale": 7.5, "width": 768, "height": 1024,
        "preview": "https://replicate.delivery/pbxt/ZIGcmJ0FkZqCZUHU8G3v8g20PpLKczlA8uZB2cVVWz1EvDOQA/out-0.png"
    },
    "Photomaker (Uncensored)": {
        "ref": "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
        "steps": 30, "scale": 7.0, "width": 512, "height": 512,
        "preview": None
    },
    "Flux 1.1 Pro (Uncensored)": {
        "ref": "black-forest-labs/flux-1.1-pro",
        "extra_input": {
            "prompt_upsampling": True
        },
        "steps": 30, "scale": 7.5, "width": 512, "height": 512,
        "preview": None
    }
}

# ==================== UI OPTIONS ====================
POSES = [
    "standing nude", "bent over", "doggystyle",
    "missionary", "blowjob POV", "cowgirl riding",
    "kneeling spread legs"
]

NSFW_LEVELS = {
    "Vanilla":  "nude, soft lighting, erotic, sensual, highly detailed",
    "NSFW":     "nude, exposed genitals, wet skin, erotic gaze, realistic pussy, spread legs",
    "Hardcore": "hardcore porn, penetration, spread pussy, cum dripping, rough sex, open mouth, cock, blowjob"
}

PRESETS = {
    "Princess Jasmine Nude": {
        "prompt":  (
            "Princess Jasmine nude, seductive arabian princess, "
            "long black hair, large breasts, spread legs, wet, cum dripping, "
            "soft caramel skin, hyper-realistic, 8k, erotic pose"
        ),
        "negative": "cartoon, blurry, bad anatomy, unrealistic, lowres, watermark"
    },
    "Hardcore POV Sex": {
        "prompt":  (
            "beautiful nude woman giving blowjob POV, open mouth, "
            "tongue out, saliva dripping, deepthroat, ultra realistic, "
            "8k, wet skin, cinematic lighting"
        ),
        "negative": (
            "deformed, blurry, bad hands, cartoon, unrealistic, "
            "duplicate, cropped"
        )
    },
    "Full Body Glamour": {
        "prompt":  (
            "full body nude glamour model, large breasts, arched back, "
            "natural lighting, wet skin, realistic nipples, erotic expression, "
            "8k photo, highly detailed"
        ),
        "negative": "lowres, low quality, jpeg artifacts, poorly drawn face"
    }
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Model & Preset")
    model_choice = st.selectbox("Choose Model", list(MODELS.keys()))
    preview_url = MODELS[model_choice].get("preview")
    if preview_url:
        st.image(preview_url, caption=model_choice, use_container_width=True)

    preset_choice = st.selectbox("Choose Prompt Preset", list(PRESETS.keys()))
    nsfw_level = st.selectbox("NSFW Level", list(NSFW_LEVELS.keys()))
    pose = st.selectbox("Pose/Action", POSES)
    use_random_seed = st.checkbox("Use random seed", value=True)

    config = MODELS[model_choice]
    preset = PRESETS[preset_choice]

    prompt = st.text_area(
        "Prompt",
        value=f"{preset['prompt']}, {pose}, {NSFW_LEVELS[nsfw_level]}",
        height=150
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value=preset["negative"],
        height=80
    )

    # Sliders
    steps = st.slider("Steps", 20, 60, config.get("steps", 40))
    scale = st.slider("Guidance Scale", 1.0, 20.0, config.get("scale", 7.5))
    width = st.slider("Width", 256, 1024, config.get("width", 768))
    height = st.slider("Height", 256, 1024, config.get("height", 1024))

    # Scheduler (if available)
    scheduler = None
    if "scheduler" in config:
        scheduler = st.selectbox(
            "Scheduler",
            VALID_SCHEDULERS,
            index=VALID_SCHEDULERS.index(config["scheduler"])
        )

    seed = (
        random.randint(1, 999999)
        if use_random_seed
        else st.number_input("Seed", value=1337)
    )

# ==================== GENERATE BUTTON ====================
if st.button("Generate"):
    st.info(f"Using {model_choice}")
    # Base payload
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": int(width // 8) * 8,
        "height": int(height // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        # Use correct key for steps
        "num_inference_steps" if "ReLiberate" in model_choice else "steps": steps
    }
    if scheduler:
        payload["scheduler"] = scheduler

    # Merge any extra_input for models like Flux
    extra = config.get("extra_input", {})
    payload.update(extra)

    try:
        with st.spinner("Generating..."):
            output = replicate_client.run(config["ref"], input=payload)

        # Display first successful output
        if isinstance(output, list):
            for item in output:
                if hasattr(item, "read"):
                    st.image(item.read(), use_container_width=True)
                    break
                elif isinstance(item, str) and item.startswith("http"):
                    st.image(item, use_container_width=True)
                    break
        elif isinstance(output, str) and output.startswith("http"):
            st.image(output, use_container_width=True)
        else:
            st.warning("Output format not recognized.")
            st.write(output)

    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.info("Try lowering resolution, changing prompt, or selecting another model.")
