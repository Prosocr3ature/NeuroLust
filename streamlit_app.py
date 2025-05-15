import streamlit as st
import replicate
import random
import time
import tempfile
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Authentication ────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()
replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Utility: Retry wrapper for 5xx errors ─────────────────────────────────────
def run_with_retry(model_ref, payload, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            return replicate_client.run(model_ref, input=payload)
        except replicate.exceptions.ReplicateError as e:
            err_str = str(e).lower()
            if "503" in err_str or "internal server error" in err_str:
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
                    continue
            raise

# ─── Model Definitions ───────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0, "width": 768, "height": 1152,
        "schedulers": ["DPMSolverMultistep", "PNDM", "DDIM"],
        "scheduler": "DPMSolverMultistep"
    },
    "Aisha Illust3 Relustion": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40, "scale": 9.0, "width": 768, "height": 1152,
        "schedulers": ["PNDM", "DDIM", "DPMSolverMultistep"],
        "scheduler": "PNDM"
    }
}

# ─── Base Prompts ─────────────────────────────────────────────────────────────
JASMINE_BASE = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model with glistening, wet soft skin and hyper-realistic detail. "
    "She has voluptuous curves—huge round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—adorned in sheer blue fishnet stockings, no underwear, pussy showing. "
    "Cinematic studio lighting, sharp focus, intricate textures, explicit nudity."
)
NEGATIVE_PROMPT = (
    "ugly face, poorly drawn hands, blurry, lowres, extra limbs, cartoon, censored, watermark, jpeg artifacts, error"
)

# ─── Pose Presets ─────────────────────────────────────────────────────────────
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "POV deepthroat blowjob, eyes locked on viewer, mouth wrapped around large cock, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, full penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl position riding a large cock, bouncing motion, breasts jiggling, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, hands on thighs, direct eye contact",
    "Face Covered in Cum": "cum dripping on face, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose model:", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    pose_choice = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[pose_choice]
    custom_text = st.text_area(
        "Custom Pose/Action (overrides preset):",
        height=100,
        placeholder="e.g. handjob POV, licking, mounting from behind"
    ).strip()
    action_text = custom_text if custom_text else preset_text

    steps = st.slider("Sampling Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    width = st.slider("Width (px)", 512, 1024, config["width"], step=64)
    height = st.slider("Height (px)", 512, 1536, config["height"], step=64)
    scheduler = st.selectbox(
        "Scheduler:",
        config["schedulers"],
        index=config["schedulers"].index(config["scheduler"])
    )

    extra_neg = st.text_area("Add extra negatives (optional):", value="", height=80).strip()
    negative_prompt = NEGATIVE_PROMPT + (", " + extra_neg if extra_neg else "")

    seed_random = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if seed_random else st.number_input("Seed:", value=1337)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
full_prompt = f"{action_text}, {JASMINE_BASE}" if action_text else JASMINE_BASE

# ─── Generation Helper ───────────────────────────────────────────────────────
def generate_image():
    payload = {
        "prompt": full_prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": int(width // 8) * 8,
        "height": int(height // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler
    }
    return run_with_retry(config["ref"], payload)

# ─── Generate & Animate ───────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Using model: {model_choice}")
    try:
        # Static image generation
        with st.spinner("Generating image..."):
            result = generate_image()
        items = result if isinstance(result, list) else [result]
        image_url = None
        raw_bytes = None
        for img in items:
            if hasattr(img, "read"):
                raw_bytes = img.read()
                st.image(raw_bytes, use_container_width=True)
            elif hasattr(img, "url"):
                image_url = img.url
                st.image(image_url, use_container_width=True)
            elif isinstance(img, str) and img.startswith("http"):
                image_url = img
                st.image(image_url, use_container_width=True)

        if not image_url and not raw_bytes:
            st.error("Failed to retrieve image data for animation.")
            st.stop()

        # Prepare and run animation
        anim_source = image_url if image_url else raw_bytes
        anim_prompt = f"{action_text}, subtle realistic movement loop, breathing and slight motion"
        anim_payload = {
            "image": anim_source,
            "prompt": anim_prompt,
            "loop": True,
            "fps": 10
        }

        st.success("Image generated. Now animating...")
        with st.spinner("Generating animation..."):
            anim_result = run_with_retry("wavespeedai/wan-2.1-i2v-480p", anim_payload)

        # Display animation
        if hasattr(anim_result, "read"):
            st.video(anim_result.read())
        elif hasattr(anim_result, "url"):
            st.video(anim_result.url)
        elif isinstance(anim_result, str) and anim_result.startswith("http"):
            st.video(anim_result)
        elif isinstance(anim_result, list):
            for a in anim_result:
                if hasattr(a, "read"):
                    st.video(a.read())
                    break
                if hasattr(a, "url"):
                    st.video(a.url)
                    break
        else:
            st.error("Unrecognized animation output.")

    except Exception as e:
        st.error(f"Generation failed: {e}")

