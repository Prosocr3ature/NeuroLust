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
            err = str(e).lower()
            if any(code in err for code in ["503", "internal server error"]):
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
    "She has voluptuous curves—huge round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—" 
    "adorned in sheer blue fishnet stockings, no underwear, pussy showing. "
    "Cinematic studio lighting, sharp focus, intricate textures, explicit nudity."
)
NEGATIVE_PROMPT = (
    "ugly face, poorly drawn hands, blurry, lowres, extra limbs, cartoon, censored, watermark, jpeg artifacts, error"
)

# ─── Pose Presets ─────────────────────────────────────────────────────────────
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "POV deepthroat blowjob, mouth fully covering a large erect cock, eyes locked on viewer, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, deep penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl position riding a large cock, bouncing motion, breasts jiggling, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, hands on thighs, direct eye contact",
    "Cum Covered Face": "kneeling with cum dripping on face, tongue out, messy hair, lustful eyes"
}

# ─── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose model:", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    pose_choice = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[pose_choice]
    custom = st.text_area(
        "Custom Pose/Action (overrides preset):",
        height=120,
        placeholder="e.g. licking, handjob POV, mounting from behind"
    ).strip()
    action_text = custom if custom else preset_text

    # Generation parameters
    steps = st.slider("Sampling Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    width = st.slider("Width (px)", 512, 1024, config["width"], step=64)
    height = st.slider("Height (px)", 512, 1536, config["height"], step=64)
    scheduler = st.selectbox("Scheduler:", config["schedulers"], index=config["schedulers"].index(config["scheduler"]))

    neg_extra = st.text_area("Add extra negatives (optional):", value="", height=80).strip()
    negative_prompt = NEGATIVE_PROMPT + (", " + neg_extra if neg_extra else "")

    use_rand = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1337)

    # Animation settings
    st.header("Animation Settings")
    frames = st.slider("Animation Frames", 5, 100, 32)
    fps = st.slider("Animation FPS", 5, 24, 10)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
full_prompt = f"Perform explicitly: {action_text}, {JASMINE_BASE}" if action_text else JASMINE_BASE

# ─── Image Generation ─────────────────────────────────────────────────────────
def generate_image():
    payload = {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "width": (width // 8) * 8,
        "height": (height // 8) * 8,
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
        with st.spinner("Generating image..."):
            result = generate_image()
        items = result if isinstance(result, list) else [result]
        image_url = None
        temp_file = None
        for img in items:
            if hasattr(img, "url"):
                image_url = img.url
                st.image(image_url, use_container_width=True)
                break
            if hasattr(img, "read"):
                data = img.read()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(data)
                tmp.flush()
                temp_file = tmp.name
                st.image(data, use_container_width=True)
                break
        if not image_url and not temp_file:
            st.error("No valid image for animation.")
            st.stop()

        anim_source = image_url if image_url else open(temp_file, "rb")
        st.success("Image generated. Now animating...")
        anim_payload = {
            "image": anim_source,
            "prompt": full_prompt,
            "loop": True,
            "fps": fps,
            "num_frames": frames
        }
        with st.spinner("Generating animation..."):
            anim_out = run_with_retry("wavespeedai/wan-2.1-i2v-480p", anim_payload)
        if hasattr(anim_out, "url"):
            st.video(anim_out.url)
        elif hasattr(anim_out, "read"):
            st.video(anim_out.read())
        os.unlink(temp_file) if temp_file else None
    except Exception as e:
        st.error(f"Error: {e}")
