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

# ─── Utility: Retry wrapper for transient errors ─────────────────────────────────
def run_with_retry(model_ref, payload, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            return replicate_client.run(model_ref, input=payload)
        except replicate.exceptions.ReplicateError as e:
            err = str(e).lower()
            if any(term in err for term in ["503", "internal server error"]):
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
                    continue
            raise

# ─── Model Config ──────────────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "steps": 45, "scale": 8.0, "width": 768, "height": 1152,
        "schedulers": ["DPMSolverMultistep", "PNDM", "DDIM"], "scheduler": "DPMSolverMultistep"
    },
    "Aisha Illust3 Relustion": {
        "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "steps": 40, "scale": 9.0, "width": 768, "height": 1152,
        "schedulers": ["PNDM", "DDIM", "DPMSolverMultistep"], "scheduler": "PNDM"
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
    "POV Blowjob": "POV deepthroat blowjob, mouth fully covering a large erect cock, eyes locked on viewer, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, full penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl straddling and riding, breasts bouncing, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, direct eye contact",
    "Cum Covered Face": "cum dripping on face, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose model:", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[preset]
    custom = st.text_area("Custom Pose/Action (overrides preset):", height=120,
                         placeholder="e.g. licking, handjob POV, mounting from behind").strip()
    action_text = custom if custom else preset_text

    steps = st.slider("Inference Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    width = st.slider("Width (px)", 512, 1024, config["width"], step=64)
    height = st.slider("Height (px)", 512, 1536, config["height"], step=64)
    scheduler = st.selectbox("Scheduler:", config["schedulers"], index=config["schedulers"].index(config["scheduler"]))

    extra_neg = st.text_area("Extra Negative Terms (optional):", value="", height=80).strip()
    negative_prompt = NEGATIVE_PROMPT + (", " + extra_neg if extra_neg else "")

    st.header("Animation Settings")
    # Stable-Video-Diffusion params
    video_len = st.selectbox("Video Length:", ["14_frames_with_svd", "25_frames_with_svd_xt"], index=0)
    fps = st.slider("Animation FPS", 5, 30, 10)
    motion_bucket = st.slider("Motion Intensity (1-255)", 1, 255, 127)
    cond_aug = st.slider("Noise (cond_aug)", 0.0, 0.5, 0.02, step=0.01)
    decoding_t = st.slider("Decoding T", 1, 20, 7)

    use_rand = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1337)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
full_prompt = f"Perform explicitly: {action_text}. {JASMINE_BASE}" if action_text else JASMINE_BASE

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

# ─── Execute ──────────────────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Using model: {model_choice}")
    try:
        with st.spinner("Generating static image..."):
            res = generate_image()
        items = res if isinstance(res, list) else [res]
        img_url = None
        tmp_file = None
        for img in items:
            if hasattr(img, "url"):
                img_url = img.url; st.image(img_url, use_container_width=True); break
            if hasattr(img, "read"):
                data = img.read()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(data); tmp.flush(); tmp_file = tmp.name
                st.image(data, use_container_width=True); break
        if not img_url and not tmp_file:
            st.error("Image generation failed.")
            st.stop()

        # ── Video Generation via Stable Video Diffusion ──
        st.success("Static image done. Generating animation...")
        video_payload = {
            "input_image": img_url or open(tmp_file, "rb"),
            "video_length": video_len,
            "frames_per_second": fps,
            "motion_bucket_id": motion_bucket,
            "cond_aug": cond_aug,
            "decoding_t": decoding_t,
            "sizing_strategy": "maintain_aspect_ratio",
            "seed": seed
        }
        with st.spinner("Generating video (Stable Video Diffusion)..."):
            video_url = run_with_retry("christophy/stable-video-diffusion:43b6ee89", video_payload)
        st.video(video_url)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'tmp_file' in locals() and tmp_file:
            try: os.unlink(tmp_file)
            except: pass
