import streamlit as st
import replicate
import random
import time
import tempfile
import os

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Authentication ────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()
client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Utility: Retry wrapper for transient errors ────────────────────────────────
def run_with_retry(model_ref, payload, retries=3, backoff=2):
    for i in range(retries):
        try:
            return client.run(model_ref, input=payload)
        except replicate.exceptions.ReplicateError as e:
            err = str(e).lower()
            if i < retries - 1 and any(code in err for code in ["503", "internal server error"]):
                time.sleep(backoff * (i + 1))
                continue
            raise

# ─── Model & Pose Presets ─────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
    "Aisha Centerfold v9":  "aisha-ai-official/centerfold-v9:93d4b4d9a8c6f39384ed5ba439ee24c1769c8c8fe292db3e3800d2c627f03bb0"
}
POSE_PRESETS = {
    "None": "",
    "POV Blowjob":    "POV deepthroat blowjob, mouth fully around erect cock, eyes locked, saliva dripping, explicit oral",
    "Doggy Style":    "doggy style from behind, full-body shot, deep penetration with a thick realistic cock, arched back, wet skin, rhythmic thrusting",
    "Cowgirl Ride":   "cowgirl straddling and riding a large erect cock, steady bouncing, breasts jiggling, intense gaze",
    "Spread Legs":    "laying back with legs spread wide around a realistic erect cock, full pussy exposure, hands on thighs, direct eye contact",
    "Cum Covered Face":"cum dripping on face and cock tip, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Static Image Settings")
    model_name = st.selectbox("Choose Image Model:", list(IMAGE_MODELS.keys()))
    model_ref  = IMAGE_MODELS[model_name]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[preset]
    custom_text = st.text_area(
        "Custom Pose/Action (override):", height=100,
        placeholder="e.g. licking, handjob POV"
    ).strip()
    action = custom_text if custom_text else preset_text

    # Common settings
    steps     = st.slider("Sampling Steps", 20, 100, 90)
    scale     = st.slider("Guidance Scale", 1.0, 20.0, 13.0)
    width     = st.slider("Width (px)", 512, 1024, 1024, step=64)
    height    = st.slider("Height (px)", 512, 1536, 1536, step=64)
    scheduler = st.selectbox("Scheduler:", ["DPMSolverMultistep", "PNDM", "DDIM"], index=0)

    neg_extra = st.text_area("Extra Negative Terms (optional):", value="", height=80).strip()
    negative_prompt = (
        "bad anatomy, deformed, blurry, lowres, poorly drawn, cartoon, watermark, jpeg artifacts, unrealistic colors, extra limbs, glitch"
        + (", " + neg_extra if neg_extra else "")
    )

    use_rand = st.checkbox("Random seed", value=True)
    seed     = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1234)

    st.header("Animation Settings")
    fps      = st.slider("Animation FPS", 1, 30, 12)
    duration = st.slider("Animation Duration (seconds)", 1, 10, 5)

# ─── Prompt Templates ─────────────────────────────────────────────────────────
BASE_PROMPT = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge round breasts with nipple piercings, "
    "a tiny waist, thick thighs, and a sculpted, big sexy ass—adorned in sheer blue fishnet stockings, no underwear, pussy showing. "
    "Cinematic studio lighting, sharp focus, intricate textures, explicit nudity."
)

actions_static = POSE_PRESETS  # reuse same mappings for static images

# ─── Static Image Generation ──────────────────────────────────────────────────
def generate_image():
    # Aisha Centerfold uses steps-only schema
    if model_name == "Aisha Centerfold v9":
        return run_with_retry(model_ref, {"steps": steps})

    payload = {
        "prompt":          f"Perform explicitly: {action}. {BASE_PROMPT}" if action else BASE_PROMPT,
        "negative_prompt": negative_prompt,
        "width":           (width // 8) * 8,
        "height":          (height // 8) * 8,
        "guidance_scale":  scale,
        "seed":            seed,
        "num_inference_steps": steps,
        "scheduler":       scheduler
    }
    return run_with_retry(model_ref, payload)

# ─── Main Execution ──────────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Generating static image with {model_name}...")
    tmp_file   = None
    img_source = None
    try:
        # Static Image
        with st.spinner("Generating image..."):
            img_output = generate_image()
        if model_name == "Aisha Centerfold v9":
            for idx, item in enumerate(img_output):
                data = item.read()
                st.image(data, caption=f"Image {idx+1}", use_container_width=True)
            img_source = None
        else:
            items = img_output if isinstance(img_output, list) else [img_output]
            for item in items:
                if hasattr(item, "url"):
                    img_source = item.url
                    st.image(img_source, use_container_width=True)
                    break
                if hasattr(item, "read"):
                    data = item.read()
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp.write(data); tmp.flush(); tmp_file = tmp.name
                    st.image(data, use_container_width=True)
                    img_source = tmp_file
                    break
            if not img_source:
                st.error("Failed to generate image.")
                st.stop()

        # Animation
        st.success("Image done. Generating animation...")
        if model_name == "Aisha Centerfold v9":
            # Use same Aisha call for animation
            video_out = run_with_retry(model_ref, {"steps": steps})
            for idx, item in enumerate(video_out):
                data = item.read()
                st.video(data, format="mp4")
        else:
            actions_anim = actions_static
            anim_descr = actions_anim.get(preset, action)
            anim_prompt = f"Perform explicitly: {anim_descr}. {BASE_PROMPT}" if anim_descr else BASE_PROMPT
            video_input = {
                "image":    img_source,
                "prompt":   anim_prompt,
                "fps":      fps,
                "duration": duration
            }
            with st.spinner("Generating video..."):
                video_out = run_with_retry("wavespeedai/wan-2.1-i2v-480p", video_input)
            if hasattr(video_out, "url"):
                st.video(video_out.url)
            elif hasattr(video_out, "read"):
                st.video(video_out.read())
            elif isinstance(video_out, list):
                for v in video_out:
                    data = v.read()
                    st.video(data, format="mp4")
                    break
            else:
                st.error("Unrecognized video output.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if tmp_file and os.path.exists(tmp_file): os.unlink(tmp_file)
