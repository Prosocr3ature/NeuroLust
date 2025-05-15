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
    "Aisha Illust3 Relustion": "aisha-ai-official/illust3relustion:92a0c9a9cb1fd93ea0361d15e499dc879b35095077b2feed47315ccab4524036"
}
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "POV deepthroat blowjob, mouth fully around erect cock, eyes locked, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, deep penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl straddling and riding, bouncing, breasts jiggling, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, direct eye contact",
    "Cum Covered Face": "cum dripping on face, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Static Image Settings")
    model_name = st.selectbox("Choose Image Model:", list(IMAGE_MODELS.keys()))
    model_ref = IMAGE_MODELS[model_name]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[preset]
    custom_text = st.text_area(
        "Custom Pose/Action (override):", height=100,
        placeholder="e.g. licking, handjob POV"
    ).strip()
    action = custom_text if custom_text else preset_text

    steps = st.slider("Sampling Steps", 20, 100, 90)
    scale = st.slider("Guidance Scale", 1.0, 20.0, 13.0)
    width = st.slider("Width (px)", 512, 1024, 1024, step=64)
    height = st.slider("Height (px)", 512, 1536, 1536, step=64)
    scheduler = st.selectbox("Scheduler:", ["DPMSolverMultistep", "PNDM", "DDIM"], index=0)

    neg_extra = st.text_area("Extra Negative Terms (optional):", value="", height=80).strip()
    negative_prompt = (
        "bad anatomy, deformed, blurry, lowres, poorly drawn, cartoon, watermark, jpeg artifacts, unrealistic colors, extra limbs, mutation, glitch, detached head, out of frame, text, logo"
        + (", " + neg_extra if neg_extra else "")
    )

    use_rand = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1234)

    st.header("Animation Settings")
    fps = st.slider("Animation FPS", 1, 30, 12)
    duration = st.slider("Animation Duration (seconds)", 1, 10, 5)

# ─── Prompt Templates ─────────────────────────────────────────────────────────
# Jasmine Base
BASE_PROMPT = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model with glistening, wet soft skin and hyper-realistic detail. "
    "She has voluptuous curves—huge round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—adorned in sheer blue fishnet stockings, no underwear, pussy showing. "
    "Cinematic studio lighting, sharp focus, intricate textures, explicit nudity."
)
# Static action-specific prompts
actions_static = {
    "POV Blowjob": "POV deepthroat blowjob, close-up of mouth and cock, eyes locked on viewer, saliva dripping, explicit oral",
    "Doggy Style": "doggy style from behind, full-body shot, deep penetration, arched back, wet skin, rhythmic thrusting",
    "Cowgirl Ride": "cowgirl position, straddling and riding, bouncing motion, breasts jiggling, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, hands on thighs, direct eye contact",
    "Cum Covered Face": "cum dripping on face, messy hair, tongue out, lustful eyes"
}
# Animation action-specific prompts
actions_anim = {
    "POV Blowjob": (
        "deepthroat blowjob, gentle suction and head bobbing, natural slow fellatio motion, "
        "she willingly takes the cock fully into her mouth then gracefully withdraws"
    ),
    "Doggy Style": (
        "doggystyle from behind, pelvic thrusts with realistic hip movement, muscle flex, wet skin sheen,"
        "slight body sway"
    ),
    "Cowgirl Ride": (
        "cowgirl straddling and riding, steady bouncing, natural up-and-down motion, breasts and thighs movement,"
        "subtle torso lean"
    ),
    "Spread Legs": (
        "laying back with legs spread wide, rhythmic thigh opening and closing, slight pelvic lifts, direct eye contact"
    ),
    "Cum Covered Face": (
        "cum dripping on face in slow motion, messy hair, tongue out, subtle facial expressions of pleasure"
    )
}

# Determine static prompt
if preset in actions_static and not custom_text:
    static_prompt = f"Perform explicitly: {actions_static[preset]}. {BASE_PROMPT}"
elif custom_text:
    static_prompt = f"Perform explicitly: {custom_text}. {BASE_PROMPT}"
else:
    static_prompt = BASE_PROMPT

# ─── Static Image Generation ──────────────────────────────────────────────────
def generate_image():
    # Build payload differently for Aisha vs Realism XL
    if model_name == "Aisha Illust3 Relustion":
        # Aisha Illust3 Relustion expects a different schema
        payload = {
            "model": "Illust3Relustion",
            "vae": "default",
            "prompt_conjunction": False,
            "prompt": static_prompt,
            "negative_prompt_conjunction": False,
            "negative_prompt": negative_prompt,
            "cfg_scale": scale,
            "guidance_rescale": 1.0,
            "pag_scale": 0.0,
            "clip_skip": 2,
            "width": width,
            "height": height,
            "upscale": "Original",
            "refiner": False,
            "prepend_preprompt": False,
            # uses its own defaults for refiner and scheduler
        }
    else:
        # Realism XL or other text-to-image models
        payload = {
            "prompt": static_prompt,
            "negative_prompt": negative_prompt,
            "width": (width // 8) * 8,
            "height": (height // 8) * 8,
            "guidance_scale": scale,
            "seed": seed,
            "num_inference_steps": steps,
            "scheduler": scheduler
        }
    return run_with_retry(model_ref, payload)

# ─── Main Execution ────────────────────────────────────────────────────────── ──────────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Generating static image with {model_name}...")
    tmp_file = None
    try:
        # Generate static image
        with st.spinner("Generating image..."):
            img_output = generate_image()
        items = img_output if isinstance(img_output, list) else [img_output]
        img_source = None
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

        # Determine animation prompt
        if preset in actions_anim and not custom_text:
            anim_prompt = f"Perform explicitly: {actions_anim[preset]}."
        elif custom_text:
            anim_prompt = f"Perform explicitly: {custom_text}."
        else:
            anim_prompt = static_prompt
        anim_prompt += f" {BASE_PROMPT}"

        # Generate video via WAN-2.1 I2V
        st.success("Image done. Generating animation...")
        video_input = {
            "image": img_source,
            "prompt": anim_prompt,
            "fps": fps,
            "duration": duration
        }
        with st.spinner("Generating video..."):
            video_out = run_with_retry("wavespeedai/wan-2.1-i2v-480p", video_input)

        # Display video
        if hasattr(video_out, "url"):
            st.video(video_out.url)
        elif hasattr(video_out, "read"):
            st.video(video_out.read())
        elif isinstance(video_out, list):
            for v in video_out:
                if hasattr(v, "url"): st.video(v.url); break
                if hasattr(v, "read"): st.video(v.read()); break
        else:
            st.error("Unrecognized video output.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if tmp_file and os.path.exists(tmp_file): os.unlink(tmp_file)
