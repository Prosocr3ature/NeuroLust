# pip install replicate  # and any other deps
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

# ─── Model Config & Presets ────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
    "Aisha Illust3 Relustion": "aisha-ai-official/illust3relustion:92a0c9a9cb1fd93ea0361d15e499dc879b35095077b2feed47315ccab4524036"
}
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "deepthroat blowjob, mouth fully around erect cock, eyes locked on viewer, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, deep penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl straddling and riding, bouncing, breasts jiggling, intense gaze",
    "Spread Legs": "laying back with legs spread wide, full pussy exposure, direct eye contact",
    "Cum Covered Face": "cum dripping on face, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Image Generation Settings")
    model_name = st.selectbox("Model:", list(IMAGE_MODELS.keys()))
    model_ref = IMAGE_MODELS[model_name]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    preset_text = POSE_PRESETS[preset]
    custom_text = st.text_area("Custom Pose/Action (override):", height=100, placeholder="e.g. licking, handjob POV").strip()
    action = custom_text if custom_text else preset_text

    steps = st.slider("Steps", 20, 100, 50)
    scale = st.slider("Guidance Scale", 1.0, 20.0, 9.0)
    w = st.slider("Width (px)", 512, 1024, 768, step=64)
    h = st.slider("Height (px)", 512, 1536, 1152, step=64)
    scheduler = st.selectbox("Scheduler:", ["DPMSolverMultistep", "PNDM", "DDIM"], index=0)

    neg_extra = st.text_area("Extra negatives (optional):", value="", height=80).strip()
    negative_prompt = "ugly face, blurry, bad anatomy, cartoon, watermark" + (", " + neg_extra if neg_extra else "")

    use_rand = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1234)

    st.header("Animation Settings")
    frames_length = st.selectbox("Video Length:", ["14_frames_with_svd", "25_frames_with_svd_xt"], index=0)
    fps = st.slider("FPS", 5, 30, 10)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
prompt_base = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model with wet, glistening soft skin and explicit nudity."
)
full_prompt = f"Perform explicitly: {action}. {prompt_base}" if action else prompt_base

# ─── Static Image Generation ──────────────────────────────────────────────────
def generate_image():
    payload = {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "width": (w // 8) * 8,
        "height": (h // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler
    }
    return run_with_retry(model_ref, payload)

# ─── Main Execution ──────────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Generating with {model_name}...")
    try:
        # Static image
        with st.spinner("Generating image..."):
            img_output = generate_image()
        items = img_output if isinstance(img_output, list) else [img_output]
        img_url = None
        tmp_path = None
        for item in items:
            if hasattr(item, "url"):
                img_url = item.url
                st.image(img_url, use_container_width=True)
                break
            if hasattr(item, "read"):
                data = item.read()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(data)
                tmp.flush()
                tmp_path = tmp.name
                st.image(data, use_container_width=True)
                break

        if not img_url and not tmp_path:
            st.error("Image generation failed.")
            st.stop()

        # Animation via Stable Video Diffusion
        st.success("Image done. Generating video...")
        vid_payload = {
            "input_image": img_url if img_url else open(tmp_path, "rb"),
            "video_length": frames_length,
            "frames_per_second": fps,
            "seed": seed
        }
        with st.spinner("Generating video..."):
            vid_output = run_with_retry(
                "christophy/stable-video-diffusion:92a0c9a9cb1fd93ea0361d15e499dc879b35095077b2feed47315ccab4524036",
                vid_payload
            )

        # Display video
        if hasattr(vid_output, "url"):
            st.video(vid_output.url)
        elif hasattr(vid_output, "read"):
            st.video(vid_output.read())
        elif isinstance(vid_output, list):
            for v in vid_output:
                if hasattr(v, "url"):
                    st.video(v.url)
                    break
                if hasattr(v, "read"):
                    st.video(v.read())
                    break
        else:
            st.error("Unrecognized video output.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
