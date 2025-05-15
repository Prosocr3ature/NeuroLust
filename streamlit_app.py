# pip install replicate
# Learn more: https://replicate.com/christophy/stable-video-diffusion/api/schema

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
client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Utility: Retry wrapper for 5xx errors ─────────────────────────────────────
def run_with_retry(model_ref, payload, retries=3, backoff=2):
    for i in range(retries):
        try:
            return client.run(model_ref, input=payload)
        except replicate.exceptions.ReplicateError as e:
            if i < retries-1 and any(code in str(e).lower() for code in ["503","internal server error"]):
                time.sleep(backoff * (i+1))
                continue
            raise

# ─── Models & Presets ─────────────────────────────────────────────────────────
IMAGE_MODELS = {
    "Realism XL (Uncensored)": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
    "Aisha Illust3 Relustion": "aisha-ai-official/illust3relustion:92a0c9a9cb1fd93ea0361d15e499dc879b35095077b2feed47315ccab4524036"
}
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "deepthroat blowjob, mouth fully around erect cock, eyes locked, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, deep penetration, arched back, wet skin, thrusting",
    "Cowgirl Ride": "cowgirl straddling, bouncing, breasts jiggling, intense gaze",
    "Spread Legs": "laying back legs spread wide, full pussy exposure, direct eye contact",
    "Cum Covered Face": "cum dripping on face, messy hair, tongue out, lustful eyes"
}

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Static Generation Settings")
    model = st.selectbox("Model:", list(IMAGE_MODELS.keys()))
    model_ref = IMAGE_MODELS[model]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    base_action = POSE_PRESETS[preset]
    custom_action = st.text_area("Custom pose/action (override):", height=100, placeholder="e.g. licking, handjob POV").strip()
    action = custom_action if custom_action else base_action

    steps = st.slider("Steps", 20, 100, 50)
    scale = st.slider("Guidance scale", 1.0, 20.0, 9.0)
    w = st.slider("Width", 512, 1024, 768, step=64)
    h = st.slider("Height", 512, 1536, 1152, step=64)
    sched = st.selectbox("Scheduler:", ["DPMSolverMultistep","PNDM","DDIM"], index=0)

    neg_extra = st.text_area("Extra negatives (optional):", value="", height=80).strip()
    negative = "ugly face, blurry, bad anatomy, cartoon, watermark" + (", "+neg_extra if neg_extra else "")

    use_rand = st.checkbox("Random seed", True)
    seed = random.randint(1,999999) if use_rand else st.number_input("Seed:", value=1234)

    st.header("Animation Settings")
    video_length = st.selectbox("Video length:", ["14_frames_with_svd","25_frames_with_svd_xt"], index=0)
    fps = st.slider("FPS", 5, 30, 10)
    sizing = st.selectbox("Sizing strategy:", ["maintain_aspect_ratio","crop_to_16_9","use_image_dimensions"], index=0)
    motion = st.slider("Motion intensity (1-255)", 1, 255, 127)
    noise = st.slider("Noise (cond_aug)", 0.0, 0.5, 0.02, step=0.01)
    decode = st.slider("Decoding T", 1, 20, 14)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
prompt_base = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with wet, glistening skin and hyper-realistic detail. Explicit nudity."
)
full_prompt = f"Perform explicitly: {action}. {prompt_base}" if action else prompt_base

# ─── Generate Static Image ───────────────────────────────────────────────────
def gen_image():
    return run_with_retry(
        model_ref,
        {
            "prompt": full_prompt,
            "negative_prompt": negative,
            "width": (w//8)*8,
            "height": (h//8)*8,
            "guidance_scale": scale,
            "seed": seed,
            "num_inference_steps": steps,
            "scheduler": sched
        }
    )

if st.button("Generate"):
    st.info(f"Generating with {model}...")
    try:
        # Static
        with st.spinner("Generating image..."):
            out = gen_image()
        items = out if isinstance(out, list) else [out]
        img_url = None
        tmp_path = None
        for item in items:
            if hasattr(item,"url"):
                img_url = item.url; st.image(img_url, use_container_width=True); break
            if hasattr(item,"read"):
                data = item.read()
                tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".png")
                tmp.write(data); tmp.flush(); tmp_path = tmp.name
                st.image(data, use_container_width=True); break
        if not img_url and not tmp_path:
            st.error("Image gen failure."); st.stop()

        # Animation
        st.success("Image done. Generating video...")
        anim_input = {
            "input_image": img_url if img_url else open(tmp_path,"rb"),
            "video_length": video_length,
            "frames_per_second": fps,
            "sizing_strategy": sizing,
            "motion_bucket_id": motion,
            "cond_aug": noise,
            "decoding_t": decode,
            "seed": seed
        }
        with st.spinner("Generating video..."):
            video_url = run_with_retry(
                "christophy/stable-video-diffusion:92a0c9a9cb1fd93ea0361d15e499dc879b35095077b2feed47315ccab4524036",
                anim_input
            )
        st.video(video_url)
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if 'tmp_path' in locals() and tmp_path:
            try: os.unlink(tmp_path)
            except: pass
