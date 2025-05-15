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
            if any(code in str(e).lower() for code in ["503", "internal server error"]):
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
    "She has voluptuous curves—huge round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass —adorned in sheer blue fishnet stockings, no underwear, pussy showing. "
    "Cinematic studio lighting, sharp focus, intricate textures, explicit nudity."
)
NEGATIVE_PROMPT = (
    "ugly face, poorly drawn hands, blurry, lowres, extra limbs, cartoon, censored, watermark, jpeg artifacts, error"
)

# ─── Pose Presets ─────────────────────────────────────────────────────────────
POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "deepthroat blowjob in POV, mouth fully covering a large erect cock, eyes locked on viewer, saliva dripping, explicit oral",
    "Doggy Style": "doggystyle from behind, deep penetration, arched back, wet skin, thrusting"
}

# ─── Sidebar UI ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose model:", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    preset = st.selectbox("Pose Preset:", list(POSE_PRESETS.keys()))
    custom = st.text_area("Custom Pose/Action (overrides):", height=100,
                         placeholder="e.g. licking, handjob, mounting from behind").strip()
    action = custom if custom else POSE_PRESETS[preset]

    steps = st.slider("Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    w = st.slider("Width", 512, 1024, config["width"], step=64)
    h = st.slider("Height", 512, 1536, config["height"], step=64)
    scheduler = st.selectbox("Scheduler:", config["schedulers"],
                             index=config["schedulers"].index(config["scheduler"]))

    extra_neg = st.text_area("Extra Negative Terms (optional):", value="", height=80).strip()
    neg = NEGATIVE_PROMPT + (", " + extra_neg if extra_neg else "")

    use_rand = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if use_rand else st.number_input("Seed:", value=1337)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
if action:
    full_prompt = f"Perform explicitly: {action}. {JASMINE_BASE}"
else:
    full_prompt = JASMINE_BASE

# ─── Generation ───────────────────────────────────────────────────────────────
def generate_image():
    payload = {
        "prompt": full_prompt,
        "negative_prompt": neg,
        "width": (w // 8) * 8,
        "height": (h // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler
    }
    return run_with_retry(config["ref"], payload)

if st.button("Generate"):
    st.info(f"Using {model_choice}")
    try:
        # Static image
        with st.spinner("Generating image..."):
            out = generate_image()
        items = out if isinstance(out, list) else [out]
        img_url = None
        temp_file = None
        for i in items:
            if hasattr(i, "url"):
                img_url = i.url
                st.image(img_url, use_container_width=True)
                break
            if hasattr(i, "read"):
                data = i.read()
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp.write(data)
                temp.flush()
                temp_file = temp.name
                st.image(data, use_container_width=True)
                img_url = None
                break

        # Animation input selection
        if img_url:
            anim_img = img_url
        elif temp_file:
            anim_img = open(temp_file, "rb")
        else:
            st.error("No valid image for animation.")
            st.stop()

        # Animate
        st.success("Animating...")
        anim_payload = {
            "image": anim_img,
            "prompt": full_prompt,
            "loop": True,
            "fps": 10
        }
        with st.spinner("Generating animation..."):
            anim_out = run_with_retry("wavespeedai/wan-2.1-i2v-480p", anim_payload)

        # Display video
        if hasattr(anim_out, "url"):
            st.video(anim_out.url)
        elif hasattr(anim_out, "read"):
            st.video(anim_out.read())

        # Cleanup
        if temp_file:
            os.unlink(temp_file)
    except Exception as e:
        st.error(f"Error: {e}")
