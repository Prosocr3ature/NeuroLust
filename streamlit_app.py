import streamlit as st
import replicate
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Authentication ────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()
replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

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

JASMINE_BASE = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a glamorous model "
    "with glistening, wet soft skin and hyper-realistic detail. She has voluptuous curves—huge "
    "round breasts with nipple piercings, a tiny waist, thick thighs, and a sculpted, big sexy ass—"
    "adorned in sheer blue fishnet stockings, no underwear, pussy showing. Cinematic studio "
    "lighting, sharp focus, intricate textures, explicit nudity."
)

NEGATIVE_PROMPT = (
    "ugly face, poorly drawn hands, blurry, lowres, extra limbs, cartoon, censored, watermark, "
    "jpeg artifacts, error"
)

POSE_PRESETS = {
    "None": "",
    "POV Blowjob": "on her knees giving deepthroat POV blowjob, eyes locked on viewer, wet mouth, messy",
    "Doggy Style": "on all fours, viewed from behind, huge ass up, submissive posture",
    "Cowgirl Ride": "straddling and riding, breasts bouncing, looking down",
    "Face Covered in Cum": "kneeling with cum on face, tongue out, cock in frame"
}

# ─── Sidebar UI ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Model", list(IMAGE_MODELS.keys()))
    config = IMAGE_MODELS[model_choice]

    pose_choice = st.selectbox("Pose Preset", list(POSE_PRESETS.keys()))
    pose_text = POSE_PRESETS[pose_choice]

    custom_pose = st.text_area(
        "Custom Pose/Action (overrides preset):",
        height=100,
        placeholder="e.g. licking, handjob POV, mounting from behind"
    )
    action = custom_pose.strip() if custom_pose else pose_text

    steps = st.slider("Sampling Steps", 20, 100, config["steps"])
    scale = st.slider("Guidance Scale", 5.0, 15.0, config["scale"])
    width = st.slider("Width (px)", 512, 1024, config["width"], step=64)
    height = st.slider("Height (px)", 512, 1536, config["height"], step=64)
    scheduler = st.selectbox(
        "Scheduler",
        config["schedulers"],
        index=config["schedulers"].index(config["scheduler"])
    )

    extra_negative = st.text_area(
        "Extra Negative Terms (optional):",
        value="",
        height=80  # must be >= 68px
    )
    negative_prompt = NEGATIVE_PROMPT + (", " + extra_negative.strip() if extra_negative.strip() else "")

    seed_random = st.checkbox("Random seed", value=True)
    seed = random.randint(1, 999999) if seed_random else st.number_input("Seed", value=1337)

# ─── Prompt Assembly ─────────────────────────────────────────────────────────
final_prompt = f"{JASMINE_BASE} She is {action}. This must be pornographic and shown clearly."

# ─── Generate & Animate ───────────────────────────────────────────────────────
if st.button("Generate"):
    st.info(f"Using model: {model_choice}")
    payload = {
        "prompt": final_prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": int(width // 8) * 8,
        "height": int(height // 8) * 8,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler
    }

    try:
        # Generate static image
        with st.spinner("Generating image..."):
            output = replicate.run(config["ref"], input=payload)

        image_url = output[0] if isinstance(output, list) else output
        st.image(image_url, use_container_width=True)

        # Animate with wavespeedai WAN
        st.success("Now animating...")
        anim_input = {
            "image": image_url,
            "prompt": "A woman breathing and moving gently, erotic motion.",
            "loop": True,
            "fps": 10
        }
        with st.spinner("Generating animation..."):
            anim_url = replicate.run("wavespeedai/wan-2.1-i2v-480p", input=anim_input)
            st.video(anim_url)

    except Exception as e:
        st.error(f"Generation failed: {e}")
