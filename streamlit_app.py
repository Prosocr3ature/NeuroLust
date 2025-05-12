import streamlit as st
import replicate
import random

# ─── App Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── Authentication ────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Please add your Replicate API token to Streamlit secrets as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Model Definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Realism XL (Uncensored)": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
    "ReLiberate v3 (Uncensored)": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
    "Realistic Vision v5.1": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
    "Pony Realism (NSFW)": "aisha-ai-official/realism-pony-sy-v4:942f52c5b1f04384a988fd7df8425eaf77eebf155f09d5735463a724e36b826c",
}

VALID_SCHEDULERS = [
    "DDIM", "PNDM", "DPMSolverMultistep", "HeunDiscrete",
    "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER",
]

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Generation Settings")

    model_choice = st.selectbox("Choose model", list(MODELS.keys()))
    prompt = st.text_area("Prompt", value="Enter detailed prompt here...", height=120)
    negative_prompt = st.text_area("Negative prompt", value="deformed, blurry, bad anatomy, lowres", height=80)

    steps = st.slider("Inference steps", min_value=10, max_value=100, value=50)
    if steps < 30:
        st.warning("Low step count may lead to poor quality. Consider >=30.")
    scale = st.slider("Guidance scale", min_value=1.0, max_value=20.0, value=9.0)
    if scale < 5.0:
        st.info("Low guidance scale may yield creative but unsteady results.")

    width = st.selectbox("Width (px)", [512, 768, 1024], index=2)
    height = st.selectbox("Height (px)", [512, 768, 1024], index=2)
    width = (width // 8) * 8
    height = (height // 8) * 8

    scheduler = st.selectbox("Scheduler", VALID_SCHEDULERS, index=VALID_SCHEDULERS.index("PNDM"))

    use_random_seed = st.checkbox("Use random seed", value=True)
    seed = random.randint(1, 999_999) if use_random_seed else st.number_input("Seed", value=1337)

# ─── Generation Handler ────────────────────────────────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.error("Prompt cannot be empty.")
        st.stop()

    st.info(f"Generating with {model_choice}...")

    payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": width,
        "height": height,
        "guidance_scale": scale,
        "seed": seed,
        "num_inference_steps": steps,
        "scheduler": scheduler,
    }

    try:
        with st.spinner("Calling Replicate API…"):
            output = replicate_client.run(MODELS[model_choice], input=payload)

        def show_image(item):
            if hasattr(item, "read"):
                st.image(item.read(), use_column_width=True)
            elif hasattr(item, "url"):
                st.image(item.url, use_column_width=True)
            elif isinstance(item, str) and item.startswith("http"):
                st.image(item, use_column_width=True)
            else:
                st.error("Unrecognized output format.")
                st.write(item)

        if isinstance(output, list):
            for img in output:
                show_image(img)
        else:
            show_image(output)

        st.success("Generation complete!")

    except replicate.exceptions.ReplicateException as err:
        st.error(f"Model error: {err}")
        st.info("Try changing scheduler or reducing resolution.")
    except Exception as ex:
        st.error(f"Unexpected error: {ex}")
        st.info("Check your inputs and try again.")
