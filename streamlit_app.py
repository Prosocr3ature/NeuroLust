import streamlit as st
import replicate
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token ───────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets. Please add it as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Valid Schedulers ─────────────────────────────────────────────────────────
VALID_SCHEDULERS = [
    "DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM",
    "K_EULER_ANCESTRAL", "K_EULER", "PNDM"
]

# ─── Model Config ────────────────────────────────────────────────────────────
MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "params": {
            "steps": {"type": "slider", "label": "Sampling Steps", "min": 20, "max": 50, "default": 35},
            "width": {"type": "slider", "label": "Width", "min": 512, "max": 1024, "default": 768},
            "height": {"type": "slider", "label": "Height", "min": 512, "max": 1024, "default": 768},
            "guidance_scale": {"type": "slider", "label": "Guidance Scale", "min": 1.0, "max": 20.0, "default": 7.0},
            "scheduler": {"type": "select", "label": "Scheduler", "options": VALID_SCHEDULERS, "default": "DPMSolverMultistep"}
        }
    },
    "ReLiberate v3 (Uncensored)": {
        "ref": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
        "params": {
            "num_inference_steps": {"type": "slider", "label": "Sampling Steps", "min": 20, "max": 50, "default": 30},
            "guidance_scale": {"type": "slider", "label": "Guidance Scale", "min": 1.0, "max": 20.0, "default": 7.0},
            "width": {"type": "slider", "label": "Width", "min": 512, "max": 1024, "default": 768},
            "height": {"type": "slider", "label": "Height", "min": 512, "max": 1024, "default": 768}
        }
    },
    "Realistic Vision v5.1": {
        "ref": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb",
        "params": {
            "steps": {"type": "slider", "label": "Sampling Steps", "min": 20, "max": 50, "default": 35},
            "width": {"type": "slider", "label": "Width", "min": 512, "max": 1024, "default": 768},
            "height": {"type": "slider", "label": "Height", "min": 512, "max": 1024, "default": 768},
            "guidance_scale": {"type": "slider", "label": "Guidance Scale", "min": 1.0, "max": 20.0, "default": 7.0}
        }
    }
}

# ─── Sidebar UI ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(MODELS.keys()))

    prompt = st.text_area("Prompt", height=150, placeholder="Enter your detailed prompt...")
    negative_prompt = st.text_area("Negative Prompt", height=80, value="deformed, blurry, bad anatomy, cartoonish, unrealistic")

    model_params = {}
    with st.expander("Advanced Settings"):
        for param_name, config in MODELS[model_choice]["params"].items():
            if config["type"] == "slider":
                val = st.slider(config["label"], config["min"], config["max"], config["default"])
                if param_name in ["width", "height"]:
                    val = int(val // 8) * 8
                model_params[param_name] = val
            elif config["type"] == "select":
                model_params[param_name] = st.selectbox(config["label"], config["options"], index=config["options"].index(config["default"]))

        use_random_seed = st.checkbox("Use random seed", value=True)
        seed = random.randint(1, 999999) if use_random_seed else st.number_input("Seed", value=13961)

# ─── Image Generation ─────────────────────────────────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    st.info(f"Using model: {model_choice}")
    model_config = MODELS[model_choice]

    input_payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "seed": seed,
        **model_params
    }

    with st.spinner("Generating image..."):
        try:
            outputs = replicate_client.run(model_config["ref"], input=input_payload)

            # Unified image URL extractor
            image_url = None
            if isinstance(outputs, list):
                for item in outputs:
                    if hasattr(item, "url"):
                        image_url = item.url
                        break
                    elif isinstance(item, str) and item.startswith("http"):
                        image_url = item
                        break
            elif isinstance(outputs, dict):
                for v in outputs.values():
                    if isinstance(v, str) and v.startswith("http"):
                        image_url = v
                        break
            elif isinstance(outputs, str) and outputs.startswith("http"):
                image_url = outputs

            # Final display
            if image_url:
                st.image(image_url, caption="Generated Image", use_container_width=True)
                st.success("Image generated successfully.")
            else:
                st.error("No valid image URL found in output.")
                st.write("DEBUG OUTPUT:", outputs)

        except Exception as e:
            st.error(f"Image generation failed: {str(e)}")
            st.info("Try different prompt, seed, model, or reduce steps.")

# ─── Prompt Tips ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
**Prompt Tips:**
- "8k photo, arched back, natural light, wet skin"
- "hyperrealistic close-up, parted lips, bedroom eyes"
- "standing pose, cinematic shadows, full-body"
""")
