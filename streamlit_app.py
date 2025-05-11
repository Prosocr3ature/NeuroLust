import streamlit as st
import replicate
import io
from PIL import Image

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLust", layout="wide")
st.title("🔥 NeuroLust: Uncensored AI Image Generator")

# ─── API Token ───────────────────────────────────────────────────────────────
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Replicate API token not found in Streamlit secrets. Please add it as REPLICATE_API_TOKEN.")
    st.stop()

replicate_client = replicate.Client(api_token=st.secrets["REPLICATE_API_TOKEN"])

# ─── Updated Model Configuration with Parameter Mapping ───────────────────────
MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "params": {
            "steps": {"type": "slider", "label": "Sampling Steps", "min": 20, "max": 50, "default": 35},
            "width": {"type": "slider", "label": "Width", "min": 512, "max": 1024, "default": 768},
            "height": {"type": "slider", "label": "Height", "min": 512, "max": 1024, "default": 768},
            "guidance_scale": {"type": "slider", "label": "Guidance Scale", "min": 1.0, "max": 20.0, "default": 7.0},
            "scheduler": {"type": "select", "label": "Scheduler", "options": ["DPMSolverMultistep", "K_EULER", "DPM++"], "default": "DPMSolverMultistep"}
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
    "Realistic Vision v6.0": {
        "ref": "lucataco/realistic-vision-v60:7e30a5b8d49c9a91a8ea4c2a370a5b1b5b0a7f7b0a9f9f7a8b7a8b7a8b7a8b7a",
        "params": {
            "steps": {"type": "slider", "label": "Sampling Steps", "min": 20, "max": 50, "default": 35},
            "width": {"type": "slider", "label": "Width", "min": 512, "max": 1024, "default": 768},
            "height": {"type": "slider", "label": "Height", "min": 512, "max": 1024, "default": 768},
            "guidance_scale": {"type": "slider", "label": "Guidance Scale", "min": 1.0, "max": 20.0, "default": 7.0}
        }
    }
}

# ─── Enhanced UI with Dynamic Parameter Handling ──────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(MODELS.keys()))
    
    # Common parameters
    prompt = st.text_area("Prompt", height=150, placeholder="Enter your detailed prompt...", 
                         help="Use descriptive language, mention body features, lighting, and environment")
    negative_prompt = st.text_area("Negative Prompt", height=80, placeholder="deformed, blurry, bad anatomy...",
                                  value="deformed, blurry, bad anatomy, cartoonish, unrealistic")
    
    # Model-specific parameters
    model_params = {}
    with st.expander("Advanced Settings"):
        for param_name, config in MODELS[model_choice]["params"].items():
            if config["type"] == "slider":
                model_params[param_name] = st.slider(
                    label=config["label"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"]
                )
            elif config["type"] == "select":
                model_params[param_name] = st.selectbox(
                    label=config["label"],
                    options=config["options"],
                    index=config["options"].index(config["default"])
                )
        
        seed = st.number_input("Seed", value=13961, help="For reproducibility")

# ─── Generation Logic with Fixed Parameter Mapping ────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    st.info(f"Using model: {model_choice}")
    model_config = MODELS[model_choice]
    
    # Build input payload
    input_payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "seed": seed,
        **model_params
    }

    # Special parameter handling
    if "width" in input_payload and "height" in input_payload:
        input_payload["width"] = int(input_payload["width"])
        input_payload["height"] = int(input_payload["height"])

    with st.spinner("Generating image..."):
        try:
            outputs = replicate_client.run(model_config["ref"], input=input_payload)
            
            if isinstance(outputs, list):
                cols = st.columns(len(outputs))
                for i, (col, item) in enumerate(zip(cols, outputs)):
                    col.image(item, caption=f"Image {i+1}", use_column_width=True)
            else:
                st.image(outputs, caption="Generated Image", use_column_width=True)
            
            st.success("Generation complete! Tip: Use specific descriptors like 'realistic skin texture' or 'detailed facial features' for better results.")
            
        except Exception as e:
            st.error(f"Image generation failed: {str(e)}")
            st.info("Common fixes: 1) Check NSFW content restrictions 2) Try different seed 3) Reduce steps/scale")

# ─── Prompt Tips Section ──────────────────────────────────────────────────────
st.sidebar.markdown("""
**Prompt Engineering Tips:**
- Use explicit details: "perfect facial symmetry" 
- Specify lighting: "soft cinematic lighting"
- Add textures: "smooth skin texture"
- Mention perspective: "full-body view from low angle"
- Include style: "hyper-realistic photography"
""")
