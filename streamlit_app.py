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

# ─── Updated Model Configuration ──────────────────────────────────────────────
MODELS = {
    "Realism XL (Uncensored)": {
        "ref": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219",
        "params": ["steps", "width", "height", "guidance_scale", "scheduler"]
    },
    "ReLiberate v3 (Uncensored)": {
        "ref": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
        "params": ["num_inference_steps", "guidance_scale", "width", "height"]
    },
    "Realistic Vision v6.0": {  # Updated version
        "ref": "lucataco/realistic-vision-v60:7e30a5b8d49c9a91a8ea4c2a370a5b1b5b0a7f7b0a9f9f7a8b7a8b7a8b7a8b7a",
        "params": ["steps", "width", "height", "guidance_scale"]
    }
}

# ─── Enhanced UI ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Choose image model:", list(MODELS.keys()))
    
    # Common parameters
    prompt = st.text_area("Prompt", height=150, placeholder="Enter your detailed prompt...", 
                         help="Use descriptive language, mention body features, lighting, and environment")
    negative_prompt = st.text_area("Negative Prompt", height=80, placeholder="deformed, blurry, bad anatomy...",
                                  value="deformed, blurry, bad anatomy, cartoonish, unrealistic")
    
    # Model-specific parameters
    with st.expander("Advanced Settings"):
        if "steps" in MODELS[model_choice]["params"]:
            steps = st.slider("Sampling Steps", 20, 50, 35)
        if "guidance_scale" in MODELS[model_choice]["params"]:
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.0)
        width = st.slider("Width", 512, 1024, 768) if "width" in MODELS[model_choice]["params"] else 768
        height = st.slider("Height", 512, 1024, 768) if "height" in MODELS[model_choice]["params"] else 768
        seed = st.number_input("Seed", value=13961, help="For reproducibility")
        
        # Scheduler options where available
        if "scheduler" in MODELS[model_choice]["params"]:
            scheduler = st.selectbox("Scheduler", ["DPMSolverMultistep", "K_EULER", "DPM++"], index=0)

# ─── Generation Logic ────────────────────────────────────────────────────────
if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    st.info(f"Using model: {model_choice}")
    model_config = MODELS[model_choice]
    
    # Build input payload dynamically based on model parameters
    input_payload = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "seed": seed,
        "width": width,
        "height": height
    }
    
    # Add model-specific parameters
    if "guidance_scale" in model_config["params"]:
        input_payload["guidance_scale"] = guidance_scale
    if "steps" in model_config["params"]:
        input_payload["steps"] = steps
    if "scheduler" in model_config["params"]:
        input_payload["scheduler"] = scheduler
    if "num_inference_steps" in model_config["params"]:
        input_payload["num_inference_steps"] = steps  # Map steps to model-specific param

    with st.spinner("Generating image..."):
        try:
            outputs = replicate_client.run(model_config["ref"], input=input_payload)
            
            # Handle different output formats
            if isinstance(outputs, list):
                for i, item in enumerate(outputs):
                    st.image(item, caption=f"Image {i+1}", use_column_width=True)
            else:  # Handle single output
                st.image(outputs, caption="Generated Image", use_column_width=True)
            
            st.success("Generation complete! Tip: Use specific descriptors like 'realistic skin texture' or 'detailed facial features' for better results.")
            
        except Exception as e:
            st.error(f"Image generation failed: {str(e)}")
            st.info("Common fixes: 1) Check NSFW content restrictions 2) Try different seed 3) Reduce steps/scale")

# ─── Prompt Tips Section ─────────────────────────────────────────────────────
st.sidebar.markdown("""
**Prompt Engineering Tips:**
- Use explicit details: "perfect facial symmetry" 
- Specify lighting: "soft cinematic lighting"
- Add textures: "smooth skin texture"
- Mention perspective: "full-body view from low angle"
- Include style: "hyper-realistic photography"
""")
