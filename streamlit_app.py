import streamlit as st import replicate import io from PIL import Image

─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="NeuroLust", layout="wide") st.title("🔥 NeuroLust: Uncensored Image Generator")

─── Load API Token ──────────────────────────────────────────────────────────

REPLICATE_TOKEN = st.secrets["replicate_api_token"] replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

─── Available Models ─────────────────────────────────────────────────────────

MODELS = { "Realistic Vision v5.1": { "id": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb", "params": lambda prompt, neg: {"prompt": prompt, "negative_prompt": neg}, }, "Illust3Relustion": { "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8", "params": lambda prompt, neg: { "prompt": prompt, "negative_prompt": neg, "steps": 20, "refiner": True, "upscale": "x2", "scheduler": "Euler a beta", "refiner_strength": 0.6, "prompt_conjunction": True, }, }, "Realism XL": { "id": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219", "params": lambda prompt, neg: { "prompt": prompt, "negative_prompt": neg, }, }, }

─── Preset Tags ─────────────────────────────────────────────────────────────

PRESETS = { "Custom": "", "Blowjob Side POV": "A nude woman giving a blowjob from side, cum on face, deepthroat, POV, realistic, 8K", "Facial Cumshot": "A woman receiving a facial, cum on tongue and lips, realistic detail, 8K", "Missionary POV": "Woman in missionary position having sex, visible pussy and penis, male POV, realistic 8K", "Doggystyle": "Woman on all fours from behind in doggystyle, anus and pussy visible, realistic POV, 8K", "Ahegao Face": "A woman making an exaggerated ahegao face, tongue out, eyes crossed, realistic 8K" }

─── UI ───────────────────────────────────────────────────────────────────────

preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys())) prompt = st.text_area("Prompt", PRESETS[preset], height=100) neg_prompt = st.text_area("Negative Prompt (optional)", value="blurry, low quality, extra limbs", height=60) model_name = st.selectbox("Choose Image Model:", list(MODELS.keys()))

if st.button("🚀 Generate Image"): if not prompt.strip(): st.warning("Please enter a prompt.") else: with st.spinner("Generating image..."): try: model_info = MODELS[model_name] inputs = model_info["params"](prompt, neg_prompt) output = replicate_client.run(model_info["id"], input=inputs)

# Save and display each image
            for idx, image_file in enumerate(output):
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
                st.image(image, caption=f"Result {idx+1}", use_container_width=True)
        except Exception as e:
            st.error(f"Image generation failed: {e}")

