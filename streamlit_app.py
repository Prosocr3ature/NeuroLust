import streamlit as st import replicate

─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="NeuroLust", layout="wide") st.title("🔥 NeuroLust: Uncensored AI Image Generator")

─── Load Secret ─────────────────────────────────────────────────────────────

REPLICATE_TOKEN = st.secrets["REPLICATE_API_TOKEN"] replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

─── Model Selection ─────────────────────────────────────────────────────────

MODEL_OPTIONS = { "Realistic Vision v5.1 (lucataco)": { "ref": "lucataco/realistic-vision-v5.1:2c8e954decbf70b7607a4414e5785ef9e4de4b8c51d50fb8b8b349160e0ef6bb", "args": lambda prompt: {"prompt": prompt}, }, "Illust3Relustion (aisha-ai)": { "ref": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8", "args": lambda prompt: { "steps": 20, "prompt": prompt, "refiner": True, "upscale": "x2", "scheduler": "Euler a beta", "refiner_strength": 0.6, "prompt_conjunction": True, }, }, }

─── Preset Prompts ──────────────────────────────────────────────────────────

PRESETS = { "Custom": {"example": ""}, "Blowjob / Deepthroat from side": { "example": ( "A nude woman with eyes closed while giving a blowjob to a man from the side, long copper hair in ornate braids, wearing earrings and a choker, realistic outdoors raw photo 8k, male pubic hair, testicles, deepthroat, penis, tongue out, fellatio, close-up" ) }, "Facial": { "example": ( "A woman with long hair, looking at the viewer with brown eyes and wearing jewelry; she shows an open mouth and tongue as a nude man’s penis provides cum—cum in mouth and cum on tongue—realistic raw photo" ) }, "Ahegao face": { "example": ( "A woman making an exaggerated ahegao expression—with her tongue playfully sticking out and her eyes crossed—realistic 8k raw photo" ) }, "Innie pussy": {"example": "A close-up shot of a nude pussy, realistic 8k photo"}, "Missionary position": { "example": ( "Woman lying on her back in a missionary pose having vaginal sex, light streaming through a window, raw photo 8k, from male POV; auburn hair, large breasts, visible nipples, emerald striped thighhigh socks, spread legs reveal pussy and penis, hetero, solo focus" ) }, "Doggystyle position": { "example": ( "Woman on all fours in doggystyle position with a man, realistic raw photo 8k; sandy blonde hair, big ass, pussy, anus, big veiny penis, male pubic hair, hetero, solo focus" ) }, "Cumshot": { "example": ( "Excessive amount of cum dripping off a penis and a woman’s face and tongue, realistic 8k photo" ) }, "Spreading pussy and ass from behind": { "example": ( "A kneeling woman from behind spreading her ass and pussy, realistic 8k raw photo" ) }, }

─── UI ───────────────────────────────────────────────────────────────────────

model_label = st.selectbox("Choose image model:", list(MODEL_OPTIONS.keys())) model_info = MODEL_OPTIONS[model_label]

preset = st.selectbox("Choose a scene preset:", list(PRESETS.keys())) prompt = st.text_area( "Prompt", PRESETS[preset]["example"] if preset != "Custom" else "", height=150 )

if st.button("🚀 Generate Image"): if not prompt.strip(): st.warning("Please enter or select a prompt.") else: with st.spinner("Generating NSFW image..."): try: result = replicate_client.run( model_info["ref"], input=model_info"args" ) st.subheader("🖼️ Generated Image") for index, img in enumerate(result): st.image(img, caption=f"Image {index + 1}", use_container_width=True) except Exception as e: st.error(f"Image generation failed: {e}")

