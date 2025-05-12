import os
import base64
import requests
import streamlit as st
from io import BytesIO
from PIL import Image
import replicate
from typing import Dict, Tuple

# ==================== CONSTANTS & CONFIG ====================
DEFAULT_APPEARANCE = (
    "Princess Jasmine from Aladdin as a glamorous model with glistening, wet soft skin, "
    "voluptuous curves—huge round breasts, tiny waist, thick thighs, prominent ass—"
    "sheer blue fishnet stockings, no underwear, and elegant nipple piercings."
)

# Models with balanced speed/quality settings
IMAGE_MODELS = {
    "Unrestricted XL": {
        "id": "asiryan/unlimited-xl:1a98916be7897ab4d9fbc30d2b20d070c237674148b00d344cf03ff103eb7082",
        "steps": 40, "guidance": 9.0
    },
    "Hardcore Edition": {
        "id": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
        "steps": 40, "guidance": 8.5
    },
    "Porn Diffusion": {
        "id": "delta-lock/ponynai3:ea38949bfddea2db315b598620110edfa76ddaf6313a18e6cbc6a98f496a34e9",
        "steps": 40, "guidance": 8.0
    }
}

ACTIONS: Dict[str, str] = {
    "Doggystyle":      "INSTRUCTION: Show Jasmine bent over on hands and knees, penis entering her from behind in deep doggystyle.",
    "Missionary":      "INSTRUCTION: Show Jasmine lying on her back, legs spread, penis thrusting into her in missionary.",
    "Cowgirl":         "INSTRUCTION: Show Jasmine riding on top, bouncing slowly in a cowgirl position.",
    "Deep Throat":     "INSTRUCTION: Show Jasmine kneeling, head tilted back, taking the entire length deep into her throat.",
    "Face Fuck":       "INSTRUCTION: Show Jasmine gripping the base of the shaft with both hands, face fucking hard.",
    "Anal":            "INSTRUCTION: Show Jasmine on all fours, penis deep in her ass, cheeks spread wide.",
}

NEGATIVE_PROMPT = (
    "deformed, mutated, disfigured, bad anatomy, lowres, blurry, cartoonish, extra limbs, "
    "watermark, text, oversaturated, unrealistic"
)

# ==================== ENGINE ====================
class NSFWEngine:
    def __init__(self):
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("REPLICATE_API_TOKEN not set in environment")
        self.client = replicate.Client(api_token=token)

    def generate(self, model_key: str, action_key: str, custom: str) -> Tuple[str,str]:
        cfg = IMAGE_MODELS[model_key]
        # pick action instruction (custom overrides)
        instruction = custom.strip() if custom.strip() else ACTIONS[action_key]
        # build final prompt
        prompt = (
            f"{instruction}\nAPPEARANCE: {DEFAULT_APPEARANCE}\n"
            "photorealistic, hyper-realistic lighting, sharp focus, intricate details, "
            "perfect anatomy, explicit nudity"
        )
        payload = {
            "prompt": prompt,
            "num_inference_steps": cfg["steps"],
            "guidance_scale": cfg["guidance"],
            "negative_prompt": NEGATIVE_PROMPT,
            "width": 768,
            "height": 1152,
            "safety_checker": False
        }
        try:
            out = self.client.run(cfg["id"], input=payload)
        except Exception as e:
            return "", f"Error: {e}"
        if isinstance(out, list) and out:
            return self._fetch_base64(out[0]), ""
        return "", "Unexpected output"

    def _fetch_base64(self, url: str) -> str:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((1024,1536), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=90)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()

# ==================== UI ====================
def main():
    st.set_page_config(page_title="NSFW Jasmine Generator", layout="wide", page_icon="🔥")
    st.sidebar.title("Controls")
    model = st.sidebar.selectbox("Model", list(IMAGE_MODELS.keys()))
    action = st.sidebar.selectbox("Action", list(ACTIONS.keys()))
    custom = st.sidebar.text_area("Custom Instruction (overrides action)", height=100)
    if st.sidebar.button("Generate"):
        with st.spinner("Generating…"):
            img, err = NSFWEngine().generate(model, action, custom)
        if err:
            st.error(err)
        else:
            st.image(img, use_container_width=True)

    st.sidebar.markdown("### Example Quick Actions")
    for k in ACTIONS:
        st.sidebar.write(f"• **{k}**: {ACTIONS[k].replace('INSTRUCTION: ','')}")

if __name__ == "__main__":
    main()
