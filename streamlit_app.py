import os
import base64
import requests
import streamlit as st
from io import BytesIO
from PIL import Image
import replicate
from typing import Dict, List, Tuple, Union

# ==================== CONFIGURATION ====================
DEFAULT_PROMPT = (
    "Ultra-photorealistic 8K portrait of Princess Jasmine from Aladdin as a "
    "glamorous model with glistening, wet soft skin and hyper-realistic detail. "
    "She has voluptuous curves—huge round breasts, a tiny waist, thick thighs, "
    "and a sculpted, prominent ass—adorned in sheer blue fishnet stockings, no "
    "underwear, and elegant nipple piercings."
)

IMAGE_MODELS: Dict[str, Dict] = {
    "Unrestricted XL": {
        "id": "asiryan/unlimited-xl:1a98916be7897ab4d9fbc30d2b20d070c237674148b00d344cf03ff103eb7082",
        "width": 768, "height": 1152, "guidance": 9.0, "steps": 40
    },
    "Hardcore Edition": {
        "id": "asiryan/reliberate-v3:d70438fcb9bb7adb8d6e59cf236f754be0b77625e984b8595d1af02cdf034b29",
        "width": 768, "height": 1152, "guidance": 8.5, "steps": 40
    },
    "Porn Diffusion": {
        "id": "delta-lock/ponynai3:ea38949bfddea2db315b598620110edfa76ddaf6313a18e6cbc6a98f496a34e9",
        "width": 768, "height": 1152, "guidance": 8.0, "steps": 35
    }
}

SEXUAL_POSITIONS = [
    "Doggystyle", "Missionary", "Cowgirl", "Reverse Cowgirl",
    "Standing", "Spooning", "Anal"
]

ORAL_ACTIONS = [
    "Deep Throat", "Face Fuck", "Cum in Mouth", "Cum on Face"
]

# ==================== ENGINE ====================
class NSFWCompanionEngine:
    def __init__(self):
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise EnvironmentError("REPLICATE_API_TOKEN not set")
        self.client = replicate.Client(api_token=token)

    def generate_image(
        self,
        model_name: str,
        prompt: str,
        positions: List[str],
        orals: List[str],
        custom: str,
        steps: int
    ) -> Tuple[str, str]:
        cfg = IMAGE_MODELS[model_name]

        # assemble prompt parts
        parts = [prompt]
        if positions:
            parts.append("Positions: " + ", ".join(positions))
        if orals:
            parts.append("Oral: " + ", ".join(orals))
        if custom.strip():
            parts.append(custom.strip())
        text_prompt = ". ".join(parts) + "."

        full_prompt = (
            "NSFW ALLOWED. FOLLOW USER PROMPT EXACTLY. "
            f"{text_prompt} "
            "photorealistic, hyper-realistic lighting, studio lighting, "
            "8K RAW, sharp focus, realistic skin texture, intricate details, "
            "perfect anatomy, explicit nudity"
        )

        negative = (
            "deformed, mutated, disfigured, bad anatomy, lowres, blurry, "
            "cartoonish, extra limbs, watermark, text, oversaturated, unrealistic"
        )

        payload = {
            "prompt": full_prompt,
            "width": cfg["width"],
            "height": cfg["height"],
            "num_inference_steps": steps,
            "guidance_scale": min(cfg["guidance"], 10.0),
            "negative_prompt": negative,
            "safety_checker": False
        }

        try:
            result = self.client.run(cfg["id"], input=payload)
        except Exception as e:
            return "", f"⚠️ Error generating image: {e}"

        if isinstance(result, list) and result:
            return self._encode(result[0]), ""
        return "", "⚠️ Unexpected output format"

    def _encode(self, url: str) -> str:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize((1024, 1536), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=100)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()

# ==================== UI ====================
class NSFWCompanionInterface:
    def __init__(self):
        self.engine = NSFWCompanionEngine()
        self._init_state()
        self._configure_page()

    def _init_state(self):
        st.session_state.setdefault("model", "Unrestricted XL")
        st.session_state.setdefault("positions", [])
        st.session_state.setdefault("orals", [])
        st.session_state.setdefault("custom", "")
        # default to each model's recommended steps
        st.session_state.setdefault(
            "steps",
            IMAGE_MODELS[st.session_state["model"]]["steps"]
        )
        st.session_state.setdefault("image", "")

    def _configure_page(self):
        st.set_page_config(page_title="NSFW Generator", page_icon="🔥", layout="wide")
        st.markdown("""
            <style>
                .main {background: #1a1a1a;}
                .sidebar .block-container {background: #2b2b2b;}
                .stButton>button {margin:4px 0; width:100%;}
                .stMultiselect>div, .stTextArea>div {background:#333; color:#fff;}
            </style>
        """, unsafe_allow_html=True)

    def _sidebar(self):
        st.sidebar.selectbox("Model", list(IMAGE_MODELS.keys()), key="model")
        # update steps if model changes
        st.session_state.steps = IMAGE_MODELS[st.session_state.model]["steps"]
        st.sidebar.slider(
            "Inference Steps",
            min_value=20, max_value=60, step=5, key="steps",
            help="Fewer = faster, lower quality"
        )
        st.sidebar.multiselect("Sexual Positions", SEXUAL_POSITIONS, key="positions")
        st.sidebar.multiselect("Oral Actions", ORAL_ACTIONS, key="orals")
        st.sidebar.text_area("Custom Actions / Scene", key="custom", height=120)
        if st.sidebar.button("Generate Image"):
            self._generate()

    def _generate(self):
        st.session_state.image = ""
        with st.spinner("Generating…"):
            img, err = self.engine.generate_image(
                model_name=st.session_state.model,
                prompt=DEFAULT_PROMPT,
                positions=st.session_state.positions,
                orals=st.session_state.orals,
                custom=st.session_state.custom,
                steps=st.session_state.steps
            )
        if err:
            st.error(err)
        else:
            st.session_state.image = img

    def _render(self):
        st.markdown("## Live Preview")
        if st.session_state.image:
            st.image(st.session_state.image, use_container_width=True)
        else:
            st.info("Configure options in the sidebar and click Generate.")

    def run(self):
        self._sidebar()
        self._render()

if __name__ == "__main__":
    NSFWCompanionInterface().run()
