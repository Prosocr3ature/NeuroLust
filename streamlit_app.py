import os
import time
import base64
import requests
import streamlit as st
from io import BytesIO
from PIL import Image
import replicate
from typing import Dict, Tuple, Union

# ==================== CONSTANTS & CONFIG ====================
DEFAULT_APPEARANCE = (
    "Princess Jasmine from Aladdin as a glamorous model with glistening, wet soft skin, "
    "voluptuous curves—huge round breasts, tiny waist, thick thighs, prominent ass—"
    "sheer blue fishnet stockings, no underwear, and elegant nipple piercings."
)

IMAGE_MODELS: Dict[str, Dict] = {
    "Unrestricted XL": {
        "id": "asiryan/unlimited-xl:1a98916be7897ab4d9fbc30d2b20d070c237674148b00d344cf03ff103eb7082",
        "steps": 40, "guidance": 9.0
    },
    "Hardcore Edition": {
        "id": "asiryan/realism-xl:ff26a1f71bc27f43de016f109135183e0e4902d7cdabbcbb177f4f8817112219"
    },
    "Porn Diffusion": {
        "id": "aisha-ai-official/illust3relustion:7ff25c52350d3ef76aba554a6ae0b327331411572aeb758670a1034da3f1fec8",
        "extra_input": {
            "steps": 20,
            "refiner": True,
            "upscale": "x2",
            "scheduler": "Euler a beta",
            "refiner_strength": 0.6,
            "prompt_conjunction": True
        }
    }
}

ACTIONS: Dict[str, str] = {
    "Doggystyle":    "INSTRUCTION: Show Jasmine bent over on hands and knees, penis entering her from behind in deep doggystyle.",
    "Missionary":    "INSTRUCTION: Show Jasmine lying on her back, legs spread, penis thrusting into her in missionary.",
    "Cowgirl":       "INSTRUCTION: Show Jasmine riding on top, bouncing slowly in a cowgirl position.",
    "Deep Throat":   "INSTRUCTION: Show Jasmine kneeling, head tilted back, taking the entire length deep into her throat.",
    "Face Fuck":     "INSTRUCTION: Show Jasmine gripping the base of the shaft with both hands, face fucking hard.",
    "Anal":          "INSTRUCTION: Show Jasmine on all fours, penis deep in her ass, cheeks spread wide."
}

NEGATIVE_PROMPT = (
    "deformed, mutated, disfigured, bad anatomy, lowres, blurry, cartoonish, "
    "extra limbs, watermark, text, oversaturated, unrealistic"
)

# ==================== ENGINE ====================
class NSFWEngine:
    def __init__(self):
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("REPLICATE_API_TOKEN not set in environment")
        self.client = replicate.Client(api_token=token)

    def generate(self, model_key: str, action_key: str, custom: str) -> Tuple[str, str]:
        cfg = IMAGE_MODELS[model_key]
        instruction = custom.strip() if custom.strip() else ACTIONS[action_key]
        prompt = (
            f"{instruction}\n"
            f"APPEARANCE: {DEFAULT_APPEARANCE}\n"
            "photorealistic, hyper-realistic lighting, sharp focus, intricate details, "
            "perfect anatomy, explicit nudity"
        )

        # Build input dict
        if "extra_input" in cfg:
            inp = {"prompt": prompt, **cfg["extra_input"]}
        elif "steps" in cfg and "guidance" in cfg:
            inp = {
                "prompt": prompt,
                "num_inference_steps": cfg["steps"],
                "guidance_scale": cfg["guidance"],
                "negative_prompt": NEGATIVE_PROMPT,
                "width": 768,
                "height": 1152,
                "safety_checker": False
            }
        else:
            # realism-xl only needs prompt
            inp = {"prompt": prompt}

        # Retry logic for 502
        last_error = None
        for attempt in range(3):
            try:
                output = self.client.run(cfg["id"], input=inp)
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                # if it's a 502, retry after a pause
                if "502" in msg and attempt < 2:
                    time.sleep(1)
                    continue
                else:
                    return "", f"⚠️ Generation error: {e}"
        else:
            return "", f"⚠️ Generation failed after 3 attempts: {last_error}"

        # Process output
        if isinstance(output, list) and output:
            item = output[0]
            # binary
            if hasattr(item, "read"):
                img_bytes = item.read()
            elif isinstance(item, (bytes, bytearray)):
                img_bytes = item
            else:
                resp = requests.get(item, timeout=20)
                resp.raise_for_status()
                img_bytes = resp.content
            return self._to_data_uri(img_bytes), ""
        return "", "⚠️ Unexpected output format"

    def _to_data_uri(self, img_bytes: Union[bytes, bytearray]) -> str:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((1024, 1536), Image.Resampling.LANCZOS)
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

    st.sidebar.markdown("### Example Actions")
    for k, v in ACTIONS.items():
        st.sidebar.write(f"**{k}**: {v.replace('INSTRUCTION: ','')}")

if __name__ == "__main__":
    main()
