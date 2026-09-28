import io
import base64
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from openai import OpenAI

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False


# ============================================================
# Configuration
# ============================================================

VISION_MODEL = "gpt-4o"  # Stable, widely supported model
IMAGE_MODEL = "dall-e-2"


# ============================================================
# PDF / Image Helpers
# ============================================================

def extract_page_as_image(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def pil_to_data_url(image: Image.Image, max_side: int = 2000) -> str:
    img = image.copy()
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
# Structured Vision Analysis
# ============================================================

TEXT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "original": {"type": "string"},
                    "translation": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "role": {
                        "type": "string",
                        "enum": [
                            "title", "heading", "body", "label", "caption",
                            "table", "diagram_text", "speech_bubble", "other"
                        ],
                    },
                    "align": {
                        "type": "string",
                        "enum": ["left", "center", "right"],
                    },
                    "color": {"type": "string"},
                    "confidence": {"type": "number"},
                    "keep_original": {"type": "boolean"},
                },
                "required": [
                    "original", "translation", "bbox", "role", "align",
                    "color", "confidence", "keep_original"
                ],
            },
        }
    },
    "required": ["regions"],
}


def normalize_hex_color(value: str, fallback: str = "#111111") -> str:
    if not isinstance(value, str):
        return fallback
    value = value.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value
    if re.fullmatch(r"#[0-9a-fA-F]{3}", value):
        return "#" + "".join(c * 2 for c in value[1:])
    return fallback


def analyze_and_translate_page(
    client: OpenAI,
    image: Image.Image,
    source_lang: str,
    target_lang: str,
    glossary: str,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    data_url = pil_to_data_url(image)
    glossary_text = glossary.strip() if glossary.strip() else "(none)"

    prompt = f"""
You are a precision document-layout OCR and translation engine.

Analyze the supplied lecture slide / illustration image. Translate ONLY human-readable
natural-language text while preserving diagrams, equations, characters, and artwork.

Source language: {source_lang}
Target language: {target_lang}

Rules:
- Coordinates MUST be normalized integers from 0 to 1000: [x1, y1, x2, y2].
- Make boxes tight around visible text regions.
- For speech bubbles, signs, or titles, provide the natural translation in {target_lang}.
- Estimate original text color as a hex RGB value.
- If target language is Arabic, provide correct, standard grammatical Arabic phrasing.

Glossary:
{glossary_text}
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "slide_translation_regions",
                "schema": TEXT_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.2,
    )

    payload = json.loads(response.choices[0].message.content)
    regions = []
    w, h = image.size

    for item in payload.get("regions", []):
        bx = item["bbox"]
        if len(bx) != 4:
            continue

        x1 = int(round(max(0, min(1000, bx[0])) * w / 1000))
        y1 = int(round(max(0, min(1000, bx[1])) * h / 1000))
        x2 = int(round(max(0, min(1000, bx[2])) * w / 1000))
        y2 = int(round(max(0, min(1000, bx[3])) * h / 1000))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        item["bbox"] = [x1, y1, x2, y2]
        item["color"] = normalize_hex_color(item.get("color", "#111111"))
        item["confidence"] = float(item.get("confidence", 0.0))

        regions.append(item)

    return [r for r in regions if r["confidence"] >= min_confidence]


# ============================================================
# Font & Text Shaping (Correct Arabic Pipeline)
# ============================================================

def ensure_arabic_font() -> str:
    """Ensures a dedicated Arabic TrueType font is available."""
    local_font = "/tmp/NotoSansArabic-Bold.ttf"
    if not os.path.exists(local_font):
        try:
            url = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Bold.ttf"
            urllib.request.urlretrieve(url, local_font)
        except Exception:
            return ""
    return local_font


def find_default_font(target_lang: str) -> str:
    lang = target_lang.lower()
    if "arab" in lang:
        font = ensure_arabic_font()
        if font and os.path.exists(font):
            return font

    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def shape_line(line: str, is_arabic: bool) -> str:
    """Shapes a single line of text right before measuring/drawing."""
    if is_arabic and ARABIC_SUPPORT:
        reshaped = arabic_reshaper.reshape(line)
        return get_display(reshaped)
    return line


def wrap_natural_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    is_arabic: bool,
) -> List[str]:
    """Wraps text in logical word order, testing line width with shaped text."""
    words = text.strip().split()
    if not words:
        return [""]

    lines = []
    current_words = []

    for word in words:
        candidate_words = current_words + [word]
        candidate_text = " ".join(candidate_words)
        shaped_candidate = shape_line(candidate_text, is_arabic)
        bb = draw.textbbox((0, 0), shaped_candidate, font=font)

        if bb[2] - bb[0] <= max_width:
            current_words.append(word)
        else:
            if current_words:
                lines.append(" ".join(current_words))
                current_words = [word]
            else:
                lines.append(word)
                current_words = []

    if current_words:
        lines.append(" ".join(current_words))

    # Return lines fully shaped and ready for Pillow
    return [shape_line(line, is_arabic) for line in lines]


def fit_font(
    text: str,
    bbox: Tuple[int, int, int, int],
    font_path: str,
    target_lang: str,
    min_size: int = 10,
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    x1, y1, x2, y2 = bbox
    box_w = max(10, x2 - x1)
    box_h = max(10, y2 - y1)

    is_arabic = "arab" in target_lang.lower() or any("\u0600" <= c <= "\u06FF" for c in text)
    start_size = max(min_size, int(box_h * 0.70))

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    for size in range(start_size, min_size - 1, -2):
        try:
            font = ImageFont.truetype(font_path, size=size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        lines = wrap_natural_text(draw, text, font, int(box_w * 0.95), is_arabic)
        line_height = max(1, int(size * 1.25))
        total_h = len(lines) * line_height

        if total_h <= box_h * 0.95:
            return font, lines

    try:
        font = ImageFont.truetype(font_path, size=min_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    return font, wrap_natural_text(draw, text, font, int(box_w * 0.95), is_arabic)


# ============================================================
# Inpainting & Clean Rendering
# ============================================================

def make_inpaint_mask(image_size: Tuple[int, int], regions: List[Dict[str, Any]], padding: int = 4) -> np.ndarray:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        if region.get("keep_original") or not region.get("translation", "").strip():
            continue

        x1, y1, x2, y2 = region["bbox"]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def inpaint_text_regions(image: Image.Image, regions: List[Dict[str, Any]], radius: int = 3) -> Image.Image:
    if not regions:
        return image.copy()

    arr = np.array(image.convert("RGB"))
    mask = make_inpaint_mask(image.size, regions, padding=3)
    restored = cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(restored)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = normalize_hex_color(value)
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def render_translated_regions(
    base_image: Image.Image,
    regions: List[Dict[str, Any]],
    font_path: str,
    target_lang: str,
) -> Image.Image:
    image = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    is_arabic = "arab" in target_lang.lower()

    for region in regions:
        if region.get("keep_original"):
            continue

        text = region.get("translation", "").strip()
        if not text:
            continue

        x1, y1, x2, y2 = region["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue

        font, lines = fit_font(text, (x1, y1, x2, y2), font_path, target_lang)
        color = hex_to_rgb(region.get("color", "#111111"))

        # Default speech bubbles or Arabic to center/right
        align = region.get("align", "center" if region.get("role") == "speech_bubble" else ("right" if is_arabic else "left"))

        line_height = max(1, int(getattr(font, "size", 16) * 1.25))
        total_h = line_height * len(lines)
        y = y1 + max(0, ((y2 - y1) - total_h) // 2)

        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            tw = bb[2] - bb[0]

            if align == "center":
                x = x1 + max(0, ((x2 - x1) - tw) // 2)
            elif align == "right":
                x = x2 - tw
            else:
                x = x1

            draw.text((x, y), line, font=font, fill=color)
            y += line_height

    return image


def compile_images_to_pdf(images: List[Image.Image]) -> bytes:
    if not images:
        return b""
    buf = io.BytesIO()
    rgb_images = [img.convert("RGB") for img in images]
    rgb_images[0].save(buf, format="PDF", save_all=True, append_images=rgb_images[1:], resolution=150.0)
    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Lecture Translator", page_icon="📚", layout="wide")
st.title("📚 Lecture & Illustration Translator")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key (`sk-...`)", type="password")
    source_lang = st.selectbox("Source Language", ["English", "Arabic", "Auto-detect"], index=0)
    target_lang = st.selectbox("Target Language", ["Arabic", "English", "Finnish"], index=0)
    glossary = st.text_area("Glossary (Optional)", placeholder="haram = حرام\ncheating = الغش", height=100)
    confidence = st.slider("Minimum OCR Confidence", 0.0, 1.0, 0.60, 0.05)

uploaded = st.file_uploader("Upload PDF or Single Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    is_pdf = uploaded.name.lower().endswith(".pdf")
    if is_pdf:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        total_pages = len(doc)
    else:
        doc = None
        total_pages = 1

    st.success(f"Loaded **{uploaded.name}** ({total_pages} page(s)).")

    font_path = find_default_font(target_lang)

    if st.button("🚀 Translate", type="primary"):
        if not api_key.strip():
            st.error("Please enter your OpenAI API key.")
            st.stop()

        client = OpenAI(api_key=api_key.strip())
        results = []
        progress = st.progress(0.0)

        for page_idx in range(total_pages):
            if is_pdf:
                orig = extract_page_as_image(doc, page_idx, zoom=2.0)
            else:
                orig = Image.open(uploaded).convert("RGB")

            try:
                regions = analyze_and_translate_page(
                    client, orig, source_lang, target_lang, glossary, confidence
                )
                if not regions:
                    results.append(orig)
                else:
                    inpainted = inpaint_text_regions(orig, regions)
                    translated = render_translated_regions(inpainted, regions, font_path, target_lang)
                    results.append(translated)
            except Exception as e:
                st.warning(f"Page {page_idx + 1} error: {e}")
                results.append(orig)

            progress.progress((page_idx + 1) / total_pages)

        st.success("Complete!")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original**")
            st.image(orig, use_container_width=True)
        with col_b:
            st.markdown("**Translated (Corrected Arabic)**")
            st.image(results[0], use_container_width=True)

        st.download_button(
            "📥 Download Result PDF",
            data=compile_images_to_pdf(results),
            file_name=f"translated_{Path(uploaded.name).stem}.pdf",
            mime="application/pdf",
        )