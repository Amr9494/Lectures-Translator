import io
import base64
import json
import os
import re
import tempfile
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

VISION_MODEL = "gpt-5.6-luna"
IMAGE_MODEL = "gpt-image-2"


# ============================================================
# PDF / image helpers
# ============================================================

def extract_page_as_image(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def pil_to_data_url(image: Image.Image, max_side: int = 2200) -> str:
    img = image.copy()
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def nearest_divisible(n: int, divisor: int = 16) -> int:
    return max(divisor, round(n / divisor) * divisor)


def prepare_for_image_edit(image: Image.Image) -> Image.Image:
    """Resize only enough to satisfy GPT Image arbitrary-resolution constraints."""
    w, h = image.size
    nw = nearest_divisible(w)
    nh = nearest_divisible(h)
    if nw == w and nh == h:
        return image.copy()
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


# ============================================================
# Structured vision analysis
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
                    # Normalized coordinates, 0..1000:
                    # [x1, y1, x2, y2]
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

Analyze the supplied lecture slide image. The goal is to translate ONLY human-readable
natural-language text while preserving the original artwork, illustrations, diagrams,
photographs, equations, symbols, numbers, logos, and layout.

Source language: {source_lang}
Target language: {target_lang}

Return every distinct text region that should be translated.

IMPORTANT:
- Coordinates MUST be normalized integers from 0 to 1000:
  [x1,y1,x2,y2], where 0,0 is the top-left and 1000,1000 is the bottom-right.
- Make boxes tight around the visible text, not around surrounding artwork.
- Do not combine unrelated text regions.
- Do not include decorative marks that are not text.
- Do not translate equations, mathematical variables, units, numerical values,
  or programming/code unless they contain natural-language words.
- If text is already in the target language, set translation equal to original.
- Preserve proper names unless they are normally translated.
- Preserve terminology consistently.
- For text embedded in a diagram, translate the words but do not alter the diagram.
- For a speech bubble/sign, return only the text inside it.
- If a region is too uncertain to safely modify, set keep_original=true.
- Estimate the original text color as a hex RGB value.
- Confidence is 0 to 1.
- Never invent text that is not visible.

Glossary:
{glossary_text}

Return ONLY the requested structured data.
"""

    response = client.responses.create(
        model=VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "slide_translation_regions",
                "schema": TEXT_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=12000,
    )

    payload = json.loads(response.output_text)
    regions = []

    w, h = image.size

    for item in payload.get("regions", []):
        bx = item["bbox"]

        if len(bx) != 4:
            continue

        # Convert normalized 0..1000 coordinates to actual image pixels.
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
# Font handling / rendering
# ============================================================

def find_default_font(target_lang: str) -> str | None:
    lang = target_lang.lower()
    candidates = []

    if "arab" in lang:
        candidates += [
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansArabic-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    if any(x in lang for x in ["chinese", "japanese", "korean"]):
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttf",
        ]

    candidates += [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def shape_text(text: str, target_lang: str) -> str:
    if "arab" in target_lang.lower():
        if not ARABIC_SUPPORT:
            return text
        return get_display(arabic_reshaper.reshape(text))
    return text


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = normalize_hex_color(value)
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    words = text.split()

    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        candidate = word if not current else current + " " + word
        bb = draw.textbbox((0, 0), candidate, font=font)

        if bb[2] - bb[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def fit_font(
    text: str,
    bbox: Tuple[int, int, int, int],
    font_path: str,
    target_lang: str,
    min_size: int = 8,
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    x1, y1, x2, y2 = bbox
    box_w = max(8, x2 - x1)
    box_h = max(8, y2 - y1)

    shaped = shape_text(text, target_lang)

    # Use the original box height as the main starting point, then reduce
    # until both width and total line height fit.
    start = max(min_size, int(box_h * 0.82))

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    for size in range(start, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size=size)
        lines = wrap_text(draw, shaped, font, int(box_w * 0.96))

        line_height = max(1, int(size * 1.15))
        total_h = len(lines) * line_height

        if total_h <= box_h * 0.94:
            return font, lines

    font = ImageFont.truetype(font_path, size=min_size)
    return font, wrap_text(draw, shaped, font, int(box_w * 0.96))


# ============================================================
# Background removal / text rendering
# ============================================================

def make_inpaint_mask(
    image_size: Tuple[int, int],
    regions: List[Dict[str, Any]],
    padding: int = 2,
) -> np.ndarray:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        if region.get("keep_original"):
            continue

        if not region.get("translation", "").strip():
            continue

        x1, y1, x2, y2 = region["bbox"]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Slight expansion catches antialiased text edges without making a huge mask.
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def inpaint_text_regions(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    radius: int = 3,
) -> Image.Image:
    if not regions:
        return image.copy()

    arr = np.array(image.convert("RGB"))
    mask = make_inpaint_mask(image.size, regions, padding=2)

    restored = cv2.inpaint(
        arr,
        mask,
        radius,
        cv2.INPAINT_TELEA,
    )

    return Image.fromarray(restored)


def render_translated_regions(
    base_image: Image.Image,
    regions: List[Dict[str, Any]],
    font_path: str,
    target_lang: str,
) -> Image.Image:
    image = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    for region in regions:
        if region.get("keep_original"):
            continue

        text = region.get("translation", "").strip()

        if not text:
            continue

        x1, y1, x2, y2 = region["bbox"]

        if x2 <= x1 or y2 <= y1:
            continue

        try:
            font, lines = fit_font(
                text,
                (x1, y1, x2, y2),
                font_path,
                target_lang,
            )
        except Exception:
            continue

        color = hex_to_rgb(region.get("color", "#111111"))
        align = region.get("align", "left")

        line_height = max(1, int(font.size * 1.15))
        total_h = line_height * len(lines)

        y = y1 + max(
            0,
            ((y2 - y1) - total_h) // 2,
        )

        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            tw = bb[2] - bb[0]

            if align == "center":
                x = x1 + ((x2 - x1) - tw) // 2
            elif align == "right":
                x = x2 - tw
            else:
                x = x1

            draw.text(
                (x, y),
                line,
                font=font,
                fill=color,
            )

            y += line_height

    return image


# ============================================================
# Optional GPT Image fallback
# ============================================================

def ai_masked_edit(
    client: OpenAI,
    image: Image.Image,
    regions: List[Dict[str, Any]],
    target_lang: str,
) -> Image.Image:
    """
    Use GPT Image only for difficult pages.

    The mask limits the requested edit to detected text regions.
    This is deliberately OFF by default because it costs image-generation
    API usage and is less deterministic than local reconstruction.
    """
    editable = [
        r for r in regions
        if not r.get("keep_original") and r.get("translation", "").strip()
    ]

    if not editable:
        return image.copy()

    work = prepare_for_image_edit(image)
    w, h = work.size

    # GPT Image masks use transparent pixels to mark areas to edit.
    mask = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    mask_draw = ImageDraw.Draw(mask)

    sx = w / image.width
    sy = h / image.height

    for region in editable:
        x1, y1, x2, y2 = region["bbox"]

        mask_draw.rectangle(
            (
                int(x1 * sx),
                int(y1 * sy),
                int(x2 * sx),
                int(y2 * sy),
            ),
            fill=(0, 0, 0, 0),
        )

    image_buf = io.BytesIO()
    work.convert("RGB").save(image_buf, format="PNG")
    image_buf.seek(0)

    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_buf.seek(0)

    mapping = "\n".join(
        f'- Replace "{r["original"]}" with "{r["translation"]}".'
        for r in editable
    )

    prompt = f"""
Edit ONLY the masked text regions of this lecture slide.

Target language: {target_lang}

{mapping}

Rules:
- Do not change unmasked artwork.
- Do not redraw diagrams, photographs, people, borders, icons, equations,
  or backgrounds.
- Do not add decorations.
- Preserve the exact positions and approximate typography of the existing text.
- Replace the specified text with the specified translations.
- Make translated text correctly shaped and legible.
- Do not translate anything that is not in the supplied mapping.
"""

    response = client.images.edit(
        model=IMAGE_MODEL,
        image=("page.png", image_buf.getvalue(), "image/png"),
        mask=("mask.png", mask_buf.getvalue(), "image/png"),
        prompt=prompt[:32000],
        n=1,
        size=f"{w}x{h}",
        output_format="png",
        quality="medium",
    )

    item = response.data[0]

    if not getattr(item, "b64_json", None):
        raise RuntimeError("GPT Image returned no base64 image data.")

    result = Image.open(
        io.BytesIO(base64.b64decode(item.b64_json))
    ).convert("RGB")

    return result.resize(
        image.size,
        Image.Resampling.LANCZOS,
    )


# ============================================================
# PDF output
# ============================================================

def compile_images_to_pdf(images: List[Image.Image]) -> bytes:
    if not images:
        return b""

    buf = io.BytesIO()
    rgb_images = [image.convert("RGB") for image in images]

    rgb_images[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=rgb_images[1:],
        resolution=150.0,
    )

    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Lecture Translator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Lecture Translator")

st.caption(
    "Translate lecture pages while preserving the original artwork and layout. "
    "The app first detects text and then edits only those regions."
)

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help=(
            "Used for this Streamlit session. Do not put API keys in source code "
            "or commit them to GitHub."
        ),
    )

    source_lang = st.selectbox(
        "Source language",
        [
            "English",
            "Arabic",
            "Finnish",
            "German",
            "French",
            "Spanish",
            "Auto-detect",
        ],
    )

    target_lang = st.selectbox(
        "Target language",
        [
            "Arabic",
            "Finnish",
            "English",
            "German",
            "Spanish",
            "Swedish",
        ],
    )

    glossary = st.text_area(
        "Glossary (optional)",
        placeholder=(
            "magnetic flux density = magneettivuon tiheys\n"
            "Taqwa = تقوى"
        ),
        height=120,
    )

    confidence = st.slider(
        "Minimum OCR confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    use_ai_fallback = st.checkbox(
        "Use GPT Image fallback for difficult pages",
        value=False,
        help=(
            "Costs additional image-generation API usage. "
            "Keep this OFF while testing the deterministic pipeline."
        ),
    )

    font_upload = st.file_uploader(
        "Optional target-language font (.ttf/.otf)",
        type=["ttf", "otf"],
        help=(
            "Recommended for Arabic, Persian, Urdu, CJK, or any language "
            "where the system font may not contain all required glyphs."
        ),
    )

    if not ARABIC_SUPPORT:
        st.warning(
            "Arabic shaping packages are not installed. "
            "Add arabic-reshaper and python-bidi to requirements.txt "
            "if Arabic is a target language."
        )

uploaded = st.file_uploader(
    "Upload lecture PDF",
    type=["pdf"],
)

if uploaded is not None:
    pdf_bytes = uploaded.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    st.success(
        f"Loaded **{uploaded.name}** — {total_pages} pages."
    )

    col1, col2 = st.columns(2)

    with col1:
        start_page = st.number_input(
            "Start page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
        )

    with col2:
        end_page = st.number_input(
            "End page",
            min_value=int(start_page),
            max_value=total_pages,
            value=min(total_pages, int(start_page) + 2),
            step=1,
        )

    st.info(
        "Recommended: test 1–3 pages first. "
        "Only process the whole lecture after the preview looks correct."
    )

    if font_upload:
        suffix = Path(font_upload.name).suffix
        temp_font = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        )
        temp_font.write(font_upload.getvalue())
        temp_font.close()
        font_path = temp_font.name
    else:
        font_path = find_default_font(target_lang)

    if not font_path:
        st.error(
            "No usable font was found. Upload a .ttf/.otf font "
            "for the target language."
        )
        st.stop()

    st.caption(f"Rendering font: `{font_path}`")

    if st.button(
        "🚀 Analyze & Translate",
        type="primary",
    ):
        if not api_key.strip():
            st.error(
                "Enter your OpenAI API key first."
            )
            st.stop()

        client = OpenAI(
            api_key=api_key.strip()
        )

        pages = list(
            range(
                int(start_page) - 1,
                int(end_page),
            )
        )

        translated_images: List[Image.Image] = []
        all_regions: Dict[int, List[Dict[str, Any]]] = {}

        progress = st.progress(0.0)
        status = st.empty()

        for index, page_num in enumerate(pages):
            status.info(
                f"Analyzing page {page_num + 1} / {total_pages}"
            )

            original = extract_page_as_image(
                doc,
                page_num,
                zoom=2.0,
            )

            try:
                regions = analyze_and_translate_page(
                    client=client,
                    image=original,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary,
                    min_confidence=confidence,
                )

                all_regions[page_num] = regions

                if not regions:
                    translated = original.copy()
                else:
                    # Primary method: deterministic reconstruction.
                    cleaned = inpaint_text_regions(
                        original,
                        regions,
                    )

                    translated = render_translated_regions(
                        cleaned,
                        regions,
                        font_path,
                        target_lang,
                    )

                    # Optional generative fallback.
                    if use_ai_fallback:
                        try:
                            translated = ai_masked_edit(
                                client,
                                original,
                                regions,
                                target_lang,
                            )
                        except Exception as fallback_error:
                            st.warning(
                                f"GPT Image fallback failed on page "
                                f"{page_num + 1}. Keeping deterministic "
                                f"rendering. {fallback_error}"
                            )

                translated_images.append(
                    translated
                )

            except Exception as error:
                st.error(
                    f"Page {page_num + 1} failed: {error}"
                )

                # Never discard the page.
                translated_images.append(
                    original.copy()
                )

            progress.progress(
                (index + 1) / len(pages)
            )

        status.success(
            "🎉 Translation finished."
        )

        if translated_images:
            st.subheader(
                "Preview — first processed page"
            )

            preview_left, preview_right = st.columns(2)

            with preview_left:
                st.markdown("**Original**")
                st.image(
                    extract_page_as_image(
                        doc,
                        pages[0],
                        zoom=1.5,
                    ),
                    use_container_width=True,
                )

            with preview_right:
                st.markdown("**Translated**")
                st.image(
                    translated_images[0],
                    use_container_width=True,
                )

        with st.expander(
            "Detected text regions"
        ):
            for page_num, regions in all_regions.items():
                st.markdown(
                    f"### Page {page_num + 1}"
                )

                for region in regions:
                    st.write(
                        {
                            "original": region["original"],
                            "translation": region["translation"],
                            "bbox": region["bbox"],
                            "confidence": region["confidence"],
                            "keep_original": region["keep_original"],
                        }
                    )

        output_pdf = compile_images_to_pdf(
            translated_images
        )

        st.download_button(
            "📥 Download translated PDF",
            data=output_pdf,
            file_name=(
                f"translated_{Path(uploaded.name).stem}.pdf"
            ),
            mime="application/pdf",
        )
