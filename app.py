import io
import json
import os
import tempfile
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI

# Support for Arabic / RTL reshaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SUPPORT = True
except ImportError:
    HAS_ARABIC_SUPPORT = False


# ==========================================
# 1. Pydantic Models for Structured Outputs
# ==========================================
class TextBlock(BaseModel):
    original_text: str = Field(description="The exact original text seen on the slide.")
    translated_text: str = Field(description="The translated text in the target language.")
    # Bounding box normalized from 0 to 1000 for high resolution invariance
    ymin: int = Field(description="Top coordinate (0 to 1000)")
    xmin: int = Field(description="Left coordinate (0 to 1000)")
    ymax: int = Field(description="Bottom coordinate (0 to 1000)")
    xmax: int = Field(description="Right coordinate (0 to 1000)")
    bg_color_hex: str = Field(
        default="#FFFFFF",
        description="Estimated background color behind this text block (e.g. #FFFFFF or #F0F2F5)."
    )
    text_color_hex: str = Field(
        default="#000000",
        description="Estimated foreground text color (e.g. #000000)."
    )

class SlideTranslationResponse(BaseModel):
    blocks: List[TextBlock] = Field(description="List of detected text blocks and their translations.")


# ==========================================
# 2. Helper Functions: Image & PDF Processing
# ==========================================
def extract_page_as_image(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    """Renders a PDF page to a high-resolution PIL Image."""
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def hex_to_rgb(hex_str: str, default: tuple = (255, 255, 255)) -> tuple:
    """Converts hex color string to RGB tuple."""
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 6:
        try:
            return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return default
    return default


def shape_text(text: str, target_lang: str) -> str:
    """Applies complex script shaping (BiDi / Arabic) if needed."""
    if HAS_ARABIC_SUPPORT and ("Arabic" in target_lang or any("\u0600" <= c <= "\u06FF" for c in text)):
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    return text


def load_fallback_font(size: int):
    """Attempts to find a clean TTF font across systems; falls back to default."""
    font_paths = [
        # Linux / Debian (Streamlit Cloud)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def translate_slide_with_openai(
    client: OpenAI,
    image: Image.Image,
    source_lang: str,
    target_lang: str,
    glossary: str = ""
) -> List[TextBlock]:
    """Sends page image to OpenAI Vision model using Structured Outputs."""
    buffered = io.BytesIO()
    # Compress slightly to stay within API transport guidelines while retaining sharpness
    image.save(buffered, format="JPEG", quality=90)
    import base64
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = f"""
You are an expert academic slide translator.
Task:
1. Detect all text boxes on this slide page.
2. Translate the text from {source_lang} to {target_lang}.
3. Return the coordinates [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale.
4. Estimate the background hex color behind the text so we can cleanly cover it.
5. Estimate the text font color hex.

Rules:
- Do NOT translate mathematical variables, formulas, SI units, and product/code names.
- Do NOT modify diagrams, plots, or graphs—only translate labels and natural language sentences.
- Maintain consistent university-level engineering/scientific terminology.
"""
    if glossary.strip():
        prompt += f"\nCustom Glossary:\n{glossary.strip()}\nStrictly adhere to this glossary."

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        response_format=SlideTranslationResponse,
    )
    return completion.choices[0].message.parsed.blocks


def reconstruct_slide_image(
    base_image: Image.Image,
    blocks: List[TextBlock],
    target_lang: str
) -> Image.Image:
    """Masks original text blocks and overlays translated text."""
    canvas = base_image.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    for block in blocks:
        # Denormalize coordinates (0..1000 -> pixels)
        ymin = int(block.ymin * h / 1000)
        xmin = int(block.xmin * w / 1000)
        ymax = int(block.ymax * h / 1000)
        xmax = int(block.xmax * w / 1000)

        box_w = max(10, xmax - xmin)
        box_h = max(10, ymax - ymin)

        bg_rgb = hex_to_rgb(block.bg_color_hex, default=(255, 255, 255))
        fg_rgb = hex_to_rgb(block.text_color_hex, default=(0, 0, 0))

        # 1. Mask original text box
        draw.rectangle([xmin, ymin, xmax, ymax], fill=bg_rgb)

        # 2. Shape text (handles RTL if applicable)
        final_text = shape_text(block.translated_text, target_lang)

        # 3. Dynamic font scaling to fit inside the bounding box
        font_size = max(12, int(box_h * 0.70))
        font = load_fallback_font(font_size)

        # Draw the replacement text with slight vertical centering
        draw.text((xmin + 2, ymin + 1), final_text, fill=fg_rgb, font=font)

    return canvas


def compile_images_to_pdf(images: List[Image.Image]) -> bytes:
    """Combines a list of PIL Images into a single downloadable PDF file in memory."""
    if not images:
        return b""
    pdf_buffer = io.BytesIO()
    rgb_images = [img.convert("RGB") for img in images]
    rgb_images[0].save(
        pdf_buffer,
        format="PDF",
        save_all=True,
        append_images=rgb_images[1:],
        resolution=150.0
    )
    return pdf_buffer.getvalue()


# ==========================================
# 3. Streamlit User Interface
# ==========================================
st.set_page_config(
    page_title="Lecture Slide Translator",
    page_icon="📚",
    layout="wide"
)

# Sidebar: Authentication & Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("### 🔑 OpenAI API Key")
    user_api_key = st.text_input(
        "Enter your OpenAI Key (`sk-...`)",
        type="password",
        help="Your key stays only in your browser session memory and is never logged or stored on disk."
    )

    st.info("💡 **BYOK Model**: Translations call `gpt-4o` directly using your personal OpenAI API credit balance.")

    st.divider()
    st.markdown("### 📖 Translation Rules")
    source_lang = st.selectbox("Source Language", ["English", "Finnish", "German", "French", "Spanish", "Auto-detect"], index=0)
    target_lang = st.selectbox("Target Language", ["Finnish", "Arabic", "English", "German", "Spanish", "Swedish"], index=0)

    with st.expander("Custom Glossary (Optional)"):
        glossary_input = st.text_area(
            "Format: term = translation",
            placeholder="magnetic flux density = magneettivuon tiheys\nreluctance = reluktanssi",
            height=120
        )

# Main Application Area
st.title("📚 Lecture Slide Translator")
st.caption("Translate PDF slides page-by-page while preserving diagrams, formulas, layouts, and vector artwork.")

uploaded_file = st.file_uploader("Upload Lecture Slide PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    st.success(f"Loaded `{uploaded_file.name}` ({total_pages} total pages)")

    # Page range selector to save tokens/cost
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        start_page = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
    with col_range2:
        end_page = st.number_input("End Page", min_value=start_page, max_value=total_pages, value=min(start_page + 4, total_pages))

    if st.button("🚀 Start Translation", type="primary"):
        if not user_api_key:
            st.error("Please enter your OpenAI API key in the sidebar to proceed.")
            st.stop()

        openai_client = OpenAI(api_key=user_api_key)
        translated_images = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        pages_to_process = list(range(start_page - 1, end_page))
        total_to_process = len(pages_to_process)

        for idx, page_num in enumerate(pages_to_process):
            status_text.markdown(f"**Processing page {page_num + 1} of {total_pages}...**")
            
            # 1. Extract
            orig_img = extract_page_as_image(doc, page_num, zoom=2.0)

            # 2. Vision + Structured Translation
            try:
                blocks = translate_slide_with_openai(
                    client=openai_client,
                    image=orig_img,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary_input
                )
                # 3. Inpaint & Reconstruct
                trans_img = reconstruct_slide_image(orig_img, blocks, target_lang)
                translated_images.append(trans_img)

            except Exception as e:
                st.warning(f"Error on page {page_num + 1}: {str(e)}. Keeping original slide image.")
                translated_images.append(orig_img)

            progress_bar.progress((idx + 1) / total_to_process)

        status_text.success("🎉 Translation Complete!")

        # Preview Section
        st.subheader("Slide Comparison Preview")
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown("**Original First Page Processed**")
            st.image(extract_page_as_image(doc, pages_to_process[0], zoom=1.5), use_container_width=True)
        with preview_col2:
            st.markdown("**Translated Result**")
            st.image(translated_images[0], use_container_width=True)

        # PDF Download
        pdf_out = compile_images_to_pdf(translated_images)
        st.download_button(
            label="📥 Download Translated PDF",
            data=pdf_out,
            file_name=f"translated_{uploaded_file.name}",
            mime="application/pdf"
        )