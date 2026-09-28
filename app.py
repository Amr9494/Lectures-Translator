import io
import base64
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
from openai import OpenAI


# ==========================================
# 1. Helper Functions: Image & PDF Processing
# ==========================================
def extract_page_as_image(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    """Renders a PDF page to a high-resolution PIL Image."""
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGBA")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Converts a PIL Image to raw PNG bytes for the OpenAI API."""
    byte_stream = io.BytesIO()
    # The image-edit endpoint requires RGBA/PNG format
    image.convert("RGBA").save(byte_stream, format="PNG")
    byte_stream.seek(0)
    return byte_stream.getvalue()


def translate_page_with_image_model(
    client: OpenAI,
    image: Image.Image,
    source_lang: str,
    target_lang: str,
    glossary: str = ""
) -> Image.Image:
    """
    Calls OpenAI's image-editing engine directly with the user's master prompt.
    This eliminates flat bounding box patches and font glyph failures (tofu boxes).
    """
    raw_png = image_to_png_bytes(image)

    # Master prompt matching the exact instructions
    prompt = f"""
For this page image:
1. Detect all human-readable text.
2. Translate the text from {source_lang} to {target_lang}.
3. Preserve the original page dimensions, resolution, exact layout, and positions of all elements.
4. Do not redraw, redesign, or reinterpret diagrams, illustrations, characters, photographs, or backgrounds.
5. Replace only the original textual content, matching the original typography, perspective, and styling as closely as possible.
6. Keep mathematical equations, symbols, units, and technical notation unchanged unless they contain natural-language text.
7. If translated text is longer, adjust font size or line wrapping rather than changing the layout.
8. Preserve page numbers and headings.
9. Render clear, properly connected, high-quality script for {target_lang}.
10. Check the final output for untranslated text and ensure non-text elements remain intact.
"""
    if glossary.strip():
        prompt += f"\nGlossary of required terminology:\n{glossary.strip()}"

    # Call OpenAI's image edit endpoint
    # gpt-image-1 / dall-e-2 edits input images according to the prompt
    response = client.images.edit(
        image=("page.png", raw_png, "image/png"),
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json"
    )

    image_b64 = response.data[0].b64_json
    decoded_bytes = base64.b64decode(image_b64)
    translated_img = Image.open(io.BytesIO(decoded_bytes)).convert("RGB")
    
    # Resize back to original slide aspect ratio and resolution
    return translated_img.resize(image.size, Image.Resampling.LANCZOS)


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
# 2. Streamlit User Interface
# ==========================================
st.set_page_config(
    page_title="Lecture & Slide Translator",
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
        help="Your key stays only in browser session memory and is never logged or stored."
    )

    st.info("💡 **Direct Image Translation**: Uses OpenAI's image model to preserve artwork, characters, and textures seamlessly.")

    st.divider()
    st.markdown("### 📖 Translation Rules")
    source_lang = st.selectbox("Source Language", ["English", "Arabic", "Finnish", "German", "French", "Spanish", "Auto-detect"], index=0)
    target_lang = st.selectbox("Target Language", ["Arabic", "Finnish", "English", "German", "Spanish", "Swedish"], index=0)

    with st.expander("Custom Glossary (Optional)"):
        glossary_input = st.text_area(
            "Format: term = translation",
            placeholder="Taqwa = تقوى\nmagnetic flux density = magneettivuon tiheys",
            height=120
        )

# Main Application Area
st.title("📚 Lecture & Slide Translator")
st.caption("Translate lecture slides and illustrated materials page-by-page while maintaining artwork, characters, and background layouts intact.")

uploaded_file = st.file_uploader("Upload Lecture Slide / Document PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    st.success(f"Loaded `{uploaded_file.name}` ({total_pages} total pages)")

    # Page range selector
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        start_page = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
    with col_range2:
        end_page = st.number_input("End Page", min_value=start_page, max_value=total_pages, value=min(start_page + 1, total_pages))

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
            status_text.markdown(f"**Translating page {page_num + 1} of {total_pages}...**")
            
            # 1. Extract original page
            orig_img = extract_page_as_image(doc, page_num, zoom=2.0)

            # 2. Translate directly using OpenAI's Image Model
            try:
                trans_img = translate_page_with_image_model(
                    client=openai_client,
                    image=orig_img,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=glossary_input
                )
                translated_images.append(trans_img)

            except Exception as e:
                st.warning(f"Error on page {page_num + 1}: {str(e)}. Keeping original page image.")
                translated_images.append(orig_img.convert("RGB"))

            progress_bar.progress((idx + 1) / total_to_process)

        status_text.success("🎉 Translation Complete!")

        # Visual Comparison Preview
        st.subheader("Slide Comparison Preview")
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown("**Original Page**")
            st.image(extract_page_as_image(doc, pages_to_process[0], zoom=1.5), use_container_width=True)
        with preview_col2:
            st.markdown("**Translated Result**")
            st.image(translated_images[0], use_container_width=True)

        # PDF Download Button
        pdf_out = compile_images_to_pdf(translated_images)
        st.download_button(
            label="📥 Download Translated PDF",
            data=pdf_out,
            file_name=f"translated_{uploaded_file.name}",
            mime="application/pdf"
        )