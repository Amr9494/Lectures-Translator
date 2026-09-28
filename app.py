import io
import base64
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
from openai import OpenAI


# ==========================================
# 1. Helper Functions: 16:9 & Image Processing
# ==========================================
def extract_page_as_image(doc: fitz.Document, page_num: int, zoom: float = 2.0) -> Image.Image:
    """Renders a PDF page to a high-resolution PIL Image."""
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGBA")


def prepare_square_image(image: Image.Image, target_size: int = 1024) -> Tuple[bytes, Tuple[int, int, int, int]]:
    """
    DALL-E 2 strictly requires a square PNG under 4 MB.
    This letterboxes 16:9 slides onto a 1024x1024 transparent canvas so that
    no stretching or distortion occurs, and returns the crop coordinates.
    """
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize keeping aspect ratio
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGBA")

    # Paste onto a square transparent canvas centered
    square_canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    square_canvas.paste(resized, (offset_x, offset_y))

    byte_stream = io.BytesIO()
    square_canvas.save(byte_stream, format="PNG")
    byte_stream.seek(0)

    # Return the bytes and the crop box to remove the padding later
    crop_box = (offset_x, offset_y, offset_x + new_w, offset_y + new_h)
    return byte_stream.getvalue(), crop_box


import requests

def translate_page_with_image_model(
    client: OpenAI,
    image: Image.Image,
    source_lang: str,
    target_lang: str,
    glossary: str = ""
) -> Image.Image:
    """
    Pads 16:9 slide to square, translates via OpenAI Image Edit API,
    crops back to 16:9, and scales to original resolution.
    """
    # 1. Letterbox to square PNG for the API
    square_bytes, crop_box = prepare_square_image(image, target_size=1024)

    # Prompt crafted for image-edit models (kept compact to comply with DALL-E's character limit)
    prompt = f"""
For the slide artwork in the center of the image:
1. Detect all human-readable text.
2. Translate the text from {source_lang} to {target_lang}.
3. Keep the original illustration, background, speech bubbles, and layout completely intact.
4. Replace only the original text, blending seamlessly with the style and lighting.
5. Render clean, properly connected typography for {target_lang}.
"""
    if glossary.strip():
        prompt += f"\nGlossary: {glossary.strip()}"

    prompt = prompt[:990].strip()

    # 2. Call OpenAI DALL-E 2 Image Edit (without response_format)
    response = client.images.edit(
        model="dall-e-2",
        image=("page.png", square_bytes, "image/png"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )

    # 3. Retrieve the generated image from URL or b64 fallback
    image_item = response.data[0]
    if hasattr(image_item, "url") and image_item.url:
        img_response = requests.get(image_item.url, timeout=30)
        img_response.raise_for_status()
        result_square = Image.open(io.BytesIO(img_response.content)).convert("RGB")
    elif hasattr(image_item, "b64_json") and image_item.b64_json:
        decoded_bytes = base64.b64decode(image_item.b64_json)
        result_square = Image.open(io.BytesIO(decoded_bytes)).convert("RGB")
    else:
        raise ValueError("No valid image data or URL returned by OpenAI.")

    # 4. Crop out the letterbox padding to retrieve pure 16:9
    result_16_9 = result_square.crop(crop_box)

    # 5. Upscale back to the user's original dimensions
    return result_16_9.resize(image.size, Image.Resampling.LANCZOS)

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

    st.info("💡 **16:9 Slide Support**: Automatically pads slides to square, translates, and crops out borders.")

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
        end_page = st.number_input("End Page", min_value=start_page, max_value=total_pages, value=min(start_page, total_pages))

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