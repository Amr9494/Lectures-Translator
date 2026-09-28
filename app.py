
import io
import base64
import time
from typing import Optional, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image
import streamlit as st
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

IMAGE_MODEL = "gpt-image-2"

# GPT Image 2 supports flexible resolutions. These two are
# convenient for lecture pages while keeping the original
# orientation.
LANDSCAPE_SIZE = "1536x1024"
PORTRAIT_SIZE = "1024x1536"

DEFAULT_DPI = 150


# ============================================================
# PDF / image helpers
# ============================================================

def render_pdf_page(doc: fitz.Document, page_number: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    """Render one PDF page to a PIL RGB image."""
    page = doc.load_page(page_number)
    zoom = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def choose_model_size(image: Image.Image) -> str:
    """Choose a GPT Image 2 output orientation based on the source page."""
    w, h = image.size

    if h > w * 1.05:
        return PORTRAIT_SIZE

    return LANDSCAPE_SIZE


def make_edit_canvas(
    image: Image.Image,
    canvas_size: Tuple[int, int],
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Fit the page into the requested model canvas while preserving aspect ratio.

    The returned crop_box tells us where the original page lives inside
    the model canvas so we can remove the padding after generation.
    """
    cw, ch = canvas_size
    iw, ih = image.size

    scale = min(cw / iw, ch / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))

    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)

    # Use a simple average-ish edge color for padding.
    tiny = image.resize((1, 1), Image.Resampling.BOX)
    background = tiny.getpixel((0, 0))

    canvas = Image.new("RGB", (cw, ch), background)

    x = (cw - nw) // 2
    y = (ch - nh) // 2

    canvas.paste(resized, (x, y))

    crop_box = (x, y, x + nw, y + nh)

    return canvas, crop_box


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def decode_b64_image(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def extract_image_from_event(event) -> Optional[Image.Image]:
    """
    Handle the different event/object shapes returned by OpenAI SDK versions.

    For image edit streaming, partial events contain base64 image data.
    """
    # SDK object attributes
    for attr in ("b64_json", "partial_image_b64"):
        value = getattr(event, attr, None)
        if value:
            try:
                return decode_b64_image(value)
            except Exception:
                pass

    # Some SDK versions expose b64_json through a nested object.
    data = getattr(event, "data", None)
    if data is not None:
        for attr in ("b64_json", "partial_image_b64"):
            value = getattr(data, attr, None)
            if value:
                try:
                    return decode_b64_image(value)
                except Exception:
                    pass

    # Dictionary-style fallback.
    if isinstance(event, dict):
        for key in ("b64_json", "partial_image_b64"):
            value = event.get(key)
            if value:
                try:
                    return decode_b64_image(value)
                except Exception:
                    pass

        data = event.get("data")
        if isinstance(data, dict):
            for key in ("b64_json", "partial_image_b64"):
                value = data.get(key)
                if value:
                    try:
                        return decode_b64_image(value)
                    except Exception:
                        pass

    return None


def extract_final_image(response) -> Optional[Image.Image]:
    """Extract the final image from a normal Image API response."""
    try:
        item = response.data[0]

        b64_data = getattr(item, "b64_json", None)
        if b64_data:
            return decode_b64_image(b64_data)

        url = getattr(item, "url", None)
        if url:
            return download_image(url)

    except Exception:
        pass

    return None


# ============================================================
# Prompt
# ============================================================

def build_translation_prompt(
    source_language: str,
    target_language: str,
    glossary: str,
) -> str:
    glossary_instruction = ""

    if glossary.strip():
        glossary_instruction = f"""
TERMINOLOGY GLOSSARY:
Use these translations exactly where applicable:
{glossary.strip()}
"""

    return f"""
EDIT THIS EXISTING IMAGE. DO NOT REDESIGN IT.

This is a lecture / educational slide or textbook page.

Translate ALL visible natural-language text from
{source_language}
to
{target_language}.

The primary objective is:
TRANSLATE THE TEXT WHILE PRESERVING THE ORIGINAL IMAGE.

IMPORTANT — PRESERVE THE ORIGINAL VISUAL DESIGN:
- Keep all illustrations exactly as visually similar as possible.
- Keep photographs, people, faces, objects and icons.
- Keep diagrams and charts.
- Keep arrows, lines, borders and shapes.
- Keep backgrounds, textures, colors, shadows and lighting.
- Keep the same page composition.
- Keep the same text locations and hierarchy.
- Keep the same overall typography style.
- Do not redesign the slide.
- Do not create a cleaner or more modern version.
- Do not replace the original illustration with a newly invented one.
- Do not add decorative elements.

TEXT:
- Translate rather than summarize.
- Translate every visible natural-language phrase that should be translated.
- Do not invent explanations.
- Do not add commentary.
- Do not omit visible text.
- Keep numbers, equations, units and mathematical symbols unchanged unless
  they are ordinary-language words.
- Preserve names and established technical terminology when appropriate.
- Keep labels in diagrams in their original locations.
- Keep text inside boxes, speech bubbles, signs and callouts in the same areas.
- Preserve approximate font size, weight, alignment and line structure.
- If the translated text is longer, intelligently wrap it or slightly reduce
  its font size so it remains inside the original text area.
- Do not leave the original-language text behind.

ARABIC / RTL:
If the target language is Arabic:
- Use real Arabic Unicode.
- Use correct right-to-left direction.
- Properly connect Arabic letters.
- Never produce missing-glyph squares or boxes.
- Never transliterate Arabic into Latin letters.
- Preserve punctuation appropriately for Arabic.
- Example:
  "NO! Cheating is haram" -> "لا! الغش حرام."
- "haram" in this context means "حرام", not an explanation of the word.

FINAL REQUIREMENT:
The output should look like the SAME ORIGINAL PAGE made by the SAME DESIGNER,
with the natural-language text translated into the target language.
Only change what is necessary to translate the text.

{glossary_instruction}
""".strip()


# ============================================================
# Image translation
# ============================================================

def translate_page_streaming(
    client: OpenAI,
    source_image: Image.Image,
    source_language: str,
    target_language: str,
    glossary: str,
    preview_placeholder,
    status_placeholder,
    timer_placeholder,
    quality: str,
    partial_images: int,
) -> Image.Image:
    """
    Translate one page using GPT Image 2 with streaming partial images.

    Partial images are displayed immediately in Streamlit.
    The final image is returned after the stream completes.
    """

    model_size = choose_model_size(source_image)

    if model_size == PORTRAIT_SIZE:
        canvas_size = (1024, 1536)
    else:
        canvas_size = (1536, 1024)

    canvas, crop_box = make_edit_canvas(source_image, canvas_size)

    prompt = build_translation_prompt(
        source_language,
        target_language,
        glossary,
    )

    image_bytes = image_to_png_bytes(canvas)

    started = time.monotonic()

    status_placeholder.info(
        f"Generating page with {IMAGE_MODEL} • {model_size} • {quality} quality"
    )

    # Current OpenAI Image API supports streaming partial images for edits.
    stream = client.images.edit(
        model=IMAGE_MODEL,
        image=("lecture_page.png", image_bytes, "image/png"),
        prompt=prompt,
        size=model_size,
        quality=quality,
        n=1,
        output_format="png",
        stream=True,
        partial_images=partial_images,
    )

    final_image = None
    partial_count = 0

    for event in stream:
        elapsed = time.monotonic() - started

        timer_placeholder.caption(
            f"Elapsed: {elapsed:.1f} seconds"
        )

        event_type = getattr(event, "type", "")

        # Partial image event.
        partial = extract_image_from_event(event)

        if partial is not None:
            partial_count += 1

            # Partial images are previews of the model's current result.
            # Show them immediately, but do not treat them as final.
            preview_placeholder.image(
                partial,
                caption=f"Live model preview {partial_count}",
                width="stretch",
            )

            status_placeholder.info(
                f"Generating page • live preview {partial_count}"
            )

        # Final image event.
        if (
            "completed" in str(event_type).lower()
            or "complete" in str(event_type).lower()
        ):
            possible_final = extract_image_from_event(event)
            if possible_final is not None:
                final_image = possible_final

        # Some SDK versions may expose the final result directly.
        response = getattr(event, "response", None)
        if response is not None:
            possible_final = extract_final_image(response)
            if possible_final is not None:
                final_image = possible_final

    # If the streaming SDK did not expose the final image as an event,
    # use the stream's final response if available.
    if final_image is None:
        response = getattr(stream, "response", None)
        if response is not None:
            final_image = extract_final_image(response)

    if final_image is None:
        raise RuntimeError(
            "The image stream finished without returning a final image."
        )

    # Remove the padding introduced before the model call.
    final_cropped = final_image.crop(crop_box)

    # Restore the exact source page pixel dimensions.
    final_restored = final_cropped.resize(
        source_image.size,
        Image.Resampling.LANCZOS,
    )

    elapsed = time.monotonic() - started

    timer_placeholder.caption(
        f"Completed in {elapsed:.1f} seconds"
    )

    status_placeholder.success(
        f"Page completed • {elapsed:.1f} seconds"
    )

    preview_placeholder.image(
        final_restored,
        caption="Final translated page",
        width="stretch",
    )

    return final_restored


def translate_page_with_fallback(
    client: OpenAI,
    source_image: Image.Image,
    source_language: str,
    target_language: str,
    glossary: str,
    preview_placeholder,
    status_placeholder,
    timer_placeholder,
    quality: str,
    partial_images: int,
) -> Image.Image:
    """
    Try streaming first.

    If an older openai Python SDK does not support streaming edit arguments,
    fall back to a normal edit request so the application still works.
    """

    try:
        return translate_page_streaming(
            client=client,
            source_image=source_image,
            source_language=source_language,
            target_language=target_language,
            glossary=glossary,
            preview_placeholder=preview_placeholder,
            status_placeholder=status_placeholder,
            timer_placeholder=timer_placeholder,
            quality=quality,
            partial_images=partial_images,
        )

    except TypeError as exc:
        # Usually means the installed SDK does not recognize stream or
        # partial_images yet.
        if "stream" not in str(exc).lower() and "partial" not in str(exc).lower():
            raise

        status_placeholder.warning(
            "Streaming is not supported by this installed OpenAI SDK. "
            "Continuing with a normal image-edit request..."
        )

        model_size = choose_model_size(source_image)

        if model_size == PORTRAIT_SIZE:
            canvas_size = (1024, 1536)
        else:
            canvas_size = (1536, 1024)

        canvas, crop_box = make_edit_canvas(source_image, canvas_size)

        prompt = build_translation_prompt(
            source_language,
            target_language,
            glossary,
        )

        started = time.monotonic()

        response = client.images.edit(
            model=IMAGE_MODEL,
            image=(
                "lecture_page.png",
                image_to_png_bytes(canvas),
                "image/png",
            ),
            prompt=prompt,
            size=model_size,
            quality=quality,
            n=1,
            output_format="png",
        )

        final_image = extract_final_image(response)

        if final_image is None:
            raise RuntimeError("OpenAI returned no image.")

        final_cropped = final_image.crop(crop_box)

        final_restored = final_cropped.resize(
            source_image.size,
            Image.Resampling.LANCZOS,
        )

        elapsed = time.monotonic() - started

        timer_placeholder.caption(
            f"Completed in {elapsed:.1f} seconds"
        )

        status_placeholder.success(
            f"Page completed • {elapsed:.1f} seconds"
        )

        preview_placeholder.image(
            final_restored,
            caption="Final translated page",
            width="stretch",
        )

        return final_restored


# ============================================================
# PDF output
# ============================================================

def images_to_pdf(images):
    """Create a PDF using each image's own dimensions."""
    if not images:
        raise ValueError("No images to put into PDF.")

    rgb_images = [img.convert("RGB") for img in images]

    output = io.BytesIO()

    first = rgb_images[0]
    rest = rgb_images[1:]

    first.save(
        output,
        format="PDF",
        save_all=True,
        append_images=rest,
        resolution=150.0,
    )

    return output.getvalue()


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
    "Translate lecture-page images with OpenAI's image editing model "
    "while preserving the original visual design."
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
        help="Your key is used by this Streamlit session to call OpenAI.",
    )

    source_language = st.selectbox(
        "Source language",
        [
            "English",
            "French",
            "German",
            "Spanish",
            "Italian",
            "Portuguese",
            "Turkish",
            "Arabic",
            "Persian",
            "Urdu",
            "Other",
        ],
        index=0,
    )

    target_language = st.selectbox(
        "Target language",
        [
            "Arabic",
            "English",
            "French",
            "German",
            "Spanish",
            "Italian",
            "Portuguese",
            "Turkish",
            "Persian",
            "Urdu",
            "Other",
        ],
        index=0,
    )

    glossary = st.text_area(
        "Optional terminology glossary",
        placeholder=(
            "magnetic flux density = كثافة الفيض المغناطيسي\n"
            "Taqwa = تقوى"
        ),
        height=120,
    )

    st.divider()

    dpi = st.slider(
        "PDF rendering quality",
        min_value=100,
        max_value=220,
        value=150,
        step=10,
        help="Higher values create larger input images.",
    )

    quality = st.selectbox(
        "Image generation quality",
        ["low", "medium", "high"],
        index=1,
        help=(
            "Medium is recommended for testing. High may improve visual "
            "quality but can take longer and cost more."
        ),
    )

    partial_images = st.slider(
        "Live previews",
        min_value=1,
        max_value=3,
        value=2,
        help=(
            "Number of partial images requested while the model generates "
            "the final page."
        ),
    )


# ------------------------------------------------------------
# Main input
# ------------------------------------------------------------

uploaded_pdf = st.file_uploader(
    "Upload lecture PDF",
    type=["pdf"],
)

if not uploaded_pdf:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded_pdf.getvalue()

try:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
except Exception as exc:
    st.error(f"Could not open the PDF: {exc}")
    st.stop()

page_count = len(doc)

st.success(
    f"Loaded **{uploaded_pdf.name}** — {page_count} pages."
)

# ------------------------------------------------------------
# Page selection
# ------------------------------------------------------------

st.subheader("Choose pages")

col1, col2 = st.columns(2)

with col1:
    start_page = st.number_input(
        "Start page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    )

with col2:
    end_page = st.number_input(
        "End page",
        min_value=1,
        max_value=page_count,
        value=min(page_count, 1),
        step=1,
    )

if start_page > end_page:
    st.error("Start page must be less than or equal to end page.")
    st.stop()

selected_count = end_page - start_page + 1

st.info(
    f"{selected_count} page(s) selected. "
    "Image editing uses API credits. Test 1–3 pages first."
)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

translate_button = st.button(
    "✨ Translate selected pages",
    type="primary",
    use_container_width=False,
)

if translate_button:
    if not api_key.strip():
        st.error("Please enter your OpenAI API key.")
        st.stop()

    if source_language == target_language:
        st.warning(
            "Source and target languages are the same. "
            "The model may make unnecessary changes."
        )

    try:
        client = OpenAI(api_key=api_key.strip())
    except Exception as exc:
        st.error(f"Could not initialize OpenAI client: {exc}")
        st.stop()

    translated_images = []

    progress = st.progress(0.0)

    overall_status = st.empty()

    st.divider()

    for index, page_number in enumerate(
        range(int(start_page), int(end_page) + 1),
        start=1,
    ):
        overall_status.info(
            f"Translating page {page_number} / {page_count} "
            f"({index} / {selected_count})..."
        )

        # Render source page.
        try:
            source_image = render_pdf_page(
                doc,
                page_number - 1,
                dpi=dpi,
            )
        except Exception as exc:
            st.error(
                f"Could not render page {page_number}: {exc}"
            )
            continue

        page_col1, page_col2 = st.columns(2)

        with page_col1:
            st.markdown(f"**Original — page {page_number}**")
            st.image(
                source_image,
                width="stretch",
            )

        with page_col2:
            st.markdown(f"**Translated — page {page_number}**")

            preview_placeholder = st.empty()
            status_placeholder = st.empty()
            timer_placeholder = st.empty()

            preview_placeholder.image(
                source_image,
                caption="Waiting for model...",
                width="stretch",
            )

        try:
            translated = translate_page_with_fallback(
                client=client,
                source_image=source_image,
                source_language=source_language,
                target_language=target_language,
                glossary=glossary,
                preview_placeholder=preview_placeholder,
                status_placeholder=status_placeholder,
                timer_placeholder=timer_placeholder,
                quality=quality,
                partial_images=partial_images,
            )

            translated_images.append(translated)

        except Exception as exc:
            status_placeholder.error(
                f"Page {page_number} failed: {exc}"
            )

            st.warning(
                f"Keeping the original page {page_number} in the output "
                "because translation failed."
            )

            translated_images.append(source_image)

        progress.progress(index / selected_count)

    overall_status.success(
        f"Finished processing {selected_count} selected page(s)."
    )

    # --------------------------------------------------------
    # Create final PDF
    # --------------------------------------------------------

    if translated_images:
        try:
            output_pdf = images_to_pdf(translated_images)

            st.divider()
            st.subheader("Finished")

            st.download_button(
                label="⬇️ Download translated PDF",
                data=output_pdf,
                file_name=(
                    f"{uploaded_pdf.name.rsplit('.', 1)[0]}"
                    f"_translated.pdf"
                ),
                mime="application/pdf",
            )

            st.success(
                "The translated PDF is ready."
            )

        except Exception as exc:
            st.error(
                f"Could not create the output PDF: {exc}"
            )
