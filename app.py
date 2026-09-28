
import io
import base64
import hashlib
import time
from typing import Optional, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
from openai import OpenAI

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    COORDINATES_AVAILABLE = True
except ImportError:
    COORDINATES_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

IMAGE_MODEL = "gpt-image-2"

LANDSCAPE_SIZE = "1536x1024"
PORTRAIT_SIZE = "1024x1536"

DEFAULT_DPI = 150
CLICK_IMAGE_WIDTH = 760


# ============================================================
# Session-state helpers
# ============================================================

def init_state():
    defaults = {
        "document_id": None,
        "source_pages": {},
        "translated_pages": {},
        "translation_done": False,
        "fix_count": {},
        "feedback_text": {},
        "clicked_points": {},
        "last_fix_message": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_results():
    st.session_state.source_pages = {}
    st.session_state.translated_pages = {}
    st.session_state.translation_done = False
    st.session_state.fix_count = {}
    st.session_state.feedback_text = {}
    st.session_state.clicked_points = {}
    st.session_state.last_fix_message = {}


init_state()


# ============================================================
# PDF / image helpers
# ============================================================

def render_pdf_page(doc: fitz.Document, page_number: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    page = doc.load_page(page_number)
    zoom = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def choose_model_size(image: Image.Image) -> str:
    w, h = image.size
    if h > w * 1.05:
        return PORTRAIT_SIZE
    return LANDSCAPE_SIZE


def make_edit_canvas(
    image: Image.Image,
    canvas_size: Tuple[int, int],
) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
    """
    Fit an image into the model canvas without changing its aspect ratio.

    Returns:
      canvas
      crop_box
      scale used to map source -> canvas
    """
    cw, ch = canvas_size
    iw, ih = image.size

    scale = min(cw / iw, ch / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))

    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)

    tiny = image.resize((1, 1), Image.Resampling.BOX)
    background = tiny.getpixel((0, 0))

    canvas = Image.new("RGB", (cw, ch), background)

    x = (cw - nw) // 2
    y = (ch - nh) // 2

    canvas.paste(resized, (x, y))

    return canvas, (x, y, x + nw, y + nh), scale


def restore_from_model_image(
    generated: Image.Image,
    crop_box: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
) -> Image.Image:
    cropped = generated.crop(crop_box)
    return cropped.resize(original_size, Image.Resampling.LANCZOS)


def decode_b64_image(value: str) -> Image.Image:
    return Image.open(
        io.BytesIO(base64.b64decode(value))
    ).convert("RGB")


def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def extract_final_image(response) -> Optional[Image.Image]:
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


def extract_stream_image(event) -> Optional[Image.Image]:
    candidates = []

    for attr in ("partial_image_b64", "b64_json"):
        value = getattr(event, attr, None)
        if value:
            candidates.append(value)

    data = getattr(event, "data", None)
    if data is not None:
        for attr in ("partial_image_b64", "b64_json"):
            value = getattr(data, attr, None)
            if value:
                candidates.append(value)

    if isinstance(event, dict):
        for key in ("partial_image_b64", "b64_json"):
            if event.get(key):
                candidates.append(event[key])

        data = event.get("data")
        if isinstance(data, dict):
            for key in ("partial_image_b64", "b64_json"):
                if data.get(key):
                    candidates.append(data[key])

    for value in candidates:
        try:
            return decode_b64_image(value)
        except Exception:
            continue

    return None


# ============================================================
# Translation prompts
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

PRIMARY OBJECTIVE:
Translate the text while preserving the original page.

PRESERVE THE ORIGINAL VISUAL DESIGN:
- Preserve illustrations, photographs, people, objects and icons.
- Preserve diagrams, charts, arrows, lines, tables and shapes.
- Preserve backgrounds, colors, textures, lighting and shadows.
- Preserve the same page composition and text locations.
- Preserve the visual hierarchy and typography style.
- Do not redesign, modernize, simplify or beautify the slide.
- Do not invent or replace illustrations.
- Do not add decorative elements.

TEXT:
- Translate rather than summarize.
- Translate every visible natural-language phrase that should be translated.
- Do not invent explanations.
- Do not add commentary.
- Do not omit visible text.
- Keep numbers, equations, units and mathematical symbols unchanged unless
  they are ordinary-language words.
- Preserve names and established technical terminology where appropriate.
- Keep labels in diagrams in their original locations.
- Keep text inside boxes, speech bubbles, signs and callouts in the same areas.
- Preserve approximate font size, weight, alignment and line structure.
- If translated text is longer, wrap it intelligently or slightly reduce
  font size so it remains inside the original text area.
- Do not leave the original-language text behind.

ARABIC / RTL:
If the target language is Arabic:
- Use real Arabic Unicode.
- Use correct right-to-left direction.
- Properly connect Arabic letters.
- Never produce missing-glyph squares or boxes.
- Never transliterate Arabic into Latin letters.
- Preserve appropriate Arabic punctuation.
- Example:
  "NO! Cheating is haram" -> "لا! الغش حرام."
- "haram" in this context means "حرام".

FINNISH:
If the target language is Finnish:
- Use natural, grammatically correct Finnish.
- Preserve Finnish special characters such as ä and ö.
- Do not translate technical terms inconsistently.
- Do not turn Finnish text into English-style word order.

FINAL REQUIREMENT:
The output should look like the SAME ORIGINAL PAGE made by the SAME DESIGNER,
with the natural-language text translated into the target language.

{glossary_instruction}
""".strip()


def build_fix_prompt(
    source_language: str,
    target_language: str,
    feedback: str,
    point_description: str,
    glossary: str,
) -> str:
    glossary_instruction = ""

    if glossary.strip():
        glossary_instruction = f"""
Use these glossary translations where applicable:
{glossary.strip()}
"""

    return f"""
EDIT THIS EXISTING TRANSLATED LECTURE PAGE.

This is a CORRECTION pass, not a redesign.

The current image is already translated from {source_language} to {target_language}.

USER'S CORRECTION:
{feedback.strip()}

{point_description}

IMPORTANT:
- Fix the mistake described by the user.
- If the user says a translation is wrong, replace it with the correct translation.
- If the user says text is missing, add the missing translated text in the
  appropriate original location.
- If the user says text is misspelled, correct only that text.
- If the user says a diagram label is wrong, correct that label.
- If the user says an illustration or layout changed, restore it as closely
  as possible to the original page.
- Compare against the ORIGINAL PAGE reference image when necessary.
- Do not redesign the slide.
- Do not change unrelated text.
- Do not change unrelated illustrations, diagrams, colors, backgrounds,
  people, objects, spacing or composition.
- Preserve the original visual style.
- Keep all correct translations unchanged.

If the target language is Arabic:
- Use correct Arabic Unicode and RTL.
- Properly connect Arabic letters.
- Never use missing-glyph squares.
- Never transliterate Arabic into Latin letters.

If the target language is Finnish:
- Use natural, grammatically correct Finnish.
- Preserve ä and ö and correct Finnish spelling.

{glossary_instruction}

Make the smallest necessary correction and return the corrected page.
""".strip()


# ============================================================
# Mask creation for a clicked mistake
# ============================================================

def create_local_edit_mask(
    canvas_size: Tuple[int, int],
    crop_box: Tuple[int, int, int, int],
    source_size: Tuple[int, int],
    clicked_point: Optional[Tuple[int, int]],
    radius_source_px: int,
) -> Optional[Image.Image]:
    """
    Create an RGBA mask with a transparent edit region around the user's
    clicked point. The rest is opaque.

    OpenAI's image-edit mask uses the alpha channel as guidance for the
    editable area.
    """
    if clicked_point is None:
        return None

    x_source, y_source = clicked_point
    sx1, sy1, sx2, sy2 = crop_box

    source_w, source_h = source_size
    canvas_w = sx2 - sx1
    canvas_h = sy2 - sy1

    scale_x = canvas_w / source_w
    scale_y = canvas_h / source_h

    x_canvas = sx1 + x_source * scale_x
    y_canvas = sy1 + y_source * scale_y

    radius_x = max(20, int(radius_source_px * scale_x))
    radius_y = max(20, int(radius_source_px * scale_y))

    mask = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)

    # Transparent = editable region; opaque = preserve region.
    draw.ellipse(
        (
            int(x_canvas - radius_x),
            int(y_canvas - radius_y),
            int(x_canvas + radius_x),
            int(y_canvas + radius_y),
        ),
        fill=(0, 0, 0, 0),
    )

    return mask


def mark_point_on_image(
    image: Image.Image,
    point: Optional[Tuple[int, int]],
    radius: int = 35,
) -> Image.Image:
    if point is None:
        return image

    result = image.copy()
    draw = ImageDraw.Draw(result)

    x, y = point

    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 0, 0),
        width=max(3, radius // 8),
    )

    return result


# ============================================================
# Streaming translation
# ============================================================

def translate_page(
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

    model_size = choose_model_size(source_image)

    canvas_size = (
        (1024, 1536)
        if model_size == PORTRAIT_SIZE
        else (1536, 1024)
    )

    canvas, crop_box, _ = make_edit_canvas(
        source_image,
        canvas_size,
    )

    prompt = build_translation_prompt(
        source_language,
        target_language,
        glossary,
    )

    started = time.monotonic()

    status_placeholder.info(
        f"Generating page • {model_size} • {quality} quality"
    )

    try:
        stream = client.images.edit(
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
            stream=True,
            partial_images=partial_images,
        )

        final_image = None
        partial_number = 0

        for event in stream:
            elapsed = time.monotonic() - started
            timer_placeholder.caption(
                f"Elapsed: {elapsed:.1f} seconds"
            )

            partial = extract_stream_image(event)

            if partial is not None:
                partial_number += 1
                preview_placeholder.image(
                    partial,
                    caption=f"Live preview {partial_number}",
                    width="stretch",
                )
                status_placeholder.info(
                    f"Generating page • live preview {partial_number}"
                )

            event_type = str(getattr(event, "type", "")).lower()

            if "completed" in event_type or "complete" in event_type:
                possible = extract_stream_image(event)
                if possible is not None:
                    final_image = possible

            response = getattr(event, "response", None)
            if response is not None:
                possible = extract_final_image(response)
                if possible is not None:
                    final_image = possible

        if final_image is None:
            response = getattr(stream, "response", None)
            if response is not None:
                final_image = extract_final_image(response)

        if final_image is None:
            raise RuntimeError(
                "The image stream finished without returning a final image."
            )

    except TypeError as exc:
        # Compatibility fallback for an older OpenAI Python SDK.
        message = str(exc).lower()

        if "stream" not in message and "partial" not in message:
            raise

        status_placeholder.warning(
            "Streaming is not supported by the installed OpenAI SDK. "
            "Using a normal image edit request instead."
        )

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

    result = restore_from_model_image(
        final_image,
        crop_box,
        source_image.size,
    )

    elapsed = time.monotonic() - started

    timer_placeholder.caption(
        f"Completed in {elapsed:.1f} seconds"
    )

    status_placeholder.success(
        f"Page completed • {elapsed:.1f} seconds"
    )

    preview_placeholder.image(
        result,
        caption="Final translated page",
        width="stretch",
    )

    return result


# ============================================================
# Correction / feedback editing
# ============================================================

def fix_page(
    client: OpenAI,
    current_image: Image.Image,
    original_image: Image.Image,
    source_language: str,
    target_language: str,
    feedback: str,
    clicked_point: Optional[Tuple[int, int]],
    radius_source_px: int,
    glossary: str,
    preview_placeholder,
    status_placeholder,
    timer_placeholder,
    quality: str,
) -> Image.Image:

    if not feedback.strip():
        raise ValueError(
            "Please describe what is wrong before clicking Fix this page."
        )

    model_size = choose_model_size(current_image)

    canvas_size = (
        (1024, 1536)
        if model_size == PORTRAIT_SIZE
        else (1536, 1024)
    )

    current_canvas, crop_box, _ = make_edit_canvas(
        current_image,
        canvas_size,
    )

    # Original page is supplied as a second reference so the model can
    # recover the intended wording/layout when the correction requires it.
    original_canvas, _, _ = make_edit_canvas(
        original_image,
        canvas_size,
    )

    point_description = "No point was selected; use the user's written feedback."

    if clicked_point is not None:
        x, y = clicked_point
        point_description = (
            f"The user clicked approximately at pixel ({x}, {y}) on the "
            f"current page. Focus the correction around that location."
        )

    prompt = build_fix_prompt(
        source_language,
        target_language,
        feedback,
        point_description,
        glossary,
    )

    mask = create_local_edit_mask(
        canvas_size=canvas_size,
        crop_box=crop_box,
        source_size=current_image.size,
        clicked_point=clicked_point,
        radius_source_px=radius_source_px,
    )

    if mask is not None:
        status_placeholder.info(
            "Fixing the marked area while preserving the rest of the page..."
        )
    else:
        status_placeholder.info(
            "Fixing the page according to your feedback..."
        )

    started = time.monotonic()

    # Current page is image 1. Original page is image 2.
    # The mask applies to the first image.
    images = [
        (
            "current_translated_page.png",
            image_to_png_bytes(current_canvas),
            "image/png",
        ),
        (
            "original_page.png",
            image_to_png_bytes(original_canvas),
            "image/png",
        ),
    ]

    kwargs = dict(
        model=IMAGE_MODEL,
        image=images,
        prompt=prompt,
        size=model_size,
        quality=quality,
        n=1,
        output_format="png",
    )

    if mask is not None:
        kwargs["mask"] = (
            "edit_mask.png",
            image_to_png_bytes(mask),
            "image/png",
        )

    # For corrections we deliberately use a normal request.
    # This makes the "Fix" action simple and reliable, and the final
    # corrected image replaces the current version in session state.
    response = client.images.edit(**kwargs)

    fixed_model_image = extract_final_image(response)

    if fixed_model_image is None:
        raise RuntimeError("OpenAI returned no corrected image.")

    fixed_page = restore_from_model_image(
        fixed_model_image,
        crop_box,
        current_image.size,
    )

    elapsed = time.monotonic() - started

    timer_placeholder.caption(
        f"Correction completed in {elapsed:.1f} seconds"
    )

    status_placeholder.success(
        "Correction completed."
    )

    preview_placeholder.image(
        fixed_page,
        caption="Corrected page",
        width="stretch",
    )

    return fixed_page


# ============================================================
# PDF output
# ============================================================

def images_to_pdf(images):
    if not images:
        raise ValueError("No images to put into PDF.")

    rgb_images = [img.convert("RGB") for img in images]

    output = io.BytesIO()

    rgb_images[0].save(
        output,
        format="PDF",
        save_all=True,
        append_images=rgb_images[1:],
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
    "Translate lecture pages with OpenAI image editing, review them, "
    "point to mistakes, and ask the model to correct individual pages."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
        help="Used only by this Streamlit session.",
    )

    languages = ["English", "Arabic", "Finnish"]

    source_language = st.selectbox(
        "Source language",
        languages,
        index=0,
    )

    target_language = st.selectbox(
        "Target language",
        languages,
        index=1,
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
    )

    quality = st.selectbox(
        "Image generation quality",
        ["low", "medium", "high"],
        index=1,
        help="Medium is recommended while testing.",
    )

    partial_images = st.slider(
        "Live previews",
        min_value=1,
        max_value=3,
        value=2,
    )

    st.divider()

    st.subheader("Correction settings")

    correction_radius = st.slider(
        "Marked-area radius",
        min_value=30,
        max_value=400,
        value=140,
        step=10,
        help=(
            "When you click a mistake, this radius defines the area the "
            "model is encouraged to edit."
        ),
    )

    if not COORDINATES_AVAILABLE:
        st.warning(
            "Click-to-point support is not installed. "
            "Run: pip install streamlit-image-coordinates"
        )


# ============================================================
# PDF upload
# ============================================================

uploaded_pdf = st.file_uploader(
    "Upload lecture PDF",
    type=["pdf"],
)

if not uploaded_pdf:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded_pdf.getvalue()
document_id = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.document_id != document_id:
    reset_results()
    st.session_state.document_id = document_id

try:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
except Exception as exc:
    st.error(f"Could not open the PDF: {exc}")
    st.stop()

page_count = len(doc)

st.success(
    f"Loaded **{uploaded_pdf.name}** — {page_count} pages."
)


# ============================================================
# Page selection
# ============================================================

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
    "You can review and correct individual pages after translation."
)


# ============================================================
# Translate selected pages
# ============================================================

translate_button = st.button(
    "✨ Translate selected pages",
    type="primary",
)

if translate_button:
    if not api_key.strip():
        st.error("Please enter your OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=api_key.strip())

    progress = st.progress(0.0)
    overall_status = st.empty()

    for index, page_number in enumerate(
        range(int(start_page), int(end_page) + 1),
        start=1,
    ):
        overall_status.info(
            f"Translating page {page_number} / {page_count} "
            f"({index} / {selected_count})..."
        )

        try:
            source_image = render_pdf_page(
                doc,
                page_number - 1,
                dpi=dpi,
            )

            st.session_state.source_pages[page_number] = (
                image_to_png_bytes(source_image)
            )

            left, right = st.columns(2)

            with left:
                st.markdown(f"**Original — page {page_number}**")
                st.image(
                    source_image,
                    width="stretch",
                )

            with right:
                st.markdown(f"**Translated — page {page_number}**")

                preview_placeholder = st.empty()
                status_placeholder = st.empty()
                timer_placeholder = st.empty()

                preview_placeholder.image(
                    source_image,
                    caption="Waiting for model...",
                    width="stretch",
                )

                translated = translate_page(
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

                st.session_state.translated_pages[page_number] = (
                    image_to_png_bytes(translated)
                )

        except Exception as exc:
            st.error(
                f"Page {page_number} failed: {exc}"
            )

            # Keep the original if translation failed.
            if page_number not in st.session_state.source_pages:
                try:
                    source_image = render_pdf_page(
                        doc,
                        page_number - 1,
                        dpi=dpi,
                    )
                    st.session_state.source_pages[page_number] = (
                        image_to_png_bytes(source_image)
                    )
                except Exception:
                    continue

            st.session_state.translated_pages[page_number] = (
                st.session_state.source_pages[page_number]
            )

        progress.progress(index / selected_count)

    st.session_state.translation_done = True

    overall_status.success(
        "Translation finished. Review the pages below and correct any mistakes."
    )


# ============================================================
# Review / correction area
# ============================================================

if st.session_state.translated_pages:

    st.divider()
    st.header("🔎 Review and correct pages")

    st.caption(
        "If a page has a mistake, click the mistake on the image, "
        "describe what is wrong, and press **Fix this page**. "
        "The correction replaces only that page in the working PDF."
    )

    if not COORDINATES_AVAILABLE:
        st.warning(
            "For click-to-point correction, install "
            "`streamlit-image-coordinates` and restart Streamlit."
        )

    selected_pages_for_review = sorted(
        st.session_state.translated_pages.keys()
    )

    for page_number in selected_pages_for_review:
        current_image = png_bytes_to_image(
            st.session_state.translated_pages[page_number]
        )

        original_bytes = st.session_state.source_pages.get(page_number)
        if original_bytes is not None:
            original_image = png_bytes_to_image(original_bytes)
        else:
            original_image = current_image

        fix_number = st.session_state.fix_count.get(
            page_number,
            0,
        )

        with st.expander(
            f"Page {page_number}"
            + (f" • {fix_number} correction(s)" if fix_number else ""),
            expanded=True,
        ):
            review_left, review_right = st.columns(2)

            with review_left:
                st.markdown("**Original**")
                st.image(
                    original_image,
                    width="stretch",
                )

            with review_right:
                st.markdown(
                    "**Current translation — click the mistake**"
                )

                clicked = None

                if COORDINATES_AVAILABLE:
                    click_result = streamlit_image_coordinates(
                        current_image,
                        width=CLICK_IMAGE_WIDTH,
                        cursor="crosshair",
                        key=f"page_click_{document_id}_{page_number}",
                    )

                    if click_result:
                        x_display = int(click_result["x"])
                        y_display = int(click_result["y"])

                        display_w = int(click_result.get(
                            "width",
                            CLICK_IMAGE_WIDTH,
                        ))
                        display_h = int(click_result.get(
                            "height",
                            round(
                                CLICK_IMAGE_WIDTH
                                * current_image.height
                                / current_image.width
                            ),
                        ))

                        # Map displayed coordinates back to native image coordinates.
                        x_native = round(
                            x_display
                            * current_image.width
                            / max(1, display_w)
                        )

                        y_native = round(
                            y_display
                            * current_image.height
                            / max(1, display_h)
                        )

                        clicked = (
                            max(0, min(current_image.width - 1, x_native)),
                            max(0, min(current_image.height - 1, y_native)),
                        )

                        st.session_state.clicked_points[page_number] = clicked

                    clicked = st.session_state.clicked_points.get(
                        page_number
                    )

                    if clicked:
                        marked = mark_point_on_image(
                            current_image,
                            clicked,
                            radius=max(
                                18,
                                correction_radius // 2,
                            ),
                        )

                        st.image(
                            marked,
                            caption=(
                                f"Selected point: "
                                f"({clicked[0]}, {clicked[1]})"
                            ),
                            width="stretch",
                        )

                else:
                    st.image(
                        current_image,
                        width="stretch",
                    )

            st.markdown("**What is wrong?**")

            feedback_key = f"feedback_{document_id}_{page_number}"

            feedback = st.text_area(
                f"Correction instructions for page {page_number}",
                value=st.session_state.feedback_text.get(
                    page_number,
                    "",
                ),
                placeholder=(
                    "Examples:\n"
                    "- The sentence at the top right is translated incorrectly. "
                    "It should say: ...\n"
                    "- The Arabic word in the diagram is wrong; it should be ...\n"
                    "- The Finnish translation of this heading is incorrect. "
                    "Use: ...\n"
                    "- The equation is correct; only fix the label next to it."
                ),
                height=120,
                key=feedback_key,
            )

            st.session_state.feedback_text[page_number] = feedback

            action_col1, action_col2, action_col3 = st.columns(
                [1, 1, 2]
            )

            with action_col1:
                fix_button = st.button(
                    "🛠️ Fix this page",
                    key=f"fix_{document_id}_{page_number}",
                    type="primary",
                )

            with action_col2:
                clear_point_button = st.button(
                    "Clear point",
                    key=f"clear_point_{document_id}_{page_number}",
                )

            with action_col3:
                if clicked:
                    st.caption(
                        "A marked area will guide the correction. "
                        "The written feedback is still the main instruction."
                    )
                else:
                    st.caption(
                        "No point selected. The model will use your written feedback."
                    )

            if clear_point_button:
                st.session_state.clicked_points.pop(
                    page_number,
                    None,
                )
                st.rerun()

            if fix_button:
                if not api_key.strip():
                    st.error(
                        "Enter your OpenAI API key before fixing a page."
                    )
                    continue

                if not feedback.strip():
                    st.error(
                        "Describe the mistake before clicking Fix this page."
                    )
                    continue

                client = OpenAI(api_key=api_key.strip())

                fix_preview = st.empty()
                fix_status = st.empty()
                fix_timer = st.empty()

                with st.spinner(
                    "Applying your correction..."
                ):
                    try:
                        fixed = fix_page(
                            client=client,
                            current_image=current_image,
                            original_image=original_image,
                            source_language=source_language,
                            target_language=target_language,
                            feedback=feedback,
                            clicked_point=clicked,
                            radius_source_px=correction_radius,
                            glossary=glossary,
                            preview_placeholder=fix_preview,
                            status_placeholder=fix_status,
                            timer_placeholder=fix_timer,
                            quality=quality,
                        )

                        st.session_state.translated_pages[page_number] = (
                            image_to_png_bytes(fixed)
                        )

                        st.session_state.fix_count[page_number] = (
                            st.session_state.fix_count.get(page_number, 0) + 1
                        )

                        st.session_state.last_fix_message[page_number] = (
                            "Correction applied. Review the page again."
                        )

                        st.success(
                            "Correction applied. You can submit another correction "
                            "if anything is still wrong."
                        )

                    except Exception as exc:
                        st.error(
                            f"Could not correct page {page_number}: {exc}"
                        )

            if page_number in st.session_state.last_fix_message:
                st.info(
                    st.session_state.last_fix_message[page_number]
                )


# ============================================================
# Final PDF
# ============================================================

if st.session_state.translated_pages:

    st.divider()
    st.header("📄 Output")

    # Only pages that have actually been translated/reviewed are included.
    output_page_numbers = sorted(
        st.session_state.translated_pages.keys()
    )

    output_images = [
        png_bytes_to_image(
            st.session_state.translated_pages[p]
        )
        for p in output_page_numbers
    ]

    try:
        output_pdf = images_to_pdf(output_images)

        st.download_button(
            "⬇️ Download current translated PDF",
            data=output_pdf,
            file_name=(
                f"{uploaded_pdf.name.rsplit('.', 1)[0]}"
                f"_translated.pdf"
            ),
            mime="application/pdf",
        )

        st.caption(
            "The downloaded PDF contains the pages you selected, "
            "including any corrections you applied."
        )

    except Exception as exc:
        st.error(f"Could not create the output PDF: {exc}")


# ============================================================
# Reset
# ============================================================

if st.session_state.translated_pages:
    if st.button("🗑️ Clear translated results"):
        reset_results()
        st.rerun()
