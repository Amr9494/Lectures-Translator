
import io
import base64
import json
import time
from typing import Optional, Tuple

import fitz
import requests
from PIL import Image, ImageDraw
import streamlit as st
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

TRANSLATION_MODEL = "gpt-5.6-luna"
IMAGE_MODEL = "gpt-image-2"

LANGUAGES = ["English", "Arabic", "Finnish"]

LANDSCAPE_SIZE = "1536x1024"
PORTRAIT_SIZE = "1024x1536"
DEFAULT_DPI = 150


# ============================================================
# General helpers
# ============================================================

def render_pdf_page(doc: fitz.Document, page_number: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    page = doc.load_page(page_number)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def decode_b64_image(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


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


def choose_model_size(image: Image.Image) -> str:
    w, h = image.size
    return PORTRAIT_SIZE if h > w * 1.05 else LANDSCAPE_SIZE


def make_edit_canvas(
    image: Image.Image,
    canvas_size: Tuple[int, int],
):
    cw, ch = canvas_size
    iw, ih = image.size

    scale = min(cw / iw, ch / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))

    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)

    bg = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", (cw, ch), bg)

    x = (cw - nw) // 2
    y = (ch - nh) // 2

    canvas.paste(resized, (x, y))
    return canvas, (x, y, x + nw, y + nh)


def restore_from_canvas(
    generated: Image.Image,
    crop_box,
    original_size,
):
    cropped = generated.crop(crop_box)
    return cropped.resize(original_size, Image.Resampling.LANCZOS)


# ============================================================
# PDF text extraction
# ============================================================

def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    """
    Extract selectable PDF text when available.

    If the page is image-only, return an empty string and the
    contextual model will use the page image for OCR/understanding.
    """
    text = doc.load_page(page_index).get_text("text")
    return text.strip()


def build_context(
    doc: fitz.Document,
    page_index: int,
    window: int = 1,
):
    """
    Return previous/current/next page text.

    Neighboring slides are context only; they are not themselves translated.
    """
    parts = []

    start = max(0, page_index - window)
    end = min(len(doc), page_index + window + 1)

    for i in range(start, end):
        label = "CURRENT SLIDE" if i == page_index else f"CONTEXT SLIDE {i + 1}"

        text = extract_page_text(doc, i)

        if text:
            parts.append(
                f"--- {label} / PDF PAGE {i + 1} ---\n{text}"
            )
        else:
            parts.append(
                f"--- {label} / PDF PAGE {i + 1} ---\n"
                "[No selectable text; use the page image for visual context.]"
            )

    return "\n\n".join(parts)


# ============================================================
# Contextual translation
# ============================================================

def translation_system_prompt(
    source_language: str,
    target_language: str,
) -> str:
    return f"""
You are a senior professional translator specializing in lecture slides,
educational material, and context-sensitive multilingual translation.

Translate from {source_language} to {target_language}.

The goal is NOT literal word-for-word translation.

The goal is a translation that:
- preserves the intended meaning;
- sounds natural to a native speaker;
- respects the surrounding sentence and lecture context;
- respects the subject/domain;
- handles idioms and expressions by MEANING rather than by individual words;
- uses established terminology where appropriate;
- preserves the speaker's tone and level of formality;
- does not add explanations that are not in the source;
- does not omit meaning;
- does not hallucinate content;
- does not blindly follow a dictionary translation when that would sound wrong.

IMPORTANT:
When a phrase has several possible meanings, infer the intended meaning from
the current slide, neighboring-slide context, title, diagrams, and subject
matter.

For Arabic:
- produce natural Modern Standard Arabic unless the source context clearly
  calls for another register;
- use idiomatic Arabic rather than English-shaped Arabic;
- preserve established Islamic/religious terminology when the context is
  religious;
- preserve technical terminology when the context is scientific/medical/etc.;
- use correct Arabic grammar, agreement, punctuation, and RTL wording.

For Finnish:
- produce natural contemporary Finnish;
- prefer idiomatic Finnish expressions over English-shaped calques;
- preserve established academic/technical terminology;
- respect Finnish sentence structure and register.

EXAMPLES OF THE PRINCIPLE:

"Be mindful of Allah" must NOT automatically become a literal phrase such as
"كن واعيًا بالله".

Depending on context, natural Arabic could instead be something like:
"اتقِ الله", "راقب الله", or "احفظ الله".
Choose the expression that actually fits the surrounding context. Do not
force any one example if the context supports another.

Likewise, "It's haram" should not mechanically become an unnatural construction
such as "الأمر حرام" when the intended meaning is simply "إنه حرام" or
"هذا حرام". Choose the natural expression for the actual sentence.

Return translations that a highly competent human translator would be comfortable
putting directly on a professional lecture slide.
""".strip()


def call_contextual_translation(
    client: OpenAI,
    source_language: str,
    target_language: str,
    current_page_image: Image.Image,
    context_text: str,
    extracted_current_text: str,
    glossary: str,
):
    """
    Use a text/vision model to create a translation map.

    The model receives the current page image plus surrounding slide text.
    It returns structured JSON with source snippets and approved translations.
    """

    glossary_block = glossary.strip() or "(No glossary provided.)"

    user_text = f"""
Translate the CURRENT SLIDE from {source_language} to {target_language}.

Use the current slide image to understand layout, visual labels, diagrams,
and text that may not be available in PDF text extraction.

Use neighboring slides only as CONTEXT.

Do not translate the neighboring slides.

Return JSON only in this exact structure:

{{
  "slide_summary": "short description of the current slide meaning",
  "translations": [
    {{
      "source": "exact or near-exact visible source phrase",
      "translation": "natural target-language translation",
      "reason": "very short context note explaining a non-literal choice"
    }}
  ]
}}

Rules:
- Include every meaningful natural-language text item that should be translated.
- Preserve numbers, equations, units, symbols and proper names when appropriate.
- Do not invent text.
- Do not translate purely decorative marks.
- Prefer natural contextual wording over literal wording.
- For short ambiguous phrases, use the whole-slide context.
- If a phrase is a technical term, use the standard term in the target language.
- If a phrase is religious, idiomatic, humorous, or culturally loaded, translate
  the intended meaning naturally.
- The "translation" field is the wording that should actually appear on the slide.

GLOSSARY:
{glossary_block}

EXTRACTED CURRENT-SLIDE TEXT:
{extracted_current_text or "(No selectable text was extracted; read the image.)"}

NEIGHBORING / CURRENT SLIDE CONTEXT:
{context_text}
"""

    response = client.responses.create(
        model=TRANSLATION_MODEL,
        input=[
            {
                "role": "system",
                "content": translation_system_prompt(
                    source_language,
                    target_language,
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text,
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + base64.b64encode(
                                image_to_png_bytes(current_page_image)
                            ).decode("utf-8")
                        ),
                    },
                ],
            },
        ],
    )

    raw = response.output_text.strip()

    # Handle accidental markdown fences.
    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "The contextual translator returned invalid JSON.\n\n"
            + raw
        ) from exc


# ============================================================
# Image translation
# ============================================================

def build_image_translation_prompt(
    source_language: str,
    target_language: str,
    translation_map,
):
    translations = translation_map.get("translations", [])

    approved_lines = []

    for item in translations:
        source = str(item.get("source", "")).strip()
        translation = str(item.get("translation", "")).strip()

        if source and translation:
            approved_lines.append(
                f'SOURCE: {source}\nTARGET: {translation}'
            )

    approved_text = "\n\n".join(approved_lines)

    return f"""
EDIT THIS EXISTING LECTURE IMAGE.

The linguistic translation has ALREADY been performed by a contextual
translation model. Your job is now VISUAL IMPLEMENTATION.

Source language: {source_language}
Target language: {target_language}

Replace the visible source-language text with the APPROVED TARGET translations
below.

APPROVED TRANSLATIONS:
{approved_text}

CRITICAL:
- Do not invent alternative translations.
- Do not translate the source again.
- Use the approved TARGET wording exactly, allowing only necessary typographic
  changes for line wrapping.
- Change only the written text.
- Preserve the original illustrations, photographs, people, diagrams, charts,
  arrows, shapes, colors, backgrounds, borders, lighting, shadows, and layout.
- Do not redesign the slide.
- Do not add explanations.
- Do not omit approved text.
- Keep text in the same visual locations.
- Match the original font weight, approximate size, alignment, hierarchy and
  visual style.
- If target text is longer, wrap or slightly resize it within the same area.

Arabic:
- render proper connected Arabic glyphs;
- use correct RTL direction;
- never use boxes/missing glyphs;
- preserve natural Arabic punctuation and line direction.

Finnish:
- render proper Finnish characters such as ä and ö.

The final image should look like the SAME original slide, except its
natural-language text has been replaced by the approved contextual
translations.
""".strip()


def translate_visual_page_streaming(
    client: OpenAI,
    source_image: Image.Image,
    source_language: str,
    target_language: str,
    translation_map,
    preview_placeholder,
    status_placeholder,
    timer_placeholder,
    quality: str,
    partial_images: int,
):
    model_size = choose_model_size(source_image)

    canvas_size = (
        (1024, 1536)
        if model_size == PORTRAIT_SIZE
        else (1536, 1024)
    )

    canvas, crop_box = make_edit_canvas(
        source_image,
        canvas_size,
    )

    prompt = build_image_translation_prompt(
        source_language,
        target_language,
        translation_map,
    )

    started = time.monotonic()

    status_placeholder.info(
        f"Applying approved translations with {IMAGE_MODEL}..."
    )

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

    for event in stream:
        elapsed = time.monotonic() - started
        timer_placeholder.caption(f"Image edit elapsed: {elapsed:.1f}s")

        event_type = str(getattr(event, "type", ""))

        b64_data = getattr(event, "b64_json", None)

        if b64_data:
            try:
                partial = decode_b64_image(b64_data)
                preview_placeholder.image(
                    partial,
                    caption="Live image-model preview",
                    width="stretch",
                )
            except Exception:
                pass

        if "completed" in event_type.lower():
            if b64_data:
                try:
                    final_image = decode_b64_image(b64_data)
                except Exception:
                    pass

    if final_image is None:
        response = getattr(stream, "response", None)
        if response is not None:
            final_image = extract_final_image(response)

    if final_image is None:
        raise RuntimeError(
            "Image editing completed without returning a final image."
        )

    result = restore_from_canvas(
        final_image,
        crop_box,
        source_image.size,
    )

    elapsed = time.monotonic() - started

    timer_placeholder.caption(
        f"Image edit completed in {elapsed:.1f}s"
    )

    status_placeholder.success("Visual translation completed.")

    preview_placeholder.image(
        result,
        caption="Final translated slide",
        width="stretch",
    )

    return result


# ============================================================
# Local correction / feedback
# ============================================================

def crop_feedback_region(
    image: Image.Image,
    x: int,
    y: int,
    radius: int = 180,
):
    w, h = image.size

    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(w, x + radius)
    bottom = min(h, y + radius)

    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def add_click_marker(image: Image.Image, x: int, y: int):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)

    r = max(8, round(min(image.size) * 0.012))

    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        outline=(255, 0, 0),
        width=max(2, r // 3),
    )

    return marked


def correction_translation(
    client: OpenAI,
    source_language: str,
    target_language: str,
    current_page: Image.Image,
    original_page: Image.Image,
    context_text: str,
    user_feedback: str,
    glossary: str,
):
    """
    Cheap correction step: use the text/vision model to determine the exact
    corrected wording before making any image edit.
    """

    prompt = f"""
We are correcting a translation on one lecture slide.

Source language: {source_language}
Target language: {target_language}

The CURRENT translated slide is the first image.
The ORIGINAL source slide is the second image.

User feedback:
{user_feedback}

Context from the lecture:
{context_text}

Glossary:
{glossary or "(none)"}

Determine the exact target-language wording that should replace the incorrect
text.

Do NOT redesign the slide.
Do NOT discuss the whole slide.
Do NOT translate unrelated text.

Return JSON only:

{{
  "corrected_source_text": "the source phrase being corrected",
  "corrected_translation": "the natural target-language wording",
  "instruction": "short instruction for the image editor"
}}

Important:
- Respect the user's correction.
- Use context, not word-for-word translation.
- If the user's proposed wording is unnatural or contextually wrong, improve it
  while preserving their intended meaning.
- For Arabic, use natural idiomatic Arabic and correct RTL.
- For Finnish, use natural contemporary Finnish.
"""

    response = client.responses.create(
        model=TRANSLATION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + base64.b64encode(
                                image_to_png_bytes(current_page)
                            ).decode("utf-8")
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/png;base64,"
                            + base64.b64encode(
                                image_to_png_bytes(original_page)
                            ).decode("utf-8")
                        ),
                    },
                ],
            }
        ],
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1).replace("```", "").strip()

    return json.loads(raw)


def make_local_correction_prompt(
    source_language: str,
    target_language: str,
    correction_result,
):
    corrected_source = correction_result.get(
        "corrected_source_text",
        "",
    )

    corrected_translation = correction_result.get(
        "corrected_translation",
        "",
    )

    instruction = correction_result.get(
        "instruction",
        "",
    )

    return f"""
CORRECT A SMALL REGION OF AN EXISTING TRANSLATED LECTURE SLIDE.

Source language: {source_language}
Target language: {target_language}

The current image is already a translated slide.

Correct ONLY the problematic text/region.

SOURCE TEXT:
{corrected_source}

CORRECT TARGET TEXT:
{corrected_translation}

EDITOR INSTRUCTION:
{instruction}

Rules:
- Change only the relevant text.
- Do not redesign the page.
- Do not alter illustrations, diagrams, people, photographs, colors, background,
  or surrounding text.
- Keep the corrected text in the same location.
- Match the existing typography.
- Preserve line wrapping and alignment as closely as possible.
- Do not modify unrelated translations.

Arabic:
- proper connected Arabic letters;
- correct RTL;
- no missing-glyph boxes.

Finnish:
- preserve ä and ö correctly.

The result should be the same translated slide with ONLY this correction.
""".strip()


def apply_local_correction(
    client: OpenAI,
    current_page: Image.Image,
    source_language: str,
    target_language: str,
    correction_result,
    clicked_box,
    quality: str,
):
    """
    Send only a crop around the clicked region to the image editor.

    The corrected crop is then composited back into the current page.
    This is deliberately cheaper than regenerating the entire slide.
    """

    left, top, right, bottom = clicked_box

    crop = current_page.crop((left, top, right, bottom))

    # Give the image model some context around the clicked area.
    crop_prompt = make_local_correction_prompt(
        source_language,
        target_language,
        correction_result,
    )

    response = client.images.edit(
        model=IMAGE_MODEL,
        image=(
            "correction_region.png",
            image_to_png_bytes(crop),
            "image/png",
        ),
        prompt=crop_prompt,
        size=choose_model_size(crop),
        quality=quality,
        n=1,
        output_format="png",
    )

    corrected_crop = extract_final_image(response)

    if corrected_crop is None:
        raise RuntimeError("Correction image was not returned.")

    corrected_crop = corrected_crop.resize(
        crop.size,
        Image.Resampling.LANCZOS,
    )

    result = current_page.copy()
    result.paste(corrected_crop, (left, top))

    return result


# ============================================================
# PDF creation
# ============================================================

def images_to_pdf(images):
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
# Streamlit state
# ============================================================

if "translated_pages" not in st.session_state:
    st.session_state.translated_pages = {}

if "translation_maps" not in st.session_state:
    st.session_state.translation_maps = {}

if "original_pages" not in st.session_state:
    st.session_state.original_pages = {}

if "feedback_points" not in st.session_state:
    st.session_state.feedback_points = {}

if "correction_results" not in st.session_state:
    st.session_state.correction_results = {}


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Contextual Lecture Translator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Contextual Lecture Translator")

st.caption(
    "English ↔ Arabic ↔ Finnish • contextual translation first, "
    "visual replacement second"
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
    )

    source_language = st.selectbox(
        "Source language",
        LANGUAGES,
        index=0,
    )

    target_options = [
        language for language in LANGUAGES
        if language != source_language
    ]

    target_language = st.selectbox(
        "Target language",
        target_options,
    )

    glossary = st.text_area(
        "Optional glossary",
        placeholder=(
            "Example:\n"
            "taqwa = تقوى\n"
            "magnetic flux density = كثافة الفيض المغناطيسي"
        ),
        height=120,
    )

    st.divider()

    dpi = st.slider(
        "PDF rendering DPI",
        100,
        220,
        150,
        10,
    )

    quality = st.selectbox(
        "Image edit quality",
        ["low", "medium", "high"],
        index=1,
    )

    partial_images = st.slider(
        "Live image previews",
        0,
        3,
        1,
    )

    context_window = st.slider(
        "Neighboring slides used as context",
        0,
        2,
        1,
        help=(
            "1 means previous and next slide are supplied as translation "
            "context. They are not translated."
        ),
    )

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
    st.error(f"Could not open PDF: {exc}")
    st.stop()

page_count = len(doc)

st.success(
    f"Loaded **{uploaded_pdf.name}** — {page_count} pages."
)

col1, col2 = st.columns(2)

with col1:
    start_page = st.number_input(
        "Start page",
        1,
        page_count,
        1,
        1,
    )

with col2:
    end_page = st.number_input(
        "End page",
        1,
        page_count,
        min(page_count, 1),
        1,
    )

if start_page > end_page:
    st.error("Start page must be <= end page.")
    st.stop()

selected_count = end_page - start_page + 1

st.info(
    f"{selected_count} page(s) selected. "
    "For the first test, use 1–3 pages."
)

translate_button = st.button(
    "✨ Translate selected pages",
    type="primary",
)

if translate_button:
    if not api_key.strip():
        st.error("Enter your OpenAI API key.")
        st.stop()

    client = OpenAI(api_key=api_key.strip())

    progress = st.progress(0.0)

    for index, page_number in enumerate(
        range(int(start_page), int(end_page) + 1),
        start=1,
    ):
        page_index = page_number - 1

        status = st.empty()
        timer = st.empty()

        status.info(
            f"Step 1/2 — Understanding and translating page "
            f"{page_number} ({index}/{selected_count})..."
        )

        try:
            source_image = render_pdf_page(
                doc,
                page_index,
                dpi=dpi,
            )

            st.session_state.original_pages[page_number] = source_image

            current_text = extract_page_text(
                doc,
                page_index,
            )

            context_text = build_context(
                doc,
                page_index,
                window=context_window,
            )

            translation_started = time.monotonic()

            translation_map = call_contextual_translation(
                client=client,
                source_language=source_language,
                target_language=target_language,
                current_page_image=source_image,
                context_text=context_text,
                extracted_current_text=current_text,
                glossary=glossary,
            )

            translation_elapsed = (
                time.monotonic() - translation_started
            )

            st.session_state.translation_maps[page_number] = (
                translation_map
            )

            status.success(
                f"Contextual translation ready "
                f"({translation_elapsed:.1f}s)"
            )

            with st.expander(
                f"Translation plan — page {page_number}",
                expanded=False,
            ):
                st.write(
                    translation_map.get(
                        "slide_summary",
                        "",
                    )
                )

                for item in translation_map.get(
                    "translations",
                    [],
                ):
                    st.markdown(
                        f"**{item.get('source', '')}**  →  "
                        f"**{item.get('translation', '')}**"
                    )

            visual_col1, visual_col2 = st.columns(2)

            with visual_col1:
                st.markdown("**Original**")
                st.image(
                    source_image,
                    width="stretch",
                )

            with visual_col2:
                st.markdown("**Translated**")

                preview = st.empty()
                visual_status = st.empty()
                visual_timer = st.empty()

                translated_image = (
                    translate_visual_page_streaming(
                        client=client,
                        source_image=source_image,
                        source_language=source_language,
                        target_language=target_language,
                        translation_map=translation_map,
                        preview_placeholder=preview,
                        status_placeholder=visual_status,
                        timer_placeholder=visual_timer,
                        quality=quality,
                        partial_images=partial_images,
                    )
                )

                st.session_state.translated_pages[
                    page_number
                ] = translated_image

        except Exception as exc:
            status.error(
                f"Page {page_number} failed: {exc}"
            )

        progress.progress(index / selected_count)


# ============================================================
# Review / correction section
# ============================================================

if st.session_state.translated_pages:
    st.divider()
    st.header("🔎 Review and correct translations")

    st.write(
        "If a translation is wrong, click approximately on the problematic "
        "area, describe the problem, and fix only that region."
    )

    for page_number in sorted(
        st.session_state.translated_pages.keys()
    ):
        current_image = st.session_state.translated_pages[
            page_number
        ]

        original_image = st.session_state.original_pages.get(
            page_number
        )

        st.subheader(f"Page {page_number}")

        review_col1, review_col2 = st.columns(2)

        with review_col1:
            st.markdown("**Current translation**")
            st.image(
                current_image,
                width="stretch",
            )

        with review_col2:
            st.markdown("**Correction**")

            # Simple coordinate input is intentionally used here because it
            # works reliably across Streamlit versions without requiring an
            # additional image-coordinate package.
            w, h = current_image.size

            x = st.number_input(
                f"X coordinate — page {page_number}",
                min_value=0,
                max_value=max(0, w - 1),
                value=min(w // 2, max(0, w - 1)),
                key=f"x_{page_number}",
            )

            y = st.number_input(
                f"Y coordinate — page {page_number}",
                min_value=0,
                max_value=max(0, h - 1),
                value=min(h // 2, max(0, h - 1)),
                key=f"y_{page_number}",
            )

            feedback = st.text_area(
                f"What is wrong on page {page_number}?",
                placeholder=(
                    "Example: This translation is too literal. "
                    "'Be mindful of Allah' should be expressed naturally "
                    "in this religious context, not as 'كن واعيًا'."
                ),
                key=f"feedback_{page_number}",
                height=110,
            )

            correction_quality = st.selectbox(
                f"Correction quality — page {page_number}",
                ["low", "medium", "high"],
                index=1,
                key=f"correction_quality_{page_number}",
            )

            if st.button(
                f"🛠️ Fix page {page_number}",
                key=f"fix_{page_number}",
            ):
                if not api_key.strip():
                    st.error("Enter your API key first.")
                    continue

                if not feedback.strip():
                    st.warning("Describe the mistake first.")
                    continue

                client = OpenAI(api_key=api_key.strip())

                # Small local area around the reported point.
                crop, box = crop_feedback_region(
                    current_image,
                    int(x),
                    int(y),
                    radius=max(
                        120,
                        round(min(current_image.size) * 0.12),
                    ),
                )

                st.session_state.feedback_points[
                    page_number
                ] = (int(x), int(y))

                correction_status = st.empty()

                try:
                    correction_status.info(
                        "Understanding your correction..."
                    )

                    context_text = build_context(
                        doc,
                        page_number - 1,
                        window=context_window,
                    )

                    correction_result = correction_translation(
                        client=client,
                        source_language=source_language,
                        target_language=target_language,
                        current_page=current_image,
                        original_page=original_image,
                        context_text=context_text,
                        user_feedback=feedback,
                        glossary=glossary,
                    )

                    st.session_state.correction_results[
                        page_number
                    ] = correction_result

                    correction_status.success(
                        "Correction wording prepared."
                    )

                    with st.expander(
                        "Proposed corrected wording",
                        expanded=True,
                    ):
                        st.write(
                            correction_result.get(
                                "corrected_translation",
                                "",
                            )
                        )

                    correction_status.info(
                        "Applying local visual correction..."
                    )

                    corrected = apply_local_correction(
                        client=client,
                        current_page=current_image,
                        source_language=source_language,
                        target_language=target_language,
                        correction_result=correction_result,
                        clicked_box=box,
                        quality=correction_quality,
                    )

                    st.session_state.translated_pages[
                        page_number
                    ] = corrected

                    correction_status.success(
                        "Page corrected. Review it again."
                    )

                    st.rerun()

                except Exception as exc:
                    correction_status.error(
                        f"Correction failed: {exc}"
                    )


# ============================================================
# Final PDF
# ============================================================

if st.session_state.translated_pages:
    st.divider()
    st.header("📄 Final PDF")

    ordered_pages = sorted(
        st.session_state.translated_pages.keys()
    )

    final_images = [
        st.session_state.translated_pages[p]
        for p in ordered_pages
    ]

    try:
        final_pdf = images_to_pdf(final_images)

        st.download_button(
            "⬇️ Download current translated PDF",
            data=final_pdf,
            file_name=(
                f"{uploaded_pdf.name.rsplit('.', 1)[0]}"
                "_translated.pdf"
            ),
            mime="application/pdf",
        )

        st.caption(
            f"{len(final_images)} translated page(s) currently included."
        )

    except Exception as exc:
        st.error(f"Could not create PDF: {exc}")
