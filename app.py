import io
import base64
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

IMAGE_MODEL = "gpt-image-2"


# ============================================================
# PDF -> IMAGE
# ============================================================

def render_pdf_page(doc, page_number: int, dpi: int = 150) -> Image.Image:
    page = doc.load_page(page_number)
    zoom = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def make_edit_canvas(
    image: Image.Image,
    canvas_size=(1536, 1024),
):
    """
    Put the original page on a canvas accepted by the image model without
    stretching it. Returns:
      canvas, crop_box
    """
    cw, ch = canvas_size
    iw, ih = image.size

    scale = min(cw / iw, ch / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))

    resized = image.resize(
        (nw, nh),
        Image.Resampling.LANCZOS,
    )

    # Use the average edge color instead of black bars.
    edge = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    canvas = Image.new("RGB", (cw, ch), edge)

    x = (cw - nw) // 2
    y = (ch - nh) // 2
    canvas.paste(resized, (x, y))

    crop_box = (x, y, x + nw, y + nh)

    return canvas, crop_box


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

def translate_image(
    client: OpenAI,
    image: Image.Image,
    source_language: str,
    target_language: str,
    glossary: str,
) -> Image.Image:

    canvas, crop_box = make_edit_canvas(image)

    glossary_instruction = ""
    if glossary.strip():
        glossary_instruction = f"""
Use this terminology glossary exactly where applicable:
{glossary.strip()}
"""

    prompt = f"""
EDIT THIS EXISTING IMAGE. DO NOT REDESIGN IT.

The image is a lecture/educational slide.

Translate all visible natural-language text from
{source_language} to {target_language}.

MOST IMPORTANT RULE:
Change ONLY the written language/text.
Preserve the original visual content.

Preserve exactly, as much as possible:
- illustrations
- photographs
- people and characters
- faces
- objects
- diagrams
- charts
- tables
- icons
- borders
- backgrounds
- colors
- lighting
- shadows
- composition
- spacing
- page orientation
- visual style

Do NOT create a new slide.
Do NOT redraw the artwork.
Do NOT simplify the artwork.
Do NOT remove illustrations.
Do NOT add illustrations.
Do NOT change the meaning of diagrams.

TEXT RULES:
- Translate rather than summarize.
- Do not invent text.
- Do not add explanations.
- Do not omit visible text.
- Keep numbers, equations, mathematical symbols and units unchanged unless
  they are ordinary-language words.
- Preserve names and technical terminology appropriately.
- Keep headings as headings.
- Keep labels as labels.
- Keep text inside speech bubbles/signs in the same visual location.
- Preserve approximate font size, weight, alignment and line breaks.
- If the translation is longer, reduce the translated font size or wrap it
  within the original text area rather than moving other objects.
- Do not replace short phrases with explanatory sentences.

ARABIC / RTL:
If the target language is Arabic:
- Use proper Arabic Unicode text.
- Use correct right-to-left direction.
- Connect Arabic letters correctly.
- Do NOT produce boxes or missing-glyph squares.
- Do not transliterate Arabic into Latin characters.
- Example:
  "NO! Cheating is haram" -> "لا! الغش حرام."
- "haram" should be translated as "حرام", not expanded into an explanation.

The final result should look like the ORIGINAL IMAGE made by the same designer,
except that its natural-language text is translated.

{glossary_instruction}
"""

    response = client.images.edit(
        model=IMAGE_MODEL,
        image=(
            "lecture_page.png",
            image_to_png_bytes(canvas),
            "image/png",
        ),
        prompt=prompt,
        size="1536x1024",
        quality="high",
        n=1,
        output_format="png",
    )

    item = response.data[0]

    # GPT Image responses can provide base64 image data.
    if getattr(item, "b64_json", None):
        result_bytes = base64.b64decode(item.b64_json)
    elif getattr(item, "url", None):
        import requests
        r = requests.get(item.url, timeout=120)
        r.raise_for_status()
        result_bytes = r.content
    else:
        raise RuntimeError("OpenAI returned no image data.")

    result = Image.open(
        io.BytesIO(result_bytes)
    ).convert("RGB")

    # Remove the padding we added to make the request 1536x1024.
    result = result.crop(crop_box)

    # Return exactly the original pixel dimensions.
    return result.resize(
        image.size,
        Image.Resampling.LANCZOS,
    )


# ============================================================
# PDF OUTPUT
# ============================================================

def images_to_pdf(images):
    if not images:
        return b""

    buffer = io.BytesIO()

    rgb = [
        img.convert("RGB")
        for img in images
    ]

    rgb[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=rgb[1:],
        resolution=150.0,
    )

    return buffer.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Lecture Translator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Lecture Translator")
st.write(
    "Automatically translate lecture-page images using the OpenAI image "
    "editing model while preserving the original design."
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
        help=(
            "Your key is used by this Streamlit session to call the OpenAI API. "
            "Do not save it in the source code."
        ),
    )

    source_language = st.selectbox(
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

    target_language = st.selectbox(
        "Target language",
        [
            "Arabic",
            "English",
            "Finnish",
            "German",
            "French",
            "Spanish",
            "Swedish",
        ],
    )

    glossary = st.text_area(
        "Optional terminology glossary",
        placeholder=(
            "magnetic flux density = magneettivuon tiheys\n"
            "Taqwa = تقوى"
        ),
        height=120,
    )

    dpi = st.select_slider(
        "PDF rendering quality",
        options=[120, 150, 180, 220],
        value=150,
    )

uploaded = st.file_uploader(
    "Upload lecture PDF",
    type=["pdf"],
)

if uploaded:
    pdf_bytes = uploaded.getvalue()
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    total_pages = len(doc)

    st.success(
        f"Loaded **{uploaded.name}** — {total_pages} pages."
    )

    st.subheader("Choose pages")

    c1, c2 = st.columns(2)

    with c1:
        start_page = st.number_input(
            "Start page",
            min_value=1,
            max_value=total_pages,
            value=1,
        )

    with c2:
        end_page = st.number_input(
            "End page",
            min_value=int(start_page),
            max_value=total_pages,
            value=min(int(start_page) + 2, total_pages),
        )

    st.warning(
        "Image editing costs API credits. Test 1–3 pages first before "
        "processing the complete lecture."
    )

    if st.button(
        "🧪 Translate selected pages",
        type="primary",
    ):
        if not api_key.strip():
            st.error("Enter your OpenAI API key first.")
            st.stop()

        client = OpenAI(
            api_key=api_key.strip()
        )

        page_numbers = list(
            range(
                int(start_page) - 1,
                int(end_page),
            )
        )

        results = []
        originals = []

        progress = st.progress(0.0)
        status = st.empty()

        for i, page_number in enumerate(page_numbers):
            display_page = page_number + 1

            status.info(
                f"Translating page {display_page} / {total_pages}..."
            )

            original = render_pdf_page(
                doc,
                page_number,
                dpi=dpi,
            )

            originals.append(original)

            try:
                translated = translate_image(
                    client=client,
                    image=original,
                    source_language=source_language,
                    target_language=target_language,
                    glossary=glossary,
                )

                results.append(translated)

            except Exception as exc:
                st.error(
                    f"Page {display_page} failed:\n\n{exc}"
                )

                # Keep the original page so the output remains complete.
                results.append(original)

            progress.progress(
                (i + 1) / len(page_numbers)
            )

        status.success(
            "Translation finished."
        )

        # ----------------------------------------------------
        # Preview
        # ----------------------------------------------------

        st.subheader("Preview")

        for i, page_number in enumerate(page_numbers):
            with st.expander(
                f"Page {page_number + 1}",
                expanded=(i == 0),
            ):
                left, right = st.columns(2)

                with left:
                    st.markdown("**Original**")
                    st.image(
                        originals[i],
                        use_container_width=True,
                    )

                with right:
                    st.markdown("**Translated**")
                    st.image(
                        results[i],
                        use_container_width=True,
                    )

        # ----------------------------------------------------
        # Download selected pages
        # ----------------------------------------------------

        output_pdf = images_to_pdf(results)

        st.download_button(
            "📥 Download translated PDF",
            data=output_pdf,
            file_name=(
                f"translated_{Path(uploaded.name).stem}.pdf"
            ),
            mime="application/pdf",
        )
