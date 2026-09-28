# 📚 Lecture Slide Translator (Streamlit + OpenAI Vision)

A streamlined web tool to translate academic and engineering slide presentations page-by-page while preserving diagrams, graphs, mathematical equations, and background layout.

## Features
- **Bring Your Own Key (BYOK):** Zero cost to the host; users provide their own OpenAI API key in ephemeral browser memory.
- **Selective Inpainting:** Only replaces text bounding boxes detected by `gpt-4o` Structured Outputs, ensuring circuit diagrams, finite element plots, and formulas remain unaltered.
- **RTL & Complex Script Ready:** Supports Arabic and other bidirectional scripts via `arabic-reshaper` and `python-bidi`.
- **Custom Glossaries:** Enforce domain-specific technical terminology across every slide.

## Quickstart (Local)

1. Clone repository:
   ```bash
   git clone [https://github.com/your-username/lecture-translator.git](https://github.com/your-username/lecture-translator.git)
   cd lecture-translator