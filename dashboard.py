"""Minimal Streamlit dashboard for MultiScope (Upload + Query only)."""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st

from app.image_encoder import encode_image
from app.text_encoder import encode_text
from app.similarity import compute_similarity
from app.utils import parse_text_queries


def _inject_styles() -> None:
    """Inject pastel theme and minimal card styling."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #F7F9FC;
                --card: #FFFFFF;
                --primary: #6C8CF5;
                --accent: #A7C7E7;
                --highlight: #FFD6E0;
                --text: #2E2E2E;
                --muted: #6B7280;
                --border: #E7ECF5;
            }

            .stApp { background: var(--bg); color: var(--text); }

            .block-container {
                max-width: 1080px;
                padding-top: 1.8rem;
                padding-bottom: 2.2rem;
            }

            .ms-hero {
                background: linear-gradient(135deg, #FFFFFF 0%, #F1F5FF 55%, #FDF2F7 100%);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 34px 28px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                margin: 0.4rem 0 1.2rem 0;
            }

            .ms-hero-title {
                font-size: clamp(2.6rem, 4vw, 3.4rem);
                font-weight: 800;
                letter-spacing: -0.03em;
                color: var(--text);
                margin: 0;
                line-height: 1.06;
            }

            .ms-hero-subtitle {
                margin-top: 0.55rem;
                font-size: 1.02rem;
                font-weight: 500;
                color: var(--muted);
            }

            .ms-hero-divider {
                width: 88px;
                height: 4px;
                border-radius: 999px;
                margin: 0.95rem auto 0 auto;
                background: linear-gradient(90deg, var(--primary), var(--accent), var(--highlight));
            }

            .ms-card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 6px 18px rgba(108, 140, 245, 0.08);
            }

            .ms-caption {
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: 0.35rem;
                text-align: center;
            }

            .ms-empty {
                border: 1px dashed #C8D2E8;
                background: #FFFFFF;
                border-radius: 12px;
                padding: 0.9rem;
                color: var(--muted);
            }

            .ms-result-card {
                border-radius: 14px;
                padding: 0.8rem 0.9rem;
                margin-bottom: 0.65rem;
                border: 1px solid rgba(46, 46, 46, 0.06);
            }

            .ms-result-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
            }

            .ms-score {
                min-width: 78px;
                text-align: center;
                font-weight: 700;
                border-radius: 999px;
                padding: 0.25rem 0.6rem;
                font-size: 0.88rem;
                background: #EEF2FF;
                color: #25316D;
            }

            .ms-query {
                flex: 1;
                font-size: 0.95rem;
                color: var(--text);
                line-height: 1.35;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_upload_to_temp(uploaded_file) -> Path:
    """Persist uploaded image to a temp file and return path."""
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _score_to_percent(score: float) -> float:
    """Map cosine score [-1, 1] to display percentage [0, 100]."""
    normalized = (float(score) + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))
    return normalized * 100.0


def _score_level(percent: float) -> Tuple[str, str]:
    """Return qualitative label and color for score."""
    if percent >= 75:
        return "High", "#22C55E"
    if percent >= 45:
        return "Medium", "#3B82F6"
    return "Low", "#EF4444"


def _result_bg(percent: float) -> str:
    """Return pastel gradient background based on score."""
    if percent >= 75:
        return "linear-gradient(90deg, #ECFDF3 0%, #FFFFFF 100%)"
    if percent >= 45:
        return "linear-gradient(90deg, #EEF4FF 0%, #FFFFFF 100%)"
    return "linear-gradient(90deg, #FFF1F2 0%, #FFFFFF 100%)"


def _render_result_card(rank: int, query: str, score: float) -> None:
    """Render one result row with aligned metadata and progress bar."""
    percent = _score_to_percent(score)
    safe_query = query.replace("<", "&lt;").replace(">", "&gt;")

    if percent >= 75:
        confidence = "High"
        badge_bg = "#E8F8EE"
        badge_color = "#1F8A4C"
        bar_color = "#6FCF97"
    elif percent >= 50:
        confidence = "Medium"
        badge_bg = "#EAF0FF"
        badge_color = "#3559C7"
        bar_color = "#6C8CF5"
    else:
        confidence = "Low"
        badge_bg = "#FFEDED"
        badge_color = "#B63B3B"
        bar_color = "#FF8A8A"

    st.markdown(
        f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:8px;">
                <div style="font-size:15px; color:#2E2E2E; font-weight:600; min-width:0;">
                    {rank}. {safe_query}
                </div>
                <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
                    <span style="font-size:14px; font-weight:700; color:#2E2E2E;">{percent:.1f}%</span>
                    <span style="padding:4px 10px; border-radius:12px; font-size:12px; font-weight:700; background:{badge_bg}; color:{badge_color};">
                        {confidence}
                    </span>
                </div>
            </div>
            <div style="width:100%; height:10px; background:#E5E7EB; border-radius:10px; overflow:hidden;">
                <div style="width:{percent:.2f}%; height:100%; background:{bar_color}; border-radius:10px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render minimal dashboard."""
    st.set_page_config(page_title="MultiScope", page_icon="🔍", layout="wide")
    _inject_styles()

    # Strong top hero
    st.markdown(
        """
        <section class="ms-hero">
            <h1 class="ms-hero-title">🔍 MultiScope</h1>
            <div class="ms-hero-subtitle">Multimodal Image–Text Retrieval with CLIP</div>
            <div class="ms-hero-divider"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Main two-column layout
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.container():
            st.markdown('<div class="ms-card">', unsafe_allow_html=True)
            st.subheader("Image")
            uploaded_image = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )

            if uploaded_image:
                st.image(uploaded_image, width="stretch")
                st.markdown('<div class="ms-caption">Uploaded Image</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ms-empty">Upload an image to begin.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        with st.container():
            st.markdown('<div class="ms-card">', unsafe_allow_html=True)
            st.subheader("Text Queries")
            text_input = st.text_area(
                "Enter one query per line",
                height=220,
                placeholder="a person working on a laptop\na cat on a sofa\nsomeone coding indoors",
                label_visibility="collapsed",
            )
            run = st.button("Compute Similarity", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Compute and show ranked results
    if run:
        if not uploaded_image:
            st.error("Please upload a valid image.")
            return

        queries: List[str] = parse_text_queries(text_input)
        if not queries:
            st.warning("Please enter at least one text query.")
            return

        temp_path = None
        try:
            with st.spinner("Computing similarities..."):
                temp_path = _save_upload_to_temp(uploaded_image)
                image_embedding = encode_image(temp_path)
                text_embeddings = encode_text(queries)
                similarities = compute_similarity(image_embedding, text_embeddings)

            ranked = sorted(
                zip(queries, np.atleast_1d(similarities).tolist()),
                key=lambda x: float(x[1]),
                reverse=True,
            )

            st.divider()
            st.subheader("Results")
            for idx, (query, score) in enumerate(ranked, start=1):
                _render_result_card(idx, query, float(score))

        except Exception as e:
            st.error(f"Could not process input. Details: {e}")
        finally:
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
