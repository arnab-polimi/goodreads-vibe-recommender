# Copyright (c) 2026 Arnab Mondal. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying or distribution is prohibited.

import html
import os
import re
import time
import zipfile
import streamlit as st
import pandas as pd
from recommender import VibeRecommender, DATA_DIR, EMBEDDINGS_PATH

st.set_page_config(
    page_title="Index — Book Recommendations by Atmosphere",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Hide Streamlit chrome, deploy button, and viewer badges while keeping the sidebar toggle accessible */
    #MainMenu, footer,
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .stDeployButton,
    .stAppDeployButton,
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137 {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        max-width: 760px !important;
        padding-top: 3.5rem !important;
        padding-bottom: 5rem !important;
    }

    /* Editorial Masthead */
    .masthead {
        border-bottom: 1px solid #2A2926;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .masthead-kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7D7972;
        margin-bottom: 0.5rem;
    }
    .masthead-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.35rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        color: #EAE6DF;
        line-height: 1.15;
        margin: 0;
    }
    .masthead-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        font-weight: 300;
        color: #8C8881;
        margin-top: 0.55rem;
    }

    /* Input styling overrides */
    .stTextInput > label,
    .stRadio > label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #7D7972 !important;
    }
    .stTextInput input {
        font-family: 'Newsreader', Georgia, serif !important;
        font-size: 1.2rem !important;
        background-color: #161614 !important;
        border: 1px solid #2A2926 !important;
        border-radius: 2px !important;
        color: #EAE6DF !important;
        padding: 0.65rem 0.8rem !important;
    }
    .stTextInput input:focus {
        border-color: #6E6A63 !important;
        box-shadow: none !important;
    }

    /* Minimalist Book Entry (Catalog Style) */
    .entry {
        padding: 1.65rem 0;
        border-bottom: 1px solid #22211F;
    }
    .entry-first {
        padding-top: 0.8rem;
    }
    .entry-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
    }
    .entry-idx {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #6E6A63;
        margin-right: 0.65rem;
    }
    .entry-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.38rem;
        font-weight: 400;
        color: #EAE6DF;
        line-height: 1.25;
    }
    .entry-title a {
        color: #EAE6DF;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 0.15s ease;
    }
    .entry-title a:hover {
        border-bottom: 1px solid #8C8881;
    }
    .entry-author {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #9E9A92;
        margin-top: 0.2rem;
    }
    .entry-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.73rem;
        color: #7D7972;
        white-space: nowrap;
        text-align: right;
    }
    .entry-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.91rem;
        font-weight: 300;
        line-height: 1.68;
        color: #C2BEB6;
        margin-top: 0.85rem;
    }
    .entry-footer {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-top: 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.71rem;
        color: #6E6A63;
        letter-spacing: 0.02em;
    }
    .vibe-underline {
        color: #EAE6DF;
        border-bottom: 1px solid #5A5751;
    }

    /* Subtle text-only preset & action buttons */
    .stButton > button,
    div[data-testid="stHorizontalBlock"] button {
        background: transparent !important;
        border: 1px solid #242320 !important;
        border-radius: 2px !important;
        color: #8C8881 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 400 !important;
        padding: 0.25rem 0.55rem !important;
        transition: color 0.15s ease, border-color 0.15s ease !important;
    }
    .stButton > button:hover,
    div[data-testid="stHorizontalBlock"] button:hover {
        color: #EAE6DF !important;
        border-color: #4A4843 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.8rem;
        border-bottom: 1px solid #22211F;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.5rem 0;
        color: #7D7972;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> VibeRecommender:
    return VibeRecommender()


def format_synopsis(text: str, matched_vibes: list[str], max_chars: int = 520) -> str:
    snippet = text.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    escaped = html.escape(snippet)
    for term in matched_vibes:
        if len(term) >= 4:
            pattern = re.compile(r"\b(" + re.escape(term) + r"\w*)\b", re.IGNORECASE)
            escaped = pattern.sub(r'<span class="vibe-underline">\1</span>', escaped)
    return escaped


def render_entry(row: pd.Series, rank: int):
    title_html = html.escape(str(row["Book"]))
    author_html = html.escape(str(row["Author"]))
    rating = float(row["Avg_Rating"])
    num_ratings = f"{int(row['Num_Ratings_Int']):,}"
    score = float(row["Match_Score"])
    url = html.escape(str(row.get("URL", "#")))

    genres = [html.escape(g) for g in row.get("Genres_List", [])[:4]]
    genres_str = " · ".join(genres) if genres else "Fiction"

    matched_vibes = row.get("Matched_Vibes", [])
    desc_html = format_synopsis(str(row["Description"]), matched_vibes, max_chars=600 if rank == 1 else 440)

    first_cls = " entry-first" if rank == 1 else ""
    idx_str = f"{rank:02d}"

    entry_html = (
        f'<div class="entry{first_cls}">'
        f'<div class="entry-header">'
        f'<div>'
        f'<span class="entry-idx">{idx_str}</span>'
        f'<span class="entry-title"><a href="{url}" target="_blank">{title_html}</a></span>'
        f'<div class="entry-author">{author_html}</div>'
        f'</div>'
        f'<div class="entry-meta">'
        f'{rating:.2f} ★ ({num_ratings}) &nbsp;·&nbsp; {score:.2f}'
        f'</div>'
        f'</div>'
        f'<div class="entry-desc">{desc_html}</div>'
        f'<div class="entry-footer">'
        f'<span>{genres_str}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(entry_html, unsafe_allow_html=True)


def check_rate_limit(max_requests: int = 25, window_seconds: int = 60) -> bool:
    now = time.time()
    if "req_timestamps" not in st.session_state:
        st.session_state.req_timestamps = []
    st.session_state.req_timestamps = [
        t for t in st.session_state.req_timestamps if now - t < window_seconds
    ]
    if len(st.session_state.req_timestamps) >= max_requests:
        return False
    st.session_state.req_timestamps.append(now)
    return True


def main():
    engine = load_engine()

    # Minimal Sidebar for filters & index management
    with st.sidebar:
        st.markdown("##### Parameters")
        top_k = st.slider("Results", min_value=3, max_value=25, value=8, step=1)
        min_rating = st.slider("Minimum rating", min_value=0.0, max_value=4.5, value=3.5, step=0.1)
        min_ratings_count = st.number_input(
            "Minimum rating count",
            min_value=0,
            max_value=50000,
            value=500,
            step=50,
            help="Set to 0 to include obscure or indie volumes across all 1.14M books.",
        )

        st.markdown("##### Ranking Balance")
        semantic_w = st.slider("Semantic (Neural)", 0.0, 1.0, 0.65, 0.05)
        lexical_w = st.slider("Lexical (TF-IDF)", 0.0, 1.0, 0.25, 0.05)
        quality_w = st.slider("Rating weight", 0.0, 0.5, 0.10, 0.05)

        st.markdown("##### Shelf Filter")
        required_genres = st.multiselect("Include", options=engine.popular_genres, default=[])
        excluded_genres = st.multiselect("Exclude", options=engine.popular_genres, default=[])

        st.markdown("---")
        st.caption(f"Catalog: {len(engine.df):,} volumes")

    # Masthead
    st.markdown(
        f"""
        <div class="masthead">
            <div class="masthead-kicker">Goodreads Catalog Index · {len(engine.df):,} Volumes</div>
            <h1 class="masthead-title">Books by Atmosphere</h1>
            <div class="masthead-sub">Describe a mood, setting, or literary tone to search the index.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_vibe, tab_similar = st.tabs(["Atmosphere Search", "Similar by Title"])

    with tab_vibe:
        presets = [
            "gothic, horror, mysterious and lonely",
            "dark academia, secret society, autumn rain",
            "isolated winter outpost, quiet dread",
            "melancholic, coastal town, memory and loss",
        ]

        if "vibe_input" not in st.session_state:
            st.session_state.vibe_input = ""
        if "vibe_offset" not in st.session_state:
            st.session_state.vibe_offset = 0
        if "last_vibe_query" not in st.session_state:
            st.session_state.last_vibe_query = st.session_state.vibe_input

        def set_preset(val: str):
            st.session_state.vibe_input = val
            st.session_state.vibe_offset = 0
            st.session_state.last_vibe_query = val

        def refresh_vibe():
            st.session_state.vibe_offset += top_k

        def reset_vibe_offset():
            st.session_state.vibe_offset = 0

        col_q, col_mode = st.columns([3, 1.35])
        with col_q:
            query = st.text_input(
                "Atmosphere / Tone",
                key="vibe_input",
                placeholder="e.g. gothic, horror, mysterious and lonely",
            )
        with col_mode:
            category_mode = st.radio(
                "Mode",
                options=["Fiction", "Nonfiction", "All"],
                index=0,
                horizontal=True,
                key="category_mode",
            )

        if "last_category_mode" not in st.session_state:
            st.session_state.last_category_mode = category_mode
        if query != st.session_state.last_vibe_query or category_mode != st.session_state.last_category_mode:
            st.session_state.vibe_offset = 0
            st.session_state.last_vibe_query = query
            st.session_state.last_category_mode = category_mode

        cols = st.columns(len(presets))
        for i, p in enumerate(presets):
            label = p.split(",")[0] + ", " + p.split(",")[1].strip()
            cols[i].button(
                label,
                key=f"p_{i}",
                on_click=set_preset,
                args=(p,),
                use_container_width=True,
            )

        if query.strip():
            if not check_rate_limit():
                st.caption("Rate limit reached. Please wait a moment before searching again.")
                return
            current_offset = st.session_state.vibe_offset
            eff_required = list(required_genres)
            eff_excluded = list(excluded_genres)
            if category_mode == "Fiction" and "Nonfiction" not in eff_excluded:
                eff_excluded.append("Nonfiction")
            elif category_mode == "Nonfiction" and "Nonfiction" not in eff_required:
                eff_required.append("Nonfiction")

            t0 = time.time()
            results = engine.recommend(
                vibe_query=query,
                top_k=top_k,
                offset=current_offset,
                semantic_weight=semantic_w,
                lexical_weight=lexical_w,
                quality_weight=quality_w,
                min_rating=min_rating,
                min_ratings_count=int(min_ratings_count),
                required_genres=eff_required,
                excluded_genres=eff_excluded,
            )
            elapsed = time.time() - t0

            if results.empty:
                st.caption("No matching volumes found for the current filter parameters.")
            else:
                start_rank = current_offset + 1
                end_rank = current_offset + len(results)
                hdr_left, hdr_right = st.columns([3, 2])
                with hdr_left:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.71rem; color:#6E6A63; margin-top:0.9rem; padding-bottom:0.4rem;">'
                        f'Showing {start_rank}–{end_rank} of {len(engine.df):,} volumes &nbsp;·&nbsp; {elapsed:.2f}s'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with hdr_right:
                    if current_offset > 0:
                        b1, b2 = st.columns(2)
                        b1.button(
                            "↺ Top",
                            key="vibe_reset_top",
                            on_click=reset_vibe_offset,
                            use_container_width=True,
                        )
                        b2.button(
                            "↻ Refresh",
                            key="vibe_refresh_top",
                            on_click=refresh_vibe,
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            "↻ Refresh suggestions",
                            key="vibe_refresh_top",
                            on_click=refresh_vibe,
                            use_container_width=True,
                        )
                st.markdown(
                    '<div style="border-bottom:1px solid #22211F; margin-top:-0.35rem;"></div>',
                    unsafe_allow_html=True,
                )
                for i in range(len(results)):
                    render_entry(results.iloc[i], rank=current_offset + i + 1)

                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                if current_offset > 0:
                    fb1, fb2 = st.columns([1, 2])
                    fb1.button(
                        "↺ Back to top matches",
                        key="vibe_reset_btm",
                        on_click=reset_vibe_offset,
                        use_container_width=True,
                    )
                    fb2.button(
                        f"↻ Refresh suggestions (next {top_k})",
                        key="vibe_refresh_btm",
                        on_click=refresh_vibe,
                        use_container_width=True,
                    )
                else:
                    st.button(
                        f"↻ Refresh suggestions (next {top_k})",
                        key="vibe_refresh_btm",
                        on_click=refresh_vibe,
                        use_container_width=True,
                    )

    with tab_similar:
        if "sim_offset" not in st.session_state:
            st.session_state.sim_offset = 0
        if "last_sim_title" not in st.session_state:
            st.session_state.last_sim_title = ""

        def refresh_sim():
            st.session_state.sim_offset += top_k

        def reset_sim_offset():
            st.session_state.sim_offset = 0

        col_sel, col_custom = st.columns([3, 2])
        with col_sel:
            selected_book = st.selectbox(
                "Popular Volumes",
                options=engine.popular_titles,
                index=0,
            )
        with col_custom:
            custom_book = st.text_input(
                "Or search any title in the 1.14M catalog",
                value="",
                placeholder="Type exact or partial title...",
            )
        target_title = custom_book.strip() if custom_book.strip() else selected_book
        if target_title != st.session_state.last_sim_title:
            st.session_state.sim_offset = 0
            st.session_state.last_sim_title = target_title

        if target_title:
            sim_offset = st.session_state.sim_offset
            sim_df = engine.find_similar_books(
                target_title,
                top_k=top_k,
                offset=sim_offset,
                min_ratings_count=int(min_ratings_count),
            )
            if sim_df.empty:
                st.caption("No matching volume found in the catalog.")
            else:
                sh_left, sh_right = st.columns([3, 2])
                with sh_left:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.71rem; color:#6E6A63; margin-top:0.5rem; padding-bottom:0.4rem;">'
                        f'Showing {sim_offset + 1}–{sim_offset + len(sim_df)} similar volumes'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with sh_right:
                    if sim_offset > 0:
                        sb1, sb2 = st.columns(2)
                        sb1.button(
                            "↺ Top",
                            key="sim_reset_top",
                            on_click=reset_sim_offset,
                            use_container_width=True,
                        )
                        sb2.button(
                            "↻ Refresh",
                            key="sim_refresh_top",
                            on_click=refresh_sim,
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            "↻ Refresh suggestions",
                            key="sim_refresh_top",
                            on_click=refresh_sim,
                            use_container_width=True,
                        )
                st.markdown(
                    '<div style="border-bottom:1px solid #22211F; margin-top:-0.35rem;"></div>',
                    unsafe_allow_html=True,
                )
                for i, row in sim_df.iterrows():
                    row["Matched_Vibes"] = []
                    render_entry(row, rank=sim_offset + i + 1)

                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                st.button(
                    f"↻ Refresh suggestions (next {top_k})",
                    key="sim_refresh_btm",
                    on_click=refresh_sim,
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
