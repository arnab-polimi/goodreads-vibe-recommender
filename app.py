import html
import re
import streamlit as st
import pandas as pd
from recommender import VibeRecommender

st.set_page_config(
    page_title="VibeRead | Goodreads Vibe Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for an atmospheric, literary UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=Inter:wght@400;500;600&display=swap');

    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #e2d1c3 0%, #c9a98d 50%, #a37b5c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .book-card {
        border: 1px solid rgba(201, 169, 141, 0.28);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        background: rgba(25, 23, 28, 0.55);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.18);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .book-card:hover {
        border-color: rgba(201, 169, 141, 0.65);
    }
    .top-match-card {
        border: 2px solid #c9a98d;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
        background: linear-gradient(135deg, rgba(42, 34, 46, 0.82) 0%, rgba(24, 22, 28, 0.9) 100%);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .book-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0;
        color: #f4ece4;
    }
    .book-author {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #c9a98d;
        margin-top: 0.2rem;
        margin-bottom: 0.7rem;
        font-weight: 500;
    }
    .badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        margin: 0.18rem 0.25rem 0.18rem 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(201, 169, 141, 0.16);
        color: #e6d3c1;
        border: 1px solid rgba(201, 169, 141, 0.35);
    }
    .vibe-hit-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        margin: 0.18rem 0.25rem 0.18rem 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(142, 192, 124, 0.18);
        color: #b8e2a8;
        border: 1px solid rgba(142, 192, 124, 0.45);
    }
    .score-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        background: rgba(201, 169, 141, 0.22);
        color: #f3e5d8;
        border: 1px solid #c9a98d;
    }
    .desc-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #dcd6d0;
        margin-top: 0.75rem;
    }
    mark.vibe-mark {
        background-color: rgba(201, 169, 141, 0.32);
        color: #fff6ed;
        padding: 0.05rem 0.25rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Goodreads catalog & neural vibe embeddings...")
def load_engine() -> VibeRecommender:
    return VibeRecommender()


def highlight_description(text: str, matched_vibes: list[str], max_chars: int = 550) -> str:
    snippet = text.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    escaped = html.escape(snippet)
    for term in matched_vibes:
        if len(term) >= 3:
            pattern = re.compile(r"\b(" + re.escape(term) + r"\w*)\b", re.IGNORECASE)
            escaped = pattern.sub(r'<mark class="vibe-mark">\1</mark>', escaped)
    return escaped


def render_book_card(row: pd.Series, rank: int, is_top: bool = False):
    card_class = "top-match-card" if is_top else "book-card"
    title_html = html.escape(str(row["Book"]))
    author_html = html.escape(str(row["Author"]))
    rating = float(row["Avg_Rating"])
    num_ratings = f"{int(row['Num_Ratings_Int']):,}"
    match_pct = min(int(round(float(row["Match_Score"]) * 100)), 99)
    sem_pct = int(round(float(row["Semantic_Score"]) * 100))
    lex_pct = int(round(float(row["Lexical_Score"]) * 100))
    url = str(row.get("URL", "#"))

    genres = row.get("Genres_List", [])[:7]
    genre_badges = "".join(f'<span class="badge">{html.escape(g)}</span>' for g in genres)

    matched_vibes = row.get("Matched_Vibes", [])
    vibe_badges = ""
    if matched_vibes:
        vibe_badges = (
            '<div style="margin-top:0.45rem;">'
            '<span style="font-size:0.8rem; opacity:0.8; margin-right:0.4rem;">✨ Matched Vibe Elements:</span>'
            + "".join(f'<span class="vibe-hit-badge">{html.escape(v)}</span>' for v in matched_vibes)
            + "</div>"
        )

    desc_html = highlight_description(
        str(row["Description"]),
        matched_vibes,
        max_chars=750 if is_top else 460,
    )

    top_banner = (
        '<div style="font-size:0.82rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#c9a98d; margin-bottom:0.35rem;">'
        '🏆 Best Matched Book for Your Vibe</div>'
        if is_top
        else ""
    )

    card_html = (
        f'<div class="{card_class}">'
        f'{top_banner}'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;">'
        f'<div>'
        f'<div class="book-title">#{rank}. {title_html}</div>'
        f'<div class="book-author">by {author_html} &nbsp;•&nbsp; ⭐ {rating:.2f} ({num_ratings} Goodreads ratings)</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<span class="score-pill">🎯 {match_pct}% Vibe Match</span>'
        f'<div style="font-size:0.74rem; opacity:0.75; margin-top:0.25rem;">'
        f'Semantic: {sem_pct}% | Keywords: {lex_pct}%'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:0.35rem;">{genre_badges}</div>'
        f'{vibe_badges}'
        f'<div class="desc-text">{desc_html}</div>'
        f'<div style="margin-top:0.85rem;">'
        f'<a href="{html.escape(url)}" target="_blank" style="color:#c9a98d; text-decoration:none; font-size:0.86rem; font-weight:600;">'
        f'↗ View on Goodreads'
        f'</a>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def main():
    engine = load_engine()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Vibe Engine Settings")
        if engine.has_aligned_embeddings:
            st.caption(f"✅ Indexed **{len(engine.df):,}** Goodreads books with full neural + lexical vectors.")
        else:
            st.caption(f"⚡ Indexed **{len(engine.df):,}** Goodreads books (Hybrid TF-IDF + On-the-fly Neural Reranking).")
            with st.expander("📥 Import Colab GPU Embeddings"):
                st.caption("Drop `goodreads_index_files.zip` or `book_embeddings.npy` from Colab into `data/` or upload below:")
                uploaded = st.file_uploader("Upload .zip or .npy", type=["zip", "npy"])
                if uploaded is not None:
                    import os, zipfile
                    from recommender import DATA_DIR, EMBEDDINGS_PATH
                    os.makedirs(DATA_DIR, exist_ok=True)
                    if uploaded.name.endswith(".zip"):
                        with zipfile.ZipFile(uploaded, "r") as zf:
                            zf.extractall(DATA_DIR)
                    else:
                        with open(EMBEDDINGS_PATH, "wb") as f:
                            f.write(uploaded.getbuffer())
                    st.cache_resource.clear()
                    st.success("Embeddings imported! Reloading engine...")
                    st.rerun()

        top_k = st.slider("Number of recommendations", min_value=1, max_value=25, value=8, step=1)
        min_rating = st.slider("Minimum Goodreads Rating ⭐", min_value=0.0, max_value=4.5, value=3.5, step=0.1)

        st.subheader("🎛️ Hybrid Ranking Weights")
        semantic_w = st.slider(
            "Neural Semantic Vibe (MiniLM)",
            0.0, 1.0, 0.65, 0.05,
            help="Matches deep mood, atmosphere, and conceptual similarity using transformer embeddings."
        )
        lexical_w = st.slider(
            "Exact Vibe Keywords (TF-IDF)",
            0.0, 1.0, 0.25, 0.05,
            help="Matches exact vibe descriptors, themes, and genre tags."
        )
        quality_w = st.slider(
            "Goodreads Rating & Popularity",
            0.0, 0.5, 0.10, 0.05,
            help="Slightly boosts higher-rated and well-loved books on Goodreads."
        )

        st.subheader("🏷️ Genre Filters (Optional)")
        required_genres = st.multiselect(
            "Must include at least one of:",
            options=engine.popular_genres,
            default=[],
        )
        excluded_genres = st.multiselect(
            "Exclude books with genres:",
            options=engine.popular_genres,
            default=[],
        )

    # Main Content
    st.markdown('<div class="main-header">📚 VibeRead: Goodreads Vibe Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Describe an atmosphere, aesthetic, mood, or feeling — and discover the Goodreads books that best match your vibe.</div>',
        unsafe_allow_html=True,
    )

    # Preset Vibe Chips
    preset_vibes = [
        "gothic, horror, mysterious and lonely",
        "dark academia, secret society, obsessive, autumn rain",
        "cozy, magical bakery, warm tea, heartwarming and gentle",
        "cyberpunk, neon rain, lonely hacker, existential sci-fi",
        "witchy, folklore, enchanted forest, eerie fairy tale",
        "Psychological thriller, unreliable narrator, claustrophobic, twisted",
    ]

    if "vibe_input" not in st.session_state:
        st.session_state.vibe_input = preset_vibes[0]

    st.caption("✨ **Try an example vibe preset:**")
    cols = st.columns(3)
    for idx, preset in enumerate(preset_vibes):
        if cols[idx % 3].button(f"“{preset}”", key=f"preset_{idx}", use_container_width=True):
            st.session_state.vibe_input = preset

    # Search input
    query = st.text_input(
        "Enter your desired book vibe, mood, or atmosphere:",
        value=st.session_state.vibe_input,
        placeholder="e.g., gothic, horror, mysterious and lonely...",
    )

    tab1, tab2 = st.tabs(["🔮 Match by Vibe", "📖 Find Similar by Book Title"])

    with tab1:
        if query.strip():
            results = engine.recommend(
                vibe_query=query,
                top_k=top_k,
                semantic_weight=semantic_w,
                lexical_weight=lexical_w,
                quality_weight=quality_w,
                min_rating=min_rating,
                required_genres=required_genres,
                excluded_genres=excluded_genres,
            )

            if results.empty:
                st.warning("No books matched your vibe and filter criteria. Try relaxing your genre or rating filters!")
            else:
                # Render #1 Best Match prominently
                render_book_card(results.iloc[0], rank=1, is_top=True)

                if len(results) > 1:
                    st.markdown(f"### 🕯️ More Books Matching *“{html.escape(query)}”*")
                    for i in range(1, len(results)):
                        render_book_card(results.iloc[i], rank=i + 1, is_top=False)
        else:
            st.info("Type any vibe above to find your next read!")

    with tab2:
        st.markdown("#### Find Books with the Same Atmospheric DNA")
        selected_book = st.selectbox(
            "Choose a book from the Goodreads catalog to find its closest vibe twins:",
            options=engine.df["Book"].head(2500).tolist(),
            index=0,
        )
        if selected_book:
            sim_df = engine.find_similar_books(selected_book, top_k=6)
            if not sim_df.empty:
                for i, row in sim_df.iterrows():
                    row["Semantic_Score"] = row["Match_Score"]
                    row["Lexical_Score"] = row["Match_Score"]
                    row["Matched_Vibes"] = row.get("Genres_List", [])[:4]
                    render_book_card(row, rank=i + 1, is_top=(i == 0))


if __name__ == "__main__":
    main()
