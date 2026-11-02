"""Streamlit web UI.  Run with:  streamlit run app.py"""
import streamlit as st

from src.recommender import (
    MODEL_PATH,
    get_genres,
    get_titles,
    recommend,
    recommend_by_genre,
)
from src.utils import fetch_poster

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.caption("Similar-movie search (TF-IDF + cosine similarity) and top picks by genre")

if not MODEL_PATH.exists():
    st.error("Model not built yet. In the terminal run:")
    st.code("python -m src.preprocess\npython -m src.recommender")
    st.stop()


def show(results, extra):
    """Display recommendation cards in rows of 5."""
    for start in range(0, len(results), 5):
        cols = st.columns(5)
        for col, (_, row) in zip(cols, results.iloc[start:start + 5].iterrows()):
            with col:
                poster = fetch_poster(row["id"])
                if poster:
                    st.image(poster)
                st.subheader(row["title"])
                st.write(row["genres"])
                st.write(f"⭐ {row['vote_average']}  |  {extra(row)}")


tab_similar, tab_genre = st.tabs(["🎯 Similar to a movie", "🏷️ Top picks by genre"])

# ---- Tab 1: content-based, similar to a chosen movie ----
with tab_similar:
    selected = st.selectbox("Pick a movie you like:", get_titles(), key="movie")
    count = st.slider("Number of recommendations", 3, 10, 5, key="count_similar")

    if st.button("Recommend", key="btn_similar"):
        results = recommend(selected, n=count)
        show(results, lambda r: f"Match: {r['similarity']:.0%}")

# ---- Tab 2: genre-based, best movies in chosen genre(s) ----
with tab_genre:
    chosen = st.multiselect(
        "Choose one or more genres (movies must match all of them):",
        get_genres(),
        key="genres",
    )
    count_genre = st.slider("Number of recommendations", 5, 20, 10, key="count_genre")

    if st.button("Show top movies", key="btn_genre"):
        if not chosen:
            st.warning("Pick at least one genre.")
        else:
            try:
                results = recommend_by_genre(chosen, n=count_genre)
                show(results, lambda r: f"Score: {r['score']}")
            except (ValueError, RuntimeError) as err:
                st.error(str(err))