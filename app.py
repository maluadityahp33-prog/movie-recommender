"""CineMatch - Movie Recommendation System (Streamlit UI).

Run with:  python -m streamlit run app.py
"""
import html
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import (
    MODEL_PATH,
    get_genres,
    get_titles,
    get_year_range,
    load_model,
    recommend,
    recommend_by_genre,
)
from src.utils import fetch_poster

st.set_page_config(page_title="CineMatch - Movie Recommender", page_icon="🎬", layout="wide")


def inject_css(css):
    """Add global CSS. st.html wraps a CSS file path in <style> tags (documented behaviour)."""
    path = Path(tempfile.gettempdir()) / "cinematch_style.css"
    try:
        path.write_text(css, encoding="utf-8")
        st.html(path)
    except Exception:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html_block(markup):
    """Render raw HTML: st.html on current Streamlit, markdown on very old versions."""
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', 'Segoe UI', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 600px at 8% -10%, rgba(123,47,247,.28), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(229,9,20,.25), transparent 60%),
    #0b0b14;
  color: #ececf4;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1350px; }

.hero {
  padding: 30px 34px; border-radius: 22px; margin-bottom: 18px;
  background: linear-gradient(120deg, #e50914 0%, #7b2ff7 62%, #2d1b69 100%);
  box-shadow: 0 12px 40px rgba(123,47,247,.35);
}
.hero h1 { margin: 0; font-size: 2.5rem; font-weight: 800; color: #fff; letter-spacing: -.5px; }
.hero p { margin: 6px 0 0; color: #ffe9f0; font-size: 1.02rem; }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }
.stat {
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px; padding: 14px 18px;
}
.stat b { display: block; font-size: 1.5rem; color: #fff; font-weight: 700; }
.stat span { font-size: .72rem; color: #b9b9cf; text-transform: uppercase; letter-spacing: .1em; }

.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,.06); border-radius: 12px; padding: 10px 20px; color: #d8d8ea;
}
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #e50914, #7b2ff7); color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stWidgetLabel"] p { color: #cfcfe6; font-weight: 600; font-size: .82rem; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(90deg, #e50914, #7b2ff7); color: #fff; border: 0;
  border-radius: 12px; padding: .6rem 1.5rem; font-weight: 600; transition: all .2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  filter: brightness(1.12); transform: translateY(-1px); color: #fff;
}

.rh { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 22px 0 4px; }
.rh h3 { margin: 0; font-size: 1.45rem; font-weight: 700; color: #fff; }
.rh span { color: #a9a9c4; font-size: .9rem; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 18px; margin: 12px 0 18px; }
.card {
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.09);
  border-radius: 16px; overflow: hidden; display: flex; flex-direction: column;
  transition: transform .2s, box-shadow .2s;
}
.card:hover { transform: translateY(-6px); box-shadow: 0 14px 34px rgba(229,9,20,.28); }
.poster {
  position: relative; aspect-ratio: 2 / 3; background-size: cover; background-position: center;
  display: flex; align-items: center; justify-content: center;
}
.ph { font-size: 3.2rem; font-weight: 800; color: rgba(255,255,255,.92); text-align: center; }
.ph small { display: block; font-size: 1.5rem; margin-top: 4px; opacity: .85; }
.badge {
  position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,.72); color: #ffd166;
  font-weight: 700; font-size: .78rem; padding: 4px 10px; border-radius: 999px;
}
.rank {
  position: absolute; top: 10px; left: 10px; background: #e50914; color: #fff;
  font-weight: 700; font-size: .74rem; padding: 4px 10px; border-radius: 999px;
}
.info { padding: 12px 14px 15px; }
.ttl { font-weight: 700; font-size: 1rem; color: #fff; line-height: 1.25; margin-bottom: 3px; }
.meta { font-size: .78rem; color: #a9a9c4; margin-bottom: 8px; }
.chips span {
  display: inline-block; font-size: .68rem; padding: 2px 9px; margin: 0 4px 4px 0; border-radius: 999px;
  background: rgba(123,47,247,.25); color: #d9c7ff; border: 1px solid rgba(123,47,247,.45);
}
.bar { height: 6px; background: rgba(255,255,255,.12); border-radius: 99px; margin: 8px 0 4px; }
.bar i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #00d2ff, #7b2ff7); }
.barlbl { font-size: .72rem; color: #9fe3ff; }
.ov { font-size: .76rem; color: #b4b4cc; line-height: 1.45; margin-top: 7px; }

.empty {
  margin: 14px 0; padding: 22px; border-radius: 16px; text-align: center; color: #cfcfe6;
  background: rgba(255,255,255,.05); border: 1px dashed rgba(255,255,255,.2);
}
.foot { text-align: center; color: #7d7da0; font-size: .8rem; margin: 34px 0 6px; }
@media (max-width: 760px) { .stats { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 1.8rem; } }
"""
inject_css(CSS)

if not MODEL_PATH.exists():
    st.error("Model not built yet. In the terminal run:")
    st.code("python -m src.preprocess\npython -m src.recommender")
    st.stop()

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
try:
    movies, _ = load_model()
except Exception:
    # model.pkl was saved by a different pandas/Python version: rebuild it here
    from src.recommender import build_model

    try:
        with st.spinner("Rebuilding the model for this Python version (one-time)..."):
            build_model()
        load_model.cache_clear()
        movies, _ = load_model()
    except Exception as err:
        st.error(f"Could not load or rebuild the model: {err}")
        st.code("python -m src.preprocess\npython -m src.recommender")
        st.stop()
info = movies.drop_duplicates("id").set_index("id")
all_genres = get_genres()
min_year, max_year = get_year_range()
YEAR_MODES = ["Any", "Before", "After", "Between"]
RESULT_COLS = ["id", "title", "genres", "vote_average", "release_date"]


@st.cache_data(show_spinner=False, ttl=86400)
def poster_for(movie_id):
    """Poster URL from TMDB (needs TMDB_API_KEY), else None."""
    return fetch_poster(movie_id)


def esc(text):
    """HTML-escape, and stop Streamlit treating $...$ as maths."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def in_filters(df, f):
    mask = pd.Series(True, index=df.index)
    if f["year_min"] is not None:
        mask &= df["year"] >= f["year_min"]
    if f["year_max"] is not None:
        mask &= df["year"] <= f["year_max"]
    if f["min_rating"]:
        mask &= df["vote_average"] >= f["min_rating"]
    return mask


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
html_block(
    '<div class="hero"><h1>🎬 CineMatch</h1>'
    "<p>Movie Recommendation System · content-based filtering with TF-IDF and cosine similarity</p></div>")
html_block(
    '<div class="stats">'
    f'<div class="stat"><b>{len(movies):,}</b><span>Movies</span></div>'
    f'<div class="stat"><b>{len(all_genres)}</b><span>Genres</span></div>'
    f'<div class="stat"><b>{min_year} – {max_year}</b><span>Release years</span></div>'
    '<div class="stat"><b>TF-IDF + Cosine</b><span>Algorithm</span></div>'
    "</div>")


# --------------------------------------------------------------------------
# Reusable UI pieces
# --------------------------------------------------------------------------
def filter_controls(prefix, col_mode, col_year, col_rating):
    """Year + rating filters inside three given columns -> dict of filters."""
    year_min = year_max = None

    with col_mode:
        mode = st.selectbox("Release year", YEAR_MODES, key=f"{prefix}_year_mode")

    with col_year:
        if mode == "Before":
            y = st.number_input("Before year", 1900, 2100, 2000, step=1, key=f"{prefix}_before")
            year_max = int(y) - 1
        elif mode == "After":
            y = st.number_input("After year", 1900, 2100, 2000, step=1, key=f"{prefix}_after")
            year_min = int(y) + 1
        elif mode == "Between":
            year_min, year_max = st.slider(
                "Years", min_year, max_year, (min_year, max_year), key=f"{prefix}_between"
            )
        else:
            st.number_input("Year", 1900, 2100, 2000, disabled=True, key=f"{prefix}_off")

    with col_rating:
        min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, 0.5, key=f"{prefix}_rating")

    if (year_min is not None and year_min > max_year) or (
        year_max is not None and year_max < min_year
    ):
        st.warning(f"No movies in that range: the dataset only has {min_year}-{max_year}.")

    return dict(year_min=year_min, year_max=year_max, min_rating=min_rating)


def card_html(rank, row, kind):
    title = str(row["title"])
    mid = row["id"]

    url = poster_for(int(mid))
    if url:
        poster_open = f'<div class="poster" style="background-image:url(\'{esc(url)}\')">'
        placeholder = ""
    else:
        hue = sum(ord(c) for c in title) % 360
        poster_open = (
            f'<div class="poster" style="background:linear-gradient(160deg,'
            f'hsl({hue},65%,38%),hsl({(hue + 55) % 360},70%,16%))">'
        )
        initials = "".join(w[0] for w in title.split()[:2]).upper() or "?"
        placeholder = f'<div class="ph">{esc(initials)}<small>🎬</small></div>'

    year = "n/a" if pd.isna(row["release_date"]) else str(row["release_date"])[:4]
    rating = f"{float(row['vote_average']):.1f}"
    genres = [g for g in str(row["genres"]).split(", ") if g and g != "nan"][:3]
    chips = "".join(f"<span>{esc(g)}</span>" for g in genres)

    overview = info["overview"].get(mid, "")
    overview = "" if pd.isna(overview) else " ".join(str(overview).split())
    if len(overview) > 120:
        overview = overview[:117].rstrip() + "…"

    bar = ""
    if kind == "sim":
        pct = int(round(float(row["similarity"]) * 100))
        bar = f'<div class="bar"><i style="width:{max(pct, 3)}%"></i></div><div class="barlbl">Match {pct}%</div>'
    elif kind == "gen":
        score = float(row["score"])
        bar = f'<div class="bar"><i style="width:{max(int(score * 10), 3)}%"></i></div><div class="barlbl">Score {score:.2f}</div>'

    return (
        '<div class="card">'
        f"{poster_open}<span class=\"rank\">#{rank}</span><span class=\"badge\">⭐ {rating}</span>{placeholder}</div>"
        f'<div class="info"><div class="ttl">{esc(title)}</div><div class="meta">📅 {year}</div>'
        f'<div class="chips">{chips}</div>{bar}<div class="ov">{esc(overview)}</div></div>'
        "</div>"
    )


def store(key, title, sub, df, kind):
    st.session_state[key] = dict(title=title, sub=sub, df=df, kind=kind)


def render_results(key):
    data = st.session_state.get(key)
    if not data:
        return
    df, kind = data["df"], data["kind"]

    html_block(
        f'<div class="rh"><h3>{esc(data["title"])}</h3><span>{esc(data["sub"])}</span></div>')
    if df.empty:
        html_block(
            '<div class="empty">😕 No movies match these filters.<br>'
            "Try widening the year range or lowering the minimum rating "
            f"(dataset covers {min_year}-{max_year}).</div>")
        return

    cards = "".join(card_html(i, row, kind) for i, (_, row) in enumerate(df.iterrows(), start=1))
    html_block(f'<div class="grid">{cards}</div>')

    csv = df.drop(columns=["id"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download this list (CSV)", csv, "recommendations.csv", "text/csv", key=f"dl_{key}")


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_similar, tab_genre, tab_search = st.tabs(
    ["🎯 Similar to a movie", "🏷️ Top picks by genre", "🔎 Search"]
)
COLS = [3, 1.3, 1.5, 1.7, 1.5]

# ---- 1. Similar to a movie (content-based filtering) ----
with tab_similar:
    titles = get_titles()
    default_idx = titles.index("Avatar") if "Avatar" in titles else 0
    c1, c2, c3, c4, c5 = st.columns(COLS)
    with c1:
        selected = st.selectbox("Pick a movie you like", titles, index=default_idx, key="movie")
    flt = filter_controls("sim", c2, c3, c4)
    with c5:
        count = st.slider("How many", 5, 30, 20, key="count_similar")

    if st.button("🎬 Recommend", key="btn_similar"):
        try:
            with st.spinner("Finding similar movies..."):
                res = recommend(selected, n=count, **flt)
            store("res_sim", f"Because you liked {selected}",
                  f"{len(res)} movies ranked by content similarity", res, "sim")
        except ValueError as err:
            st.error(str(err))
    render_results("res_sim")

# ---- 2. Top picks by genre ----
with tab_genre:
    c1, c2, c3, c4, c5 = st.columns(COLS)
    with c1:
        chosen = st.multiselect("Genres (movie must match all)", all_genres, key="genres")
    flt = filter_controls("gen", c2, c3, c4)
    with c5:
        count = st.slider("How many", 5, 30, 20, key="count_genre")
    sort_by = st.radio(
        "Sort by",
        ["Best rated", "Most popular", "Newest first", "Random"],
        horizontal=True,
        key="sort_genre",
    )

    if st.button("🏆 Show top movies", key="btn_genre"):
        if not chosen:
            st.warning("Pick at least one genre.")
        else:
            try:
                res = recommend_by_genre(chosen, n=5000, **flt)
                res["vote_count"] = res["id"].map(info["vote_count"])
                if sort_by == "Most popular":
                    res = res.sort_values("vote_count", ascending=False)
                elif sort_by == "Newest first":
                    res = res.sort_values("release_date", ascending=False)
                elif sort_by == "Random":
                    res = res.sample(frac=1)  # shuffle: click again for a different set
                res = res.head(count).reset_index(drop=True)
                order = "random order" if sort_by == "Random" else f"sorted by {sort_by.lower()}"
                store("res_gen", f"Top {' + '.join(chosen)} movies",
                      f"{len(res)} movies · {order}", res, "gen")
            except (ValueError, RuntimeError) as err:
                st.error(str(err))
    render_results("res_gen")

# ---- 3. Search by title, actor, director or keyword ----
with tab_search:
    c1, c2, c3, c4, c5 = st.columns(COLS)
    with c1:
        query = st.text_input(
            "Search title, actor, director or keyword",
            placeholder="e.g. Nolan, Tom Hanks, heist, space",
            key="query",
        )
    flt = filter_controls("srch", c2, c3, c4)
    with c5:
        count = st.slider("How many", 5, 30, 20, key="count_search")

    if st.button("🔎 Search", key="btn_search"):
        q = query.strip().lower()
        if not q:
            st.warning("Type something to search.")
        else:
            title_hit = movies["title"].str.lower().str.contains(q, regex=False)
            # names are stored squashed in tags ("tomhanks"), so squash the query too
            tag_hit = movies["tags"].fillna("").str.contains(q.replace(" ", ""), regex=False)
            text_hit = movies["overview"].fillna("").str.lower().str.contains(q, regex=False)
            hits = movies[(title_hit | tag_hit | text_hit) & in_filters(movies, flt)].copy()
            hits["_title_match"] = title_hit[hits.index]
            hits = hits.sort_values(["_title_match", "vote_count"], ascending=[False, False])
            res = hits.head(count)[RESULT_COLS].reset_index(drop=True)
            store("res_search", f'Results for "{query.strip()}"',
                  f"{len(hits)} matches found, showing {len(res)}", res, "plain")
    render_results("res_search")

html_block(
    '<div class="foot">Built with Python · pandas · scikit-learn · Streamlit · '
    "Data: TMDB 5000 Movie Dataset</div>")
