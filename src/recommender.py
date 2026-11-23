"""Step 2: content-based recommender (TF-IDF + cosine similarity).

Build the model once (from the project root):
    python -m src.recommender

Then use it:
    from src.recommender import recommend
    recommend("Avatar", n=5)
"""
import difflib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
MOVIES_CSV = ROOT / "data" / "processed" / "movies.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"


def build_model():
    """Vectorise every movie's tags and save the matrix to models/model.pkl."""
    movies = pd.read_csv(MOVIES_CSV)
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(movies["tags"].fillna(""))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"movies": movies, "matrix": matrix}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} ({matrix.shape[0]} movies, {matrix.shape[1]} features)")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python -m src.preprocess` and then "
            "`python -m src.recommender` first."
        )
    data = joblib.load(MODEL_PATH)
    movies = data["movies"].copy()
    movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    return movies, data["matrix"]


def get_titles():
    movies, _ = load_model()
    return sorted(movies["title"].tolist())


def get_year_range():
    """(earliest, latest) release year found in the dataset."""
    movies, _ = load_model()
    years = movies["year"].dropna()
    return int(years.min()), int(years.max())


def _filter_mask(movies, year_min=None, year_max=None, min_rating=0.0):
    """Boolean mask of movies that pass the release-year and rating filters.

    year_min / year_max are inclusive; leave as None to ignore.
    Movies with no release date are excluded only when a year filter is set.
    """
    mask = pd.Series(True, index=movies.index)
    if year_min is not None:
        mask &= movies["year"] >= year_min
    if year_max is not None:
        mask &= movies["year"] <= year_max
    if min_rating:
        mask &= movies["vote_average"] >= min_rating
    return mask


def recommend(title, n=5, year_min=None, year_max=None, min_rating=0.0):
    """Return a DataFrame of the n movies most similar to `title`.

    Optional filters: release year range (inclusive) and minimum rating.
    May return fewer than n rows (or none) if few movies pass the filters.
    """
    movies, matrix = load_model()

    matches = movies.index[movies["title"].str.lower() == title.strip().lower()]
    if len(matches) == 0:
        close = difflib.get_close_matches(title, movies["title"].tolist(), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Movie '{title}' not found.{hint}")

    idx = matches[0]
    scores = cosine_similarity(matrix[idx], matrix).ravel()

    allowed = _filter_mask(movies, year_min, year_max, min_rating).to_numpy().copy()
    allowed[idx] = False  # never recommend the movie itself
    candidates = np.flatnonzero(allowed)
    top = candidates[np.argsort(scores[candidates])[::-1][:n]]

    result = movies.iloc[top][["id", "title", "genres", "vote_average", "release_date"]].copy()
    result["similarity"] = scores[top].round(3)
    return result.reset_index(drop=True)


def get_genres():
    """Sorted list of every genre that appears in the dataset."""
    movies, _ = load_model()
    return sorted({g for gs in movies["genres"].fillna("") for g in gs.split(", ") if g})


def recommend_by_genre(genres, n=10, year_min=None, year_max=None, min_rating=0.0):
    """Top-rated movies that belong to ALL of the given genres.

    Optional filters: release year range (inclusive) and minimum rating.

    Uses the IMDB weighted rating so a movie with 3 votes can't beat a classic:
        score = v/(v+m) * R + m/(v+m) * C
    R = movie rating, v = its vote count, C = mean rating of all movies,
    m = minimum votes needed to be considered.
    """
    movies, _ = load_model()
    if "vote_count" not in movies.columns:
        raise RuntimeError(
            "Your data is from an older version. Re-run `python -m src.preprocess` "
            "and `python -m src.recommender`."
        )
    if isinstance(genres, str):
        genres = [genres]

    has_all = movies["genres"].fillna("").apply(
        lambda s: all(g in s.split(", ") for g in genres)
    )
    in_filter = _filter_mask(movies, year_min, year_max, min_rating)
    pool = movies[has_all & in_filter].copy()
    if pool.empty:
        raise ValueError(
            f"No movies found with genres: {', '.join(genres)} for the chosen year/rating filters."
        )

    C = movies["vote_average"].mean()
    m = movies["vote_count"].quantile(0.6)
    qualified = pool[pool["vote_count"] >= m]
    if qualified.empty:  # tiny genre combo: fall back to everything in the pool
        qualified = pool
    v, R = qualified["vote_count"], qualified["vote_average"]
    qualified = qualified.assign(score=(v / (v + m)) * R + (m / (v + m)) * C)

    result = qualified.sort_values("score", ascending=False).head(n)
    result = result[["id", "title", "genres", "vote_average", "release_date", "score"]]
    result["score"] = result["score"].round(2)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    build_model()
