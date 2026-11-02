"""Content-based recommender (TF-IDF + cosine similarity) and genre-based picks."""
import difflib
from functools import lru_cache
from pathlib import Path

import joblib
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
    return data["movies"], data["matrix"]


def get_titles():
    movies, _ = load_model()
    return sorted(movies["title"].tolist())


def recommend(title, n=5):
    """Return a DataFrame of the n movies most similar to `title`."""
    movies, matrix = load_model()

    matches = movies.index[movies["title"].str.lower() == title.strip().lower()]
    if len(matches) == 0:
        close = difflib.get_close_matches(title, movies["title"].tolist(), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Movie '{title}' not found.{hint}")

    idx = matches[0]
    scores = cosine_similarity(matrix[idx], matrix).ravel()
    top = scores.argsort()[::-1]
    top = [i for i in top if i != idx][:n]

    result = movies.iloc[top][["id", "title", "genres", "vote_average", "release_date"]].copy()
    result["similarity"] = scores[top].round(3)
    return result.reset_index(drop=True)


def get_genres():
    """Sorted list of every genre that appears in the dataset."""
    movies, _ = load_model()
    return sorted({g for gs in movies["genres"].fillna("") for g in gs.split(", ") if g})


def recommend_by_genre(genres, n=10):
    """Top-rated movies that belong to ALL of the given genres.

    Uses the IMDB weighted rating so a movie with 3 votes can't beat a classic:
        score = v/(v+m) * R + m/(v+m) * C
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
    pool = movies[has_all].copy()
    if pool.empty:
        raise ValueError(f"No movies found with genres: {', '.join(genres)}")

    C = movies["vote_average"].mean()
    m = movies["vote_count"].quantile(0.6)
    qualified = pool[pool["vote_count"] >= m]
    if qualified.empty:
        qualified = pool
    v, R = qualified["vote_count"], qualified["vote_average"]
    qualified = qualified.assign(score=(v / (v + m)) * R + (m / (v + m)) * C)

    result = qualified.sort_values("score", ascending=False).head(n)
    result = result[["id", "title", "genres", "vote_average", "release_date", "score"]]
    result["score"] = result["score"].round(2)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    build_model()