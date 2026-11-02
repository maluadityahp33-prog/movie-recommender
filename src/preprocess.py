"""Step 1: clean the raw TMDB data and build a single 'tags' text column.

Run from the project root:
    python -m src.preprocess
"""
import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _parse(text):
    """Turn a JSON-like string from the CSV into a Python list (or [])."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def _names(text, limit=None):
    names = [item["name"] for item in _parse(text)]
    return names[:limit] if limit else names


def _director(text):
    for member in _parse(text):
        if member.get("job") == "Director":
            return [member["name"]]
    return []


def _squash(words):
    """'Science Fiction' -> 'ScienceFiction' so multi-word names stay one token."""
    return [w.replace(" ", "") for w in words]


def build_movies():
    movies = pd.read_csv(RAW_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(RAW_DIR / "tmdb_5000_credits.csv")

    credits = credits.rename(columns={"movie_id": "id"})[["id", "cast", "crew"]]
    df = movies.merge(credits, on="id")

    df = df[
        ["id", "title", "overview", "genres", "keywords", "cast", "crew",
         "vote_average", "vote_count", "release_date"]
    ].dropna(subset=["title"])
    df["overview"] = df["overview"].fillna("")

    genres = df["genres"].apply(_names)
    keywords = df["keywords"].apply(_names)
    cast = df["cast"].apply(lambda x: _names(x, limit=3))
    director = df["crew"].apply(_director)

    df["tags"] = (
        df["overview"].str.split()
        + genres.apply(_squash)
        + keywords.apply(_squash)
        + cast.apply(_squash)
        + director.apply(_squash)
    ).apply(lambda words: " ".join(words).lower())

    df["genres"] = genres.apply(lambda g: ", ".join(g))

    out = df[["id", "title", "genres", "overview", "vote_average", "vote_count",
              "release_date", "tags"]].drop_duplicates(subset="title")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    print(f"Saved {len(out)} movies to {PROCESSED_DIR / 'movies.csv'}")


if __name__ == "__main__":
    build_movies()
