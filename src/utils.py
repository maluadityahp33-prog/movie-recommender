"""Small helpers, e.g. fetching posters from the TMDB API (optional)."""
import os

import requests


def fetch_poster(movie_id):
    """Return a poster URL for a TMDB movie id, or None.

    Needs a free API key from https://www.themoviedb.org/settings/api
    Set it as an environment variable:  TMDB_API_KEY=your_key
    """
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        path = resp.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{path}" if path else None
    except (requests.RequestException, ValueError):
        return None
