import requests
import pandas as pd

NEWS_API_KEY = "358bd569ef384ccba8ca10b3cb9a82b2"


def fetch_news(query="stocks"):

    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}"
        "&language=en"
        "&sortBy=publishedAt"
        "&pageSize=50"
        f"&apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    rows = []

    for article in data.get("articles", []):

        rows.append({

            "headline": article.get("title", ""),

            "description": article.get(
                "description", ""
            ),

            "published_at": article.get(
                "publishedAt", ""
            ),

            "source": article.get(
                "source", {}
            ).get("name", ""),

            "url": article.get(
                "url", ""
            )
        })

    return pd.DataFrame(rows)