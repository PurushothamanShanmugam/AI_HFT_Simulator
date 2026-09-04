import pandas as pd

from news_fetcher import fetch_news
from sentiment_engine import get_sentiment

print("=== BUILDING SENTIMENT FEATURES ===")

ASSETS = [
    "Apple",
    "Microsoft",
    "Google",
    "Amazon",
    "Intel",
    "Bitcoin",
    "Ethereum",
    "Cardano"
]

sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

rows = []

for asset in ASSETS:

    print(f"\nProcessing {asset}...")

    try:

        news_df = fetch_news(asset)

        if news_df.empty:
            print(f"No news found for {asset}")
            continue

        print(f"Found {len(news_df)} articles")

        for _, row in news_df.iterrows():

            headline = str(
                row.get("headline", "")
            ).strip()

            if headline == "":
                continue

            result = get_sentiment(
                headline
            )

            rows.append({

                "asset": str(asset),

                "headline": headline,

                "published_at": str(
                    row.get(
                        "published_at",
                        ""
                    )
                ),

                "source": str(
                    row.get(
                        "source",
                        ""
                    )
                ),

                "url": str(
                    row.get(
                        "url",
                        ""
                    )
                ),

                "sentiment_label":
                    result["sentiment"],

                "sentiment_score":
                    sentiment_map[
                        result["sentiment"]
                    ],

                "confidence":
                    float(
                        result["confidence"]
                    ),

                "positive":
                    float(
                        result["positive"]
                    ),

                "negative":
                    float(
                        result["negative"]
                    ),

                "neutral":
                    float(
                        result["neutral"]
                    )
            })

    except Exception as e:

        print(
            f"ERROR processing {asset}: {e}"
        )

sentiment_df = pd.DataFrame(rows)

print("\n========================")
print("DATASET CREATED")
print("========================")

print(
    f"Rows: {len(sentiment_df)}"
)

print(
    f"Columns: {len(sentiment_df.columns)}"
)

print("\nData Types:")
print(sentiment_df.dtypes)

print("\nPreview:")
print(sentiment_df.head())

# Save as CSV first
output_file = "sentiment_features.csv"

sentiment_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nSUCCESS: Saved to {output_file}"
)