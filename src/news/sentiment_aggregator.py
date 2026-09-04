import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "sentiment_features.csv"

OUTPUT_DIR = ROOT / "data" / "news"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "asset_sentiment.csv"

ASSET_MAP = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOG",
    "Amazon": "AMZN",
    "Intel": "INTC",
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Cardano": "ADA"
}


def build_asset_sentiment():

    print("=" * 60)
    print("Building Asset Sentiment File")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(df):,} news records")

    df["asset"] = df["asset"].map(ASSET_MAP)

    df = df.dropna(subset=["asset"])

    agg = (
        df.groupby("asset")
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            confidence=("confidence", "mean"),
            positive=("positive", "mean"),
            negative=("negative", "mean"),
            neutral=("neutral", "mean"),
            article_count=("headline", "count")
        )
        .reset_index()
    )

    agg = agg.sort_values(
        "asset"
    ).reset_index(drop=True)

    print("\nAsset Sentiment Summary\n")
    print(agg)

    agg.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved:\n{OUTPUT_FILE}"
    )

    return agg


if __name__ == "__main__":
    build_asset_sentiment()