from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_FILE = (
    ROOT /
    "data" /
    "news" /
    "asset_sentiment.csv"
)

DEFAULT_SENTIMENT = {
    "sentiment_score": 0.0,
    "confidence": 0.5,
    "positive": 0.33,
    "negative": 0.33,
    "neutral": 0.34,
    "article_count": 0
}


class LiveSentimentProvider:

    def __init__(self):

        self.sentiment = {}

        self.load()

    def load(self):

        self.sentiment = {}

        if not SENTIMENT_FILE.exists():

            print(
                f"[WARNING] Missing sentiment file: "
                f"{SENTIMENT_FILE}"
            )

            return

        try:

            df = pd.read_csv(
                SENTIMENT_FILE
            )

            for _, row in df.iterrows():

                asset = str(
                    row["asset"]
                ).upper()

                self.sentiment[
                    asset
                ] = {

                    "sentiment_score":
                        float(
                            row[
                                "sentiment_score"
                            ]
                        ),

                    "confidence":
                        float(
                            row[
                                "confidence"
                            ]
                        ),

                    "positive":
                        float(
                            row[
                                "positive"
                            ]
                        ),

                    "negative":
                        float(
                            row[
                                "negative"
                            ]
                        ),

                    "neutral":
                        float(
                            row[
                                "neutral"
                            ]
                        ),

                    "article_count":
                        int(
                            row[
                                "article_count"
                            ]
                        )
                }

            print(
                f"[INFO] Loaded "
                f"{len(self.sentiment)} "
                f"sentiment assets"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"Sentiment load failed: "
                f"{e}"
            )

    def get(self, asset):

        asset = str(
            asset
        ).upper()

        return self.sentiment.get(
            asset,
            DEFAULT_SENTIMENT
        )

    def get_score(self, asset):

        return self.get(
            asset
        )["sentiment_score"]

    def get_confidence(self, asset):

        return self.get(
            asset
        )["confidence"]

    def reload(self):

        self.load()


if __name__ == "__main__":

    provider = (
        LiveSentimentProvider()
    )

    for asset in [
        "BTC",
        "ETH",
        "ADA",
        "AAPL",
        "AMZN",
        "GOOG",
        "INTC",
        "MSFT"
    ]:

        print(
            asset,
            provider.get(asset)
        )