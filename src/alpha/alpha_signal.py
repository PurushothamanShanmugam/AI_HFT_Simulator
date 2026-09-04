from src.alpha.alpha_engine import AlphaEngine
from src.regime.regime_detector import RegimeDetector


class AlphaSignal:

    def __init__(self):
        self.alpha_engine = AlphaEngine()
        self.regime_detector = RegimeDetector()

    def generate(
        self,
        row,
        sentiment_score,
        sentiment_confidence,
    ):

        trend = float(
            row.get(
                "f_trend_strength",
                abs(float(row.get("returns", 0.0)))
            )
        )

        alpha = self.alpha_engine.score(
            sentiment=float(sentiment_score),
            confidence=float(sentiment_confidence),
            imbalance=float(row.get("imbalance", 0.0)),
            pressure=float(
                row.get(
                    "microstructure_pressure",
                    0.0
                )
            ),
            trend=trend,
        )

        signal = self.alpha_engine.signal(alpha)

        regime = self.regime_detector.detect(row)

        return {
            "alpha": alpha,
            "signal": signal,
            "regime": regime,
        }