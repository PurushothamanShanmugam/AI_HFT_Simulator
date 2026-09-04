class AlphaEngine:
    """
    Alpha score generator.

    Combines:
    - News sentiment
    - Order book imbalance
    - Microstructure pressure
    - Trend strength
    """

    def score(
        self,
        sentiment,
        confidence,
        imbalance,
        pressure,
        trend
    ):

        alpha = (
            0.30 * sentiment +
            0.25 * imbalance +
            0.25 * pressure +
            0.20 * trend
        )

        alpha *= confidence

        return float(alpha)

    def signal(self, alpha):

        if alpha > 0.25:
            return "BUY"

        elif alpha < -0.25:
            return "SELL"

        return "HOLD"