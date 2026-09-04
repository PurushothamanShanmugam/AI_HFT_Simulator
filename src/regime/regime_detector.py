class RegimeDetector:
    """
    Market Regime Detector

    Outputs:
    - BULL
    - BEAR
    - SIDEWAYS
    - HIGH_VOL
    """

    def detect(self, row):

        trend = float(row.get("f_trend_strength", 0))
        vol = float(row.get("f_vol_regime", 1))
        spread = float(row.get("f_spread_regime", 1))

        # Strong trend, normal volatility
        if trend > 0.65 and vol < 1.5:
            return "BULL"

        # Strong trend, elevated volatility
        if trend > 0.65 and vol >= 1.5:
            return "HIGH_VOL_BULL"

        # Weak trend, calm market
        if trend < 0.20 and vol < 1.5:
            return "SIDEWAYS"

        # Volatility spike
        if vol > 2.0:
            return "HIGH_VOL"

        return "BEAR"