"""
Risk Metrics — Fixed & Extended
=================================
Changes from original:
  1. Sharpe ratio now annualised (× sqrt(252)) with configurable periods_per_year.
  2. Added annualised_return().
  3. Added calmar_ratio() (annualised return / max drawdown).
  4. Added sortino_ratio() (uses downside deviation only).
  5. Added value_at_risk() (parametric and historical).
  6. All functions accept periods_per_year for correct annualisation.
"""

import numpy as np


def safe_float(value, default=0.0):
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def portfolio_returns(equity_curve):
    """Convert portfolio values into period-on-period percentage returns."""
    values = np.array([safe_float(x) for x in equity_curve], dtype=float)

    if len(values) < 2:
        return np.array([])

    prev = values[:-1]
    curr = values[1:]
    denom = np.where(prev == 0, 1, prev)
    return (curr - prev) / denom


def annualised_return(equity_curve, periods_per_year: int = 252) -> float:
    """
    Compound annual growth rate (CAGR).

    periods_per_year = 252 for daily, 252×390 for minute-level, etc.
    For a simulation, 252 is a reasonable default.
    """
    values = np.array([safe_float(x) for x in equity_curve], dtype=float)

    if len(values) < 2 or values[0] == 0:
        return 0.0

    total_return = values[-1] / values[0]
    n_years = len(values) / periods_per_year
    cagr = total_return ** (1.0 / max(n_years, 1e-9)) - 1.0
    return round(float(cagr * 100), 4)  # as percentage


def sharpe_ratio(
    equity_curve, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """
    Annualised Sharpe ratio.

    sharpe = (mean_return - risk_free_rate) / std_return × sqrt(periods_per_year)

    risk_free_rate is the per-period risk-free rate (e.g. 0.05/252 for daily).
    Default 0.0 gives excess-return Sharpe.
    """
    returns = portfolio_returns(equity_curve)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess = returns - risk_free_rate
    sr = float(np.mean(excess) / np.std(excess)) * np.sqrt(periods_per_year)
    return round(sr, 4)


def sortino_ratio(
    equity_curve, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """
    Sortino ratio — like Sharpe but penalises only downside volatility.
    """
    returns = portfolio_returns(equity_curve)

    if len(returns) == 0:
        return 0.0

    excess = returns - risk_free_rate
    downside = excess[excess < 0]

    # No downside returns → no downside risk → ratio is undefined, return 0
    if len(downside) == 0:
        return 0.0

    downside_std = float(np.std(downside))

    if downside_std == 0:
        return 0.0

    sr = float(np.mean(excess) / downside_std) * np.sqrt(periods_per_year)
    return round(sr, 4)


def max_drawdown(equity_curve) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    values = np.array([safe_float(x) for x in equity_curve], dtype=float)

    if len(values) == 0:
        return 0.0

    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / np.where(peak == 0, 1, peak)
    return round(float(drawdown.min() * 100), 2)


def calmar_ratio(equity_curve, periods_per_year: int = 252) -> float:
    """
    Calmar ratio = annualised return / |max drawdown|.
    Higher is better; negative drawdown denominator is made positive.
    """
    ann_ret = annualised_return(equity_curve, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))

    if mdd == 0:
        return 0.0

    return round(ann_ret / mdd, 4)


def portfolio_volatility(equity_curve, periods_per_year: int = 252) -> float:
    """Annualised portfolio volatility as a percentage."""
    returns = portfolio_returns(equity_curve)

    if len(returns) == 0:
        return 0.0

    return round(float(np.std(returns) * np.sqrt(periods_per_year) * 100), 4)


def value_at_risk(
    equity_curve, confidence: float = 0.95, method: str = "historical"
) -> float:
    """
    Value at Risk at given confidence level.

    method='historical' : empirical percentile of returns.
    method='parametric' : Gaussian approximation.

    Returns VaR as a positive number (loss magnitude as % of portfolio).
    """
    returns = portfolio_returns(equity_curve)

    if len(returns) == 0:
        return 0.0

    if method == "historical":
        var = float(np.percentile(returns, (1 - confidence) * 100))
    else:
        mu = float(np.mean(returns))
        sig = float(np.std(returns))
        # Approximate using normal distribution z-score
        from scipy import stats  # type: ignore

        z = stats.norm.ppf(1 - confidence)
        var = mu + z * sig

    return round(abs(var) * 100, 4)  # as percentage


def win_rate(trades) -> float:
    """Percentage of closed trades that made profit."""
    closed = [
        t
        for t in trades
        if str(t.get("trade_result", "")).lower() in ["profit", "loss"]
    ]

    if not closed:
        return 0.0

    wins = sum(1 for t in closed if str(t.get("trade_result", "")).lower() == "profit")
    return round(wins / len(closed) * 100, 2)


def profit_factor(trades) -> float:
    """Total gross profit / total gross loss. >1 means profitable overall."""
    profits = [
        safe_float(t.get("realized_pnl", 0))
        for t in trades
        if safe_float(t.get("realized_pnl", 0)) > 0
    ]
    losses = [
        abs(safe_float(t.get("realized_pnl", 0)))
        for t in trades
        if safe_float(t.get("realized_pnl", 0)) < 0
    ]

    total_profit = sum(profits)
    total_loss = sum(losses)

    if total_loss == 0:
        return round(total_profit, 4) if total_profit > 0 else 0.0

    return round(total_profit / total_loss, 4)


def average_trade_pnl(trades) -> float:
    """Average realised P&L per closed trade."""
    closed_pnl = [
        safe_float(t.get("realized_pnl", 0))
        for t in trades
        if str(t.get("trade_result", "")).lower() in ["profit", "loss"]
    ]

    if not closed_pnl:
        return 0.0

    return round(float(np.mean(closed_pnl)), 4)


def calculate_trader_risk_metrics(
    trader_id: str, portfolios: list, trades: list, periods_per_year: int = 252
) -> dict:
    """Calculate all risk metrics for one trader."""
    equity_curve = [
        safe_float(p.get(trader_id, 0.0)) for p in portfolios if trader_id in p
    ]
    trader_trades = [t for t in trades if t.get("tid") == trader_id]

    return {
        "annualised_return_pct": annualised_return(equity_curve, periods_per_year),
        "sharpe_ratio": sharpe_ratio(equity_curve, periods_per_year),
        "sortino_ratio": sortino_ratio(equity_curve, periods_per_year),
        "calmar_ratio": calmar_ratio(equity_curve, periods_per_year),
        "max_drawdown_pct": max_drawdown(equity_curve),
        "portfolio_volatility_pct": portfolio_volatility(
            equity_curve, periods_per_year
        ),
        "win_rate_pct": win_rate(trader_trades),
        "profit_factor": profit_factor(trader_trades),
        "avg_trade_pnl": average_trade_pnl(trader_trades),
    }


def calculate_all_risk_metrics(sim: dict, periods_per_year: int = 252) -> dict:
    """Calculate risk metrics for all traders in a simulation dict."""
    results = {}

    for trader in sim.get("traders", []):
        trader_id = trader["id"]
        results[trader_id] = calculate_trader_risk_metrics(
            trader_id=trader_id,
            portfolios=sim.get("portfolios", []),
            trades=sim.get("trades", []),
            periods_per_year=periods_per_year,
        )

    return results