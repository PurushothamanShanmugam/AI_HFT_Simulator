"""
Shared pytest fixtures for hft_ai_simulator tests.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data.feature_builder import get_observation_features

FEATURE_COLS = get_observation_features()


# ── Minimal LOB DataFrame ─────────────────────────────────────────────────

@pytest.fixture
def sample_crypto_df():
    """200 rows of synthetic BTC-like data."""
    np.random.seed(42)
    n = 200
    prices = 56_000 + np.random.normal(0, 50, n).cumsum()
    prices = np.maximum(prices, 1.0)
    spread = np.abs(np.random.normal(0.5, 0.1, n))
    returns = pd.Series(prices).pct_change().fillna(0).values
    vol     = pd.Series(returns).rolling(20, min_periods=2).std().fillna(0).values

    df = pd.DataFrame({
        "market_id":   "BTC_1sec",
        "asset":       "BTC",
        "source":      "CRYPTO",
        "resolution":  "1sec",
        "timestamp":   [f"2021-04-07T00:{i//60:02d}:{i%60:02d}" for i in range(n)],
        "global_tick": np.arange(n),
        "mid_price":   prices,
        "spread":      spread,
        "spread_pct":  spread / prices,
        "returns":     returns,
        "volatility":  vol,
        "imbalance":   np.random.uniform(-0.5, 0.5, n),
        "bid_price_1": prices - spread / 2,
        "ask_price_1": prices + spread / 2,
        "bid_size_1":  np.random.uniform(50, 200, n),
        "ask_size_1":  np.random.uniform(50, 200, n),
    })
    return df


@pytest.fixture
def sample_lobster_df():
    """200 rows of synthetic MSFT-like LOBSTER data."""
    np.random.seed(99)
    n = 200
    prices = 30.0 + np.random.normal(0, 0.05, n).cumsum()
    prices = np.maximum(prices, 0.01)
    spread = np.abs(np.random.normal(0.02, 0.005, n))
    returns = pd.Series(prices).pct_change().fillna(0).values
    vol     = pd.Series(returns).rolling(20, min_periods=2).std().fillna(0).values

    df = pd.DataFrame({
        "market_id":   "MSFT_LOBSTER10",
        "asset":       "MSFT",
        "source":      "LOBSTER",
        "resolution":  "LOBSTER10",
        "timestamp":   [str(i) for i in range(n)],
        "global_tick": np.arange(n),
        "mid_price":   prices,
        "spread":      spread,
        "spread_pct":  spread / prices,
        "returns":     returns,
        "volatility":  vol,
        "imbalance":   np.random.uniform(-0.3, 0.3, n),
        "bid_price_1": prices - spread / 2,
        "ask_price_1": prices + spread / 2,
        "bid_size_1":  np.random.uniform(100, 500, n),
        "ask_size_1":  np.random.uniform(100, 500, n),
    })
    return df


@pytest.fixture
def multi_asset_df(sample_crypto_df, sample_lobster_df):
    """Combined multi-asset DataFrame with features built."""
    from src.data.feature_builder import build_features
    frames = []
    for df in [sample_crypto_df, sample_lobster_df]:
        frames.append(build_features(df))
    combined = pd.concat(frames, ignore_index=True)
    return combined


@pytest.fixture
def env(multi_asset_df):
    """Fresh MultiAssetTradingEnv instance."""
    from src.envs.multi_asset_env import MultiAssetTradingEnv
    return MultiAssetTradingEnv(
        data                 = multi_asset_df,
        initial_cash         = 100_000.0,
        position_size_pct    = 0.10,
        max_inventory        = 10,
        transaction_cost_pct = 0.001,
        slippage_pct         = 0.0005,
    )