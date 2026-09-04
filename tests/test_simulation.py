"""Tests for simulation utilities and agent wrappers."""

import numpy as np
import pandas as pd
import pytest

from src.envs.multi_asset_env import MultiAssetTradingEnv, ASSETS
from src.data.feature_builder import build_features


def make_test_env(n=300):
    """Build a small test environment."""
    np.random.seed(7)
    prices  = 100.0 + np.random.normal(0, 1, n).cumsum()
    prices  = np.maximum(prices, 0.1)
    spread  = np.abs(np.random.normal(0.1, 0.02, n))
    returns = pd.Series(prices).pct_change().fillna(0).values
    vol     = pd.Series(returns).rolling(20, min_periods=2).std().fillna(0).values

    df = pd.DataFrame({
        "market_id":   "BTC_1sec",
        "asset":       "BTC",
        "source":      "CRYPTO",
        "resolution":  "1sec",
        "timestamp":   [str(i) for i in range(n)],
        "global_tick": np.arange(n),
        "mid_price":   prices,
        "spread":      spread,
        "spread_pct":  spread / prices,
        "returns":     returns,
        "volatility":  vol,
        "imbalance":   np.random.uniform(-0.5, 0.5, n),
        "bid_price_1": prices - spread / 2,
        "ask_price_1": prices + spread / 2,
        "bid_size_1":  np.full(n, 100.0),
        "ask_size_1":  np.full(n, 100.0),
    })
    return build_features(df)


class TestAgentSimulator:
    def test_hold_agent_runs_full_episode(self):
        """An agent that always HOLDs should complete the episode."""
        from src.simulation.multi_agent_sim import AgentSimulator

        class HoldAgent:
            name = "Hold"
            def predict(self, obs, env=None):
                return 0

        df  = make_test_env()
        sim = AgentSimulator("Hold", HoldAgent(), df)
        summary = sim.run()

        assert summary["portfolio_value"] > 0
        assert summary["total_trades"] == 0

    def test_random_agent_completes(self):
        """A random agent should complete the episode without crashing."""
        from src.simulation.multi_agent_sim import AgentSimulator
        np.random.seed(42)

        class RandomAgent:
            name = "Random"
            def predict(self, obs, env=None):
                return np.random.randint(0, 17)

        df  = make_test_env()
        sim = AgentSimulator("Random", RandomAgent(), df)
        summary = sim.run()

        assert summary is not None
        assert "portfolio_value" in summary
        assert "win_rate" in summary

    def test_portfolio_history_length(self):
        """Portfolio history should have one entry per step."""
        from src.simulation.multi_agent_sim import AgentSimulator

        class HoldAgent:
            name = "Hold"
            def predict(self, obs, env=None):
                return 0

        df  = make_test_env(n=100)
        sim = AgentSimulator("Hold", HoldAgent(), df)
        sim.run()

        assert len(sim.portfolio_history) > 0
        assert len(sim.portfolio_history) <= len(df)

    def test_buy_agent_trades(self):
        """An agent that always tries to BUY should execute some trades."""
        from src.simulation.multi_agent_sim import AgentSimulator

        class BuyAgent:
            name = "AlwaysBuy"
            def predict(self, obs, env=None):
                return 1  # BUY BTC

        df  = make_test_env()
        sim = AgentSimulator("AlwaysBuy", BuyAgent(), df)
        summary = sim.run()

        assert summary["total_trades"] > 0


class TestRLAgentWrapper:
    def test_rl_agent_returns_valid_action(self, multi_asset_df):
        """RLAgent wrapper should return integer action."""
        from src.simulation.multi_agent_sim import RLAgent

        class MockSB3Model:
            def predict(self, obs, deterministic=True):
                return np.array([5]), None

        agent = RLAgent("MockDQN", MockSB3Model())
        obs   = np.zeros(57, dtype=np.float32)
        action = agent.predict(obs)
        assert isinstance(action, int)
        assert 0 <= action <= 16

    def test_rl_agent_none_model_returns_hold(self):
        from src.simulation.multi_agent_sim import RLAgent
        agent  = RLAgent("NoneAgent", None)
        action = agent.predict(np.zeros(57))
        assert action == 0


class TestXGBoostAgentWrapper:
    def test_xgboost_none_model_returns_hold(self):
        from src.simulation.multi_agent_sim import XGBoostAgent
        agent  = XGBoostAgent("NoneXGB", None, None)
        action = agent.predict(np.zeros(57))
        assert action == 0


class TestLSTMAgentWrapper:
    def test_lstm_none_model_returns_hold(self):
        from src.simulation.multi_agent_sim import LSTMAgent
        agent  = LSTMAgent("NoneLSTM", None)
        action = agent.predict(np.zeros(57))
        assert action == 0

    def test_lstm_returns_hold_before_seq_filled(self):
        from src.simulation.multi_agent_sim import LSTMAgent
        agent = LSTMAgent("NoneLSTM", None)
        for _ in range(30):
            action = agent.predict(np.zeros(57))
        assert action == 0


class TestSyntheticLiveGenerator:
    def test_generates_correct_row_count(self):
        from src.data.synthetic_live_generator import generate_asset_ticks
        df = generate_asset_ticks("BTC", 56_000.0, n_total=100_000, seed=42)
        assert len(df) == 100_000

    def test_all_regimes_present(self):
        from src.data.synthetic_live_generator import generate_asset_ticks
        df = generate_asset_ticks("BTC", 56_000.0, n_total=400, seed=42)
        regimes = df["regime"].unique()
        assert "bull"     in regimes
        assert "flat"     in regimes
        assert "bear"     in regimes
        assert "volatile" in regimes

    def test_prices_all_positive(self):
        from src.data.synthetic_live_generator import generate_asset_ticks
        df = generate_asset_ticks("ADA", 1.17, n_total=200, seed=1)
        assert (df["mid_price"] > 0).all()

    def test_all_8_assets_generated(self):
        from src.data.synthetic_live_generator import (
            generate_kafka_test_data, REAL_OPEN_PRICES
        )
        df = generate_kafka_test_data(n_total=100, save=False)
        assets = df["asset"].unique()
        for asset in REAL_OPEN_PRICES.keys():
            assert asset in assets

    def test_kafka_data_has_required_columns(self):
        from src.data.synthetic_live_generator import generate_kafka_test_data
        df = generate_kafka_test_data(n_total=100, save=False)
        for col in ["mid_price", "spread", "imbalance",
                    "order_flow_imbalance", "asset", "regime"]:
            assert col in df.columns