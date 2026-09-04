"""Tests for MultiAssetTradingEnv."""

import numpy as np
import pytest

from src.envs.multi_asset_env import MultiAssetTradingEnv, ASSETS


class TestEnvInit:
    def test_action_space_size(self, env):
        assert env.action_space.n == 17

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (57,)

    def test_initial_cash(self, env):
        assert env.cash == 100_000.0

    def test_initial_inventory_zero(self, env):
        for asset in ASSETS:
            assert env.inventory[asset] == 0.0

    def test_assets_list_length(self):
        assert len(ASSETS) == 8

    def test_all_8_assets_present(self):
        expected = {"BTC", "ETH", "ADA", "AAPL", "AMZN", "GOOG", "INTC", "MSFT"}
        assert set(ASSETS) == expected


class TestEnvReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_obs_shape_after_reset(self, env):
        obs, _ = env.reset()
        assert obs.shape == (57,)

    def test_obs_dtype_float32(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_no_nan_in_obs(self, env):
        obs, _ = env.reset()
        assert not np.isnan(obs).any()

    def test_no_inf_in_obs(self, env):
        obs, _ = env.reset()
        assert not np.isinf(obs).any()

    def test_cash_reset_to_initial(self, env):
        env.reset()
        env.step(1)
        env.reset()
        assert env.cash == env.initial_cash

    def test_inventory_reset_to_zero(self, env):
        env.reset()
        env.step(1)
        env.reset()
        for asset in ASSETS:
            assert env.inventory[asset] == 0.0

    def test_trades_reset_to_zero(self, env):
        env.reset()
        env.step(1)
        env.reset()
        assert env.total_trades == 0

    def test_wins_losses_reset(self, env):
        env.reset()
        assert env.wins == 0
        assert env.losses == 0


class TestEnvStep:
    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_hold_does_not_change_cash(self, env):
        env.reset()
        cash_before = env.cash
        env.step(0)
        assert env.cash == cash_before

    def test_hold_does_not_change_inventory(self, env):
        env.reset()
        env.step(0)
        for asset in ASSETS:
            assert env.inventory[asset] == 0.0

    def test_buy_btc_reduces_cash(self, env):
        env.reset()
        cash_before = env.cash
        env.step(1)  # BUY BTC
        assert env.cash < cash_before

    def test_buy_btc_increases_inventory(self, env):
        env.reset()
        env.step(1)  # BUY BTC
        assert env.inventory["BTC"] > 0

    def test_sell_without_inventory_is_invalid(self, env):
        env.reset()
        _, _, _, _, info = env.step(2)  # SELL BTC with no inventory
        assert info["invalid_action"] is True

    def test_buy_then_sell_updates_trade_count(self, env):
        env.reset()
        env.step(1)  # BUY BTC
        env.step(2)  # SELL BTC
        assert env.total_trades == 2

    def test_reward_is_finite(self, env):
        env.reset()
        for action in [0, 1, 0, 2, 0]:
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_obs_no_nan_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert not np.isnan(obs).any()

    def test_obs_no_inf_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert not np.isinf(obs).any()

    def test_info_contains_required_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        required = [
            "portfolio_value", "cash", "inventory", "realised_pnl",
            "total_trades", "wins", "losses", "win_rate_pct",
            "drawdown", "trade_executed", "invalid_action",
        ]
        for key in required:
            assert key in info, f"Missing info key: {key}"

    def test_portfolio_value_positive(self, env):
        env.reset()
        for _ in range(10):
            _, _, _, _, info = env.step(0)
        assert info["portfolio_value"] > 0

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10_000:
            _, _, done, _, _ = env.step(0)
            steps += 1
        assert done


class TestEnvReward:
    def test_invalid_action_gives_negative_reward(self, env):
        env.reset()
        # Sell with no inventory
        _, reward, _, _, info = env.step(2)
        assert info["invalid_action"]
        assert reward < 0

    def test_portfolio_summary_structure(self, env):
        env.reset()
        summary = env.get_portfolio_summary()
        required = [
            "portfolio_value", "cash", "pnl", "pnl_pct",
            "win_rate", "total_trades", "drawdown", "inventory",
        ]
        for key in required:
            assert key in summary

    def test_action_name_hold(self, env):
        assert env.action_name(0) == "HOLD"

    def test_action_name_buy_btc(self, env):
        assert env.action_name(1) == "BUY BTC"

    def test_action_name_sell_btc(self, env):
        assert env.action_name(2) == "SELL BTC"

    def test_action_name_buy_msft(self, env):
        assert env.action_name(15) == "BUY MSFT"

    def test_action_name_sell_msft(self, env):
        assert env.action_name(16) == "SELL MSFT"


class TestEnvMultipleAssets:
    def test_can_buy_multiple_assets(self, env):
        env.reset()
        env.step(1)   # BUY BTC
        env.step(3)   # BUY ETH
        env.step(5)   # BUY ADA
        assert env.inventory["BTC"] > 0
        assert env.inventory["ETH"] > 0
        assert env.inventory["ADA"] > 0

    def test_buy_all_8_assets(self, env):
        env.reset()
        buy_actions = [1, 3, 5, 7, 9, 11, 13, 15]
        for action in buy_actions:
            env.step(action)
        assets_with_inventory = sum(
            1 for a in ASSETS if env.inventory[a] > 0
        )
        assert assets_with_inventory > 0

    def test_win_rate_calculation(self, env):
        env.reset()
        for _ in range(50):
            env.step(0)
        _, _, _, _, info = env.step(0)
        assert 0 <= info["win_rate_pct"] <= 100